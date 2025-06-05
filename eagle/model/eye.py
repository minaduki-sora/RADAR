import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import json
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


with open("config.json", "r") as f:
    config = json.load(f)
hidden_dim = config["hidden_dim"]
num_classes = config["num_classes"]
train_batch_size = config["train_batch_size"]
eval_batch_size = config.get("eval_batch_size", train_batch_size)
epochs = config["epochs"]
learning_rate = config["learning_rate"]
dataset_type = config["dataset_type"]
data_files = config["data_files"]
save_model_path = config["save_model_path"]


class Hawkeye(nn.Module):
    def __init__(self, hidden_dim=4096, num_classes=9):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, last_hidden_state, egale_1st_forward_hidden):
        x = torch.cat([last_hidden_state, egale_1st_forward_hidden], dim=-1)  # [B, 2*hidden_dim]
        logits = self.classifier(x)  # [B, num_classes]
        return logits


class EagleDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        l_h = self.data[idx]["last_hidden_state"]
        e_h = self.data[idx]["egale_1st_forward_hidden"]
        label = self.data[idx]["accept_length"]
        l_h = torch.tensor(l_h, dtype=torch.float32)
        e_h = torch.tensor(e_h, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return l_h, e_h, label


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for last_h, egale_h, labels in tqdm(dataloader, desc="Train", leave=False):
        last_h = last_h.to(device)
        egale_h = egale_h.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(last_h, egale_h)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / total
    acc = total_correct / total
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    with torch.no_grad():
        for last_h, egale_h, labels in tqdm(dataloader, desc="Eval", leave=False):
            last_h = last_h.to(device)
            egale_h = egale_h.to(device)
            labels = labels.to(device)
            logits = model(last_h, egale_h)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    acc = total_correct / total
    return avg_loss, acc


def main():
    dataset = load_dataset(
        dataset_type,
        data_files=data_files
    )
    train_data = dataset["train"]
    test_data = dataset["test"]

    train_set = EagleDatasetTorch(train_data)
    test_set = EagleDatasetTorch(test_data)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hawkeye(hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_model_path)
    print("Best Test Acc:", best_acc)
    print(f"Best model saved to: {save_model_path}")

if __name__ == "__main__":
    main()