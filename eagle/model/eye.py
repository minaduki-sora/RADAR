"""Policy network used to control RADAR draft-tree generation."""

import torch
import torch.nn as nn


class RADAR(nn.Module):
    """LSTM policy that predicts whether drafting should continue or stop."""

    def __init__(
        self,
        state_dim=10,
        lstm_hidden=128,
        mlp_hidden=128,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden + state_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, state_seq, hidden=None):
        lstm_out, hidden = self.lstm(state_seq, hidden)
        logits = self.mlp(torch.cat((lstm_out, state_seq), dim=-1))
        return logits, hidden

    def act(self, state_seq, hidden=None, deterministic=False):
        logits, hidden = self.forward(state_seq, hidden)
        probabilities = torch.softmax(logits, dim=-1)
        if deterministic:
            actions = torch.argmax(probabilities, dim=-1)
        else:
            actions = torch.distributions.Categorical(probabilities).sample()
        return actions, probabilities, hidden

    def reset_hidden(self, batch_size):
        parameter = next(self.parameters())
        shape = (self.num_layers, batch_size, self.lstm_hidden)
        h0 = parameter.new_zeros(shape)
        c0 = parameter.new_zeros(shape)
        return h0, c0
