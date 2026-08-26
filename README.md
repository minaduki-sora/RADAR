# RADAR

Official implementation of **RADAR: Accelerate Large Language Model Inference with RL-Based Dynamic Draft Trees**, accepted to Findings of EMNLP 2026.

RADAR extends EAGLE-3 with a lightweight LSTM policy that decides whether to continue or stop draft-tree expansion at every drafting step. The policy is trained with offline reinforcement learning over acceptance-length distributions, reducing redundant draft-model calls while preserving lossless speculative decoding.

## Main results

The following speedup ratios are measured against autoregressive decoding at temperature 1.0. The main experiments use two NVIDIA RTX 3090 GPUs.

| Target model | Method | MT-bench | GSM8K | Alpaca | MBPP |
|---|---|---:|---:|---:|---:|
| LLaMA-3.1-Instruct 8B | EAGLE-2 | 2.56x | 3.43x | 2.89x | 3.29x |
| | EAGLE-3 | 3.08x | 4.68x | 3.86x | 4.21x |
| | EAGLE-3 + DISCO | 3.10x | 4.51x | 3.68x | 3.71x |
| | EAGLE-3 + SpecDec++ | 3.12x | 4.44x | 3.63x | 3.98x |
| | EAGLE-3 + LTD | 2.98x | 4.61x | 3.72x | 4.28x |
| | **EAGLE-3 + RADAR** | **3.41x** | **4.82x** | **4.04x** | **4.44x** |
| Vicuna 13B | EAGLE-2 | 2.89x | 3.18x | 2.83x | 3.56x |
| | EAGLE-3 | 3.74x | 4.24x | 3.50x | 4.55x |
| | EAGLE-3 + DISCO | 3.68x | 3.64x | 3.47x | 4.25x |
| | EAGLE-3 + SpecDec++ | 3.60x | 3.64x | 3.37x | 4.38x |
| | EAGLE-3 + LTD | 3.78x | 3.82x | 3.31x | 4.56x |
| | **EAGLE-3 + RADAR** | **4.05x** | **4.36x** | **3.84x** | **4.75x** |
| DeepSeek-R1-Distill-LLaMA 8B | EAGLE-3 | 3.42x | 4.39x | 3.08x | 3.71x |
| | EAGLE-3 + LTD | 3.56x | 4.15x | **3.39x** | 3.97x |
| | **EAGLE-3 + RADAR** | **3.86x** | **4.71x** | 3.17x | **3.99x** |

RADAR achieves a 3.17x--4.82x speedup and reduces the average number of draft-model calls by 18.7% relative to EAGLE-3's fixed eight calls. The paper contains acceptance lengths, ablations, larger-model results, and additional analysis.

## Installation

Python 3.9 or later and a CUDA-capable PyTorch environment are recommended.

```bash
git clone https://github.com/minaduki-sora/RADAR.git
cd RADAR
pip install -e .
```

Access to gated base models, such as Meta LLaMA, must be requested separately from the corresponding model provider.

## Required models

| Target model | Example base-model path | Example EAGLE-3 draft-model path |
|---|---|---|
| LLaMA-3.1-Instruct 8B | `/path/to/Meta-Llama-3.1-8B-Instruct` | `/path/to/EAGLE3-LLaMA3.1-Instruct-8B` |
| Vicuna 13B v1.3 | `/path/to/vicuna-13b-v1.3` | `/path/to/EAGLE3-Vicuna1.3-13B` |
| DeepSeek-R1-Distill-LLaMA 8B | `/path/to/DeepSeek-R1-Distill-Llama-8B` | `/path/to/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B` |

## Quick start

RADAR policy checkpoints are intentionally not distributed because the reward contains device-specific latency measurements. Train a policy for the deployment device as described below, then run evaluation with the resulting checkpoint.

For LLaMA-3.1-Instruct 8B on MT-bench:

```bash
python -m eagle.evaluation.gen_ea_we_answer_llama3chat \
  --base-model-path /path/to/Meta-Llama-3.1-8B-Instruct \
  --ea-model-path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
  --eye-model-path /path/to/radar-policy.pt \
  --bench-name mt_bench \
  --depth 7 \
  --top-k 10 \
  --temperature 1.0 \
  --answer-file output/mt_bench/radar-llama3.1-t1.jsonl
```

Replace the example paths with local model directories. The CLI defaults are placeholders and must be configured before running an experiment.

Equivalent RADAR entry points are:

- `eagle.evaluation.gen_ea_we_answer_llama3chat`
- `eagle.evaluation.gen_ea_we_answer_vicuna`
- `eagle.evaluation.gen_ea_we_answer_ds`

Use the corresponding `gen_ea_answer_*.py` entry points for EAGLE-3 without RADAR and the `gen_baseline_answer_*.py` entry points for autoregressive baselines.

To calculate speedup from generated answer files:

```bash
python -m eagle.evaluation.speed \
  --tokenizer-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --answer-file output/mt_bench/radar-llama3.1-t1.jsonl \
  --baseline-answer-file output/mt_bench/baseline-llama3.1-t1.jsonl
```

## Train a device-specific RADAR policy

### 1. Prepare prompts

The offline-data generators read FastChat-style questions from:

```text
eagle/data/<benchmark>/question.jsonl
```

To reproduce the paper, prepare 1,000 sampled ShareGPT prompts at `eagle/data/shareGPT/question.jsonl`. The repository includes the evaluation question files used by the public scripts, but does not redistribute the sampled ShareGPT training subset.

### 2. Generate acceptance-length distributions

For LLaMA-3.1-Instruct 8B:

```bash
python -m eagle.data.generate.ge_data_llama3_rb \
  --bench-name shareGPT \
  --question-begin 0 \
  --question-end 1000 \
  --depth 7 \
  --top-k 10 \
  --temperature 1.0 \
  --num-gpus-per-model 1 \
  --num-gpus-total 1 \
  --answer-file output/shareGPT/llama3.1-offline.jsonl \
  --save-dataset data/scores_rb/shareGPT-llama3-d7-topk10-t1
```

Use `ge_data_vicuna_rb` or `ge_data_ds_rb` for the other target models.

### 3. Configure latency and train

The files in `eagle/train/eye_*.json` define the policy hyperparameter grid and the profiled latency terms. All latency values are measured in seconds on the deployment device:

- `eaforward_time` ($T_d$): mean latency of one EAGLE draft-model forward pass;
- `eye_time` ($T_p$): mean latency of one RADAR-policy forward pass;
- `eagen_minus_time` ($T_o$): the fixed residual overhead after subtracting all draft-model forward passes from the mean EAGLE generation-cycle latency.

For a profile collected with a fixed maximum depth `maxlen`, compute

```text
eagen_minus_time = avg_eagen_time - eaforward_time * maxlen
beta_raw          = eaforward_time + eye_time
rate              = eagen_minus_time / beta_raw
```

Here, `avg_eagen_time` is the mean EAGLE generation-cycle latency at `maxlen`. The raw $\beta$ is the marginal latency of one draft-policy step and normalizes the terminal throughput reward:

```text
terminal_reward = beta * accepted_length / estimated_generation_time
continue_reward = -alpha
```

The released `eye_*-2.json` configurations intentionally use a reward scale of 10:

```text
beta  = 10 * beta_raw
alpha = 10 * alpha_raw
```

Scaling both coefficients by the same factor preserves their relative trade-off and the optimal policy while increasing the overall reward and gradient scale. If this scale is changed, scale `alpha` and `beta` together. For example, the LLaMA configuration uses `eaforward_time = 0.002684` and `eye_time = 0.0004`, giving `beta_raw = 0.003084` and the configured `beta = 0.03084`.

Profile after model warm-up, use the same model placement, precision, batch size, tree depth, and GPU setup as deployment, synchronize CUDA immediately before and after each timed region, and average multiple post-warm-up runs. Do not reuse the example latency values on a different device.

Update these values for the deployment device and verify `dataset_path`. Then run:

```bash
python -m eagle.train.train_radar \
  --config eagle/train/eye_llama3-2.json
```

Checkpoints, plots, and `results.csv` are written under `output/<benchmark>/<model>/<setting>/`. Select the checkpoint with the best validation result and pass it to `--eye-model-path` during evaluation.

## Repository structure

```text
eagle/
├── application/          # optional Gradio demo
├── data/
│   ├── generate/         # offline acceptance-distribution generation
│   └── <benchmark>/      # evaluation questions
├── evaluation/           # baseline, EAGLE-3, and RADAR evaluation
├── model/                # EAGLE-3 integration and RADAR policy
└── train/
    ├── train_radar.py    # offline RL policy training
    └── eye_*.json        # per-model training configurations
```

## Reproducibility notes

- Report the GPU model, number of GPUs, software versions, temperature, `depth`, `top-k`, and policy checkpoint for every speed measurement.
- Speedup is hardware-sensitive. Compare methods in the same environment.
- RADAR preserves the target model's output distribution through strict speculative sampling; it does not modify the target-model weights.
- The default maximum draft-model call count is eight: the initial call plus seven expansion steps (`--depth 7`).

Run the lightweight policy smoke tests with:

```bash
python -m unittest discover -s tests
```

## License and acknowledgments

This repository is released under the Apache License 2.0. Base models, datasets, and EAGLE-3 checkpoints remain subject to their respective licenses and access requirements.

RADAR is built on the [EAGLE project](https://github.com/SafeAILab/EAGLE). We thank its authors and the broader speculative-decoding community for their open-source contributions.

## Citation

The official Findings of EMNLP 2026 citation will be added when the proceedings metadata becomes available.
