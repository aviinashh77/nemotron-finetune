# Nemotron-FineTune

Config-driven LoRA finetuning for NVIDIA Nemotron hybrid Mamba-transformer models.

Supports **SFT** (supervised fine-tuning), **CPT** (continued pre-training), and **DAPT** (domain-adaptive pre-training) through two distributed strategies: **DDP** (single/multi-GPU replication) and **FSDP** (full parameter sharding for long sequences).

## Status

| Component | Status |
|-----------|--------|
| DDP training (`train.py`) | Working — tested up to seq_len=512 on 2xA100 |
| FSDP training (`train_fsdp.py`) | Working — Verilog CPT v0.1 completed (16.5h, 7,090 steps) |
| LoRA CPT | Working — dummy data: loss 1.43 → 0.09, accuracy 64% → 97% |
| LoRA CPT (Verilog) | **v0.1 regressed downstream** — token accuracy 81%→89% but SystemVerilog pass@1 56%→42%; root causes + fixes in [docs/VERILOG_CPT_V0.1_POSTMORTEM.md](docs/VERILOG_CPT_V0.1_POSTMORTEM.md); corrected recipe: `configs/cpt_verilog_v0.2.yaml` |
| LoRA SFT | Working — tested on 5 chat examples |
| 4-bit quantization | Supported (requires `use_mamba_kernels: false`) |
| NemotronH compatibility | Patched — TRL `past_key_values` compat via `__getattr__` monkey-patch |
| `causal_conv1d` | Not available — model falls back to naive Mamba (3x more activation memory) |
| wandb logging | Working — project `vaschpforge-llm`, offline mode with 100MB buffer |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For FSDP multi-GPU training, also install Mamba kernels (optional):
```bash
pip install mamba-ssm --no-build-isolation --no-deps
```

### 2. Run DDP training (single or multi-GPU)

```bash
# Single GPU
python train.py --config configs/cpt_dummy.yaml

# Multi-GPU (DDP — replicates model on each GPU)
accelerate launch --num_processes 2 --multi_gpu train.py --config configs/cpt_dummy.yaml
```

### 3. Run FSDP training (multi-GPU, long sequences)

```bash
# FSDP — shards model across GPUs, enables 5k+ seq_len
torchrun --nproc_per_node=2 train_fsdp.py --config configs/cpt_fsdp.yaml
```

### 4. Override config from CLI

```bash
python train.py --config configs/cpt_dummy.yaml \
    training.learning_rate=1e-5 \
    data.max_seq_length=1024 \
    lora.r=8
```

## Architecture

```
nemotron-finetune/
├── train.py                          # DDP entrypoint (single/multi-GPU)
├── train_fsdp.py                     # FSDP entrypoint (multi-GPU, long sequences)
├── requirements.txt                  # Pinned dependencies
├── configs/
│   ├── base.yaml                     # All defaults (merged first)
│   ├── sft_sample.yaml               # Quick SFT demo (5 chat examples)
│   ├── cpt_dummy.yaml                # CPT test (10 text examples)
│   ├── cpt_long.yaml                 # Longer CPT (100 examples, 5 epochs)
│   ├── cpt_fsdp.yaml                 # FSDP long-sequence CPT (seq_len=5120)
│   └── cpt_verilog_fsdp.yaml         # FSDP Verilog CPT (36k samples, 1 epoch)
├── src/nemotron_finetune/
│   ├── __init__.py                   # Package (v0.1.0)
│   ├── config.py                     # Three-layer config merge + validation
│   ├── callbacks.py                  # JsonlMetricsCallback (per-step metrics)
│   ├── env.py                        # Environment/GPU capture
│   └── logging_utils.py              # Logger + JSON writer
├── data/
│   ├── dummy/                        # Sample data for testing
│   └── verilog/                      # Verilog HDL training data
│       ├── train_full.json           # 36,321 training samples (filtered ≤30k chars)
│       └── val_full.json             # 971 eval samples
├── models/
│   └── NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/  # Model weights + custom code
├── results/
│   └── output_fsdp_full/            # Verilog CPT output (in progress)
├── docs/
│   ├── SETUP.md                      # Detailed environment setup
│   ├── ARCHITECTURE.md               # How the code works
│   ├── CONFIGURATION.md              # Full config reference
│   ├── DATA_FORMATS.md               # Data format specs
│   ├── EXPERIMENTS.md                # Experiment logs and results
│   ├── TROUBLESHOOTING.md            # Known issues and fixes
│   └── VERILOG_CPT.md                # Verilog CPT project log
└── outputs/                          # DDP training run outputs
```

### How it works

1. **Config merging** — `configs/base.yaml` provides all defaults. Your run config is deep-merged on top. CLI dotlist overrides are merged last. The fully-resolved config is saved for reproducibility.

2. **Model loading** — Loads a Nemotron model from a local path. Supports optional 4-bit NF4 quantization via BitsAndBytes. A monkey-patch adds `past_key_values` to `NemotronHOutput` for TRL compatibility.

3. **LoRA application** — Uses PEFT to inject LoRA adapters. For NemotronH, the code auto-discovers all `nn.Linear` layers (excluding `lm_head`) as targets.

4. **Distributed training** — Two paths:
   - **DDP** (`train.py`): Model replicated on each GPU. Fast for short sequences. Limited by single-GPU memory (~512 tokens for 30B bf16 on A100-80GB).
   - **FSDP** (`train_fsdp.py`): Model sharded across GPUs. Enables long sequences (up to ~5120 tokens on 2xA100-80GB). Uses FSDP-native activation checkpointing.

5. **Training** — Uses TRL's `SFTTrainer` with cosine LR scheduling. Logs perplexity, throughput, and GPU memory to `logs/metrics.jsonl` and optionally to wandb.

## DDP vs FSDP

| Feature | DDP (`train.py`) | FSDP (`train_fsdp.py`) |
|---------|-------------------|------------------------|
| Launch | `accelerate launch` or `python` | `torchrun --nproc_per_node=N` |
| Model placement | Full copy per GPU | Sharded across GPUs |
| Max seq_len (2xA100-80GB) | ~512 | ~5120 |
| GPU memory per device | ~70 GB at seq_len=512 | ~78 GB at seq_len=5120 |
| Activation checkpointing | `gradient_checkpointing: true` | FSDP `activation_checkpointing` |
| Optimizer | `paged_adamw_8bit` | `adamw_torch` |
| Best for | Short sequences, fast iteration | Long sequences, production CPT |

### Memory usage (2xA100-80GB, Nemotron-30B bf16)

| Seq Len | Strategy | Peak VRAM/GPU | Status |
|---------|----------|---------------|--------|
| 512 | DDP | 70.5 GB | OOM at step 50 |
| 512 | FSDP | ~46 GB | Works |
| 2048 | FSDP | ~47 GB | Works |
| 3072 | FSDP | 67.7 GB | Works (12 GB headroom) |
| 4096 | FSDP | 76.9 GB | Works (3 GB headroom) |
| 5120 | FSDP | 78.0 GB | Works (2 GB headroom, limit) |

## Supported Modes

| Mode | Data Format | Description | Launch |
|------|------------|-------------|--------|
| `sft` | `chat` or `prompt_completion` | Supervised fine-tuning on instruction data | `train.py` |
| `cpt` | `text` | Continued pre-training on raw text corpora | `train.py` or `train_fsdp.py` |
| `dapt` | `text` | Domain-adaptive pre-training on domain text | `train.py` or `train_fsdp.py` |

## Data Formats

### Chat format (for SFT)

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"}
    ]
  }
]
```

### Text format (for CPT/DAPT)

```json
[
  {"text": "Large amounts of unstructured text for continued pre-training..."}
]
```

See [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) for full specifications.

## Configuration

The config system uses [OmegaConf](https://omegaconf.readthedocs.io/) with YAML files. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference.

### Verilog CPT config (v0.2 — corrected recipe)

The v0.1 recipe (seq 1024, LoRA on Mamba `in_proj`/`out_proj`, packing without EOS, no replay data, LR 1e-4) regressed SystemVerilog pass@1 — see the [post-mortem](docs/VERILOG_CPT_V0.1_POSTMORTEM.md). The corrected recipe is [`configs/cpt_verilog_v0.2.yaml`](configs/cpt_verilog_v0.2.yaml); key deltas:

```yaml
model:
  use_mamba_kernels: true       # install pinned mamba-ssm + causal-conv1d first

data:
  max_seq_length: 4096          # was 1024 (pretraining length is 8192)
  packing: true
  append_eos: true              # was missing — packed docs had no boundaries
  replay_path: data/replay/general_code.json   # 20% general-data replay
  replay_ratio: 0.2

lora:
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj]   # attention-only; no Mamba modules

training:
  learning_rate: 2.0e-5         # was 1e-4
  gradient_checkpointing: true  # HF non-reentrant (validate on GPU)
```

Run the GPU-session validation checklist in the post-mortem before launching the full run.

## Requirements

- Python 3.12+
- PyTorch 2.8+ with CUDA 12.8+
- GPU: A100-80GB or larger (for 30B models without quantization)
- For FSDP: 2+ GPUs

### Tested dependency versions

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.8.0+cu128 | Deep learning framework |
| `transformers` | 5.13.1 | Model loading, tokenization |
| `peft` | 0.19.1 | LoRA adapters |
| `trl` | 1.8.0 | SFTTrainer |
| `accelerate` | 1.14.0 | Distributed/device management |
| `datasets` | 5.0.0 | Data loading |
| `bitsandbytes` | 0.49.2 | 4-bit quantization |
| `omegaconf` | 2.3.1 | Config management |
| `mamba-ssm` | 2.3.2.post1 | Mamba kernel support (optional) |

## Model

Tested with **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16**:
- 30B total parameters, 3B active per token (Mixture of Experts)
- Hybrid Mamba-Transformer architecture with 52 layers
- Alternating Mamba2Mixer and MOE blocks, with attention at layers 5, 12, 19, 26, 33, 42
- bf16 weights: ~64 GB
- Custom modeling code in model directory (not standard transformers)

## Documentation

- [Setup Guide](docs/SETUP.md) — Environment setup, dependency installation, model preparation
- [Architecture](docs/ARCHITECTURE.md) — How the code works, module-by-module
- [Configuration Reference](docs/CONFIGURATION.md) — Full config schema
- [Data Formats](docs/DATA_FORMATS.md) — Data format specifications
- [Experiments](docs/EXPERIMENTS.md) — Experiment logs, memory benchmarks, results
- [Troubleshooting](docs/TROUBLESHOOTING.md) — Known issues and fixes
- [Verilog CPT Log](docs/VERILOG_CPT.md) — Verilog CPT project log and lessons learned
- [Verilog CPT v0.1 Post-Mortem](docs/VERILOG_CPT_V0.1_POSTMORTEM.md) — root-cause analysis of the pass@1 regression, memory issues, v0.2 recipe rationale, GPU validation checklist

## License

N/A (internal project)
