# Nemotron-FineTune

Config-driven LoRA finetuning for NVIDIA Nemotron hybrid Mamba-transformer models.

Supports **SFT** (supervised fine-tuning), **CPT** (continued pre-training), and **DAPT** (domain-adaptive pre-training) through a single YAML-configured entrypoint, with local + Weights & Biases logging, checkpointing, and full environment reproducibility.

## Quick Start

### 1. Run the sample training

```bash
python train.py --config configs/sft_sample.yaml
```

This trains Nemotron-Nano-30B for 2 epochs on 5 sample chat datapoints with LoRA (rank 8). Training completes in ~3 minutes on a single A100-80GB.

### 2. Check the outputs

All artifacts are written to the directory specified in `training.output_dir` (default: `outputs/<run_name>/`):

```
outputs/sft_sample_5dp_2ep/
├── adapter_model.safetensors    # LoRA weights
├── adapter_config.json          # LoRA config
├── resolved_config.yaml         # Full config snapshot (reproducibility)
├── training_summary.json        # Final metrics
├── logs/
│   ├── metrics.jsonl            # Per-step enriched metrics
│   ├── train.log                # Human-readable log
│   ├── environment.json         # Hardware/software snapshot
│   └── run_output.log           # Full stdout/stderr
└── checkpoint-4/                # Checkpoint at final step
```

### 3. Override config from the CLI

```bash
# Change learning rate
python train.py --config configs/sft_sample.yaml training.learning_rate=1e-5

# Override multiple fields
python train.py --config configs/sft_sample.yaml \
    training.num_train_epochs=5 \
    training.per_device_train_batch_size=2 \
    lora.r=32
```

## Architecture

```
nemotron-finetune/
├── train.py                          # Main entrypoint
├── configs/
│   ├── base.yaml                     # All defaults (merged first)
│   └── sft_sample.yaml               # Example run config
├── src/nemotron_finetune/
│   ├── __init__.py                   # Package (v0.1.0)
│   ├── config.py                     # Config merge + validation
│   ├── callbacks.py                  # JsonlMetricsCallback
│   ├── env.py                        # Environment capture
│   └── logging_utils.py              # Logger + JSON writer
├── data/dummy/                       # Sample training data
└── outputs/                          # Training run outputs
```

### How it works

1. **Config merging** -- `configs/base.yaml` provides all defaults. Your run config is deep-merged on top. CLI dotlist overrides are merged last. The fully-resolved config is saved for reproducibility.

2. **Model loading** -- Loads a Nemotron model from a local path or HuggingFace Hub. Supports optional 4-bit NF4 quantization via BitsAndBytes. The `use_mamba_kernels` flag controls whether the Mamba Triton kernels are used (set to `false` when using quantization).

3. **LoRA application** -- Uses PEFT to inject LoRA adapters. For standard transformers, set `target_modules: all-linear`. For NemotronH hybrid models, specify explicit target modules (see `sft_sample.yaml`).

4. **Training** -- Uses TRL's `SFTTrainer` with gradient checkpointing, cosine LR scheduling, and `paged_adamw_8bit` optimizer by default.

5. **Logging** -- `JsonlMetricsCallback` enriches each training step with perplexity, throughput (steps/sec, tokens/sec), and GPU memory stats, writing to `logs/metrics.jsonl`. Environment info is captured at run start.

## Configuration

The config system uses [OmegaConf](https://omegaconf.readthedocs.io/) with YAML files. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the full reference.

### Minimal SFT config

```yaml
run:
  mode: sft
  name: my_run
  output_dir: outputs/my_run

model:
  path: /path/to/nemotron-model

data:
  train_path: data/train.json
  format: chat
```

All other fields fall back to `configs/base.yaml` defaults.

### Key config sections

| Section | Purpose |
|---------|---------|
| `run` | Mode (sft/cpt/dapt), run name, output directory |
| `model` | Model path, dtype, attention implementation |
| `quantization` | 4-bit NF4 quantization (BitsAndBytes) |
| `lora` | LoRA rank, alpha, target modules |
| `data` | Train/eval paths, format, sequence length, packing |
| `training` | Batch size, LR, epochs, gradient checkpointing, optimizer |
| `wandb` | Weights & Biases logging mode |
| `logging` | Log level |

## Supported Modes

| Mode | Data Format | Description |
|------|------------|-------------|
| `sft` | `chat` or `prompt_completion` | Supervised fine-tuning on instruction data |
| `cpt` | `text` | Continued pre-training on raw text corpora |
| `dapt` | `text` | Domain-adaptive pre-training on domain text |

## Data Formats

See [docs/DATA_FORMATS.md](docs/DATA_FORMATS.md) for detailed format specifications.

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

### Prompt/completion format (for SFT)

```json
[
  {"prompt": "What is 2+2?", "completion": "4"}
]
```

### Text format (for CPT/DAPT)

```json
[
  {"text": "Large amounts of unstructured text for continued pre-training..."}
]
```

## Requirements

- Python 3.12+
- PyTorch 2.8+ with CUDA 12.8+
- GPU: A100-80GB or larger (for 30B models without quantization)

### Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | 4.55+ | Model loading, tokenization |
| `peft` | 0.17+ | LoRA adapters |
| `trl` | 0.21+ | SFTTrainer |
| `accelerate` | 1.10+ | Distributed/device management |
| `datasets` | 5.0+ | Data loading |
| `bitsandbytes` | 0.44+ | 4-bit quantization |
| `omegaconf` | 2.3+ | Config management |
| `mamba-ssm` | 2.3+ | Mamba kernel support |
| `causal-conv1d` | 1.6+ | Causal convolution support |

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for known issues and solutions.

## License

N/A (internal project)
