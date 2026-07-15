# Architecture

## Overview

Nemotron-FineTune is a config-driven LoRA finetuning framework for Nemotron hybrid Mamba-transformer models. It has two entrypoints (`train.py` for DDP, `train_fsdp.py` for FSDP) that share a common config system and utility library.

## Entry Points

### `train.py` — DDP Training

The original training script. Loads the model on GPU, applies LoRA, and uses TRL's `SFTTrainer` with Accelerate for distributed data parallel (DDP) training.

**Flow:**
1. Parse CLI args, load and merge config (base.yaml → user YAML → CLI overrides)
2. Load data from JSON/JSONL
3. Format data (chat → tokenizer template, or raw text)
4. Load model on GPU (optional 4-bit quantization)
5. Patch `NemotronHOutput` for TRL compatibility
6. Auto-discover LoRA target modules (PEFT `all-linear` broken on NemotronH)
7. Apply LoRA via PEFT
8. Create `SFTConfig` + `SFTTrainer`
9. Train, save adapter + summary

**Launch:**
```bash
python train.py --config configs/cpt_dummy.yaml
# or
accelerate launch --num_processes 2 --multi_gpu train.py --config configs/cpt_dummy.yaml
```

### `train_fsdp.py` — FSDP Training

Parallel entrypoint for Fully Sharded Data Parallel training. Shards model weights across GPUs, enabling long sequences (5k+ tokens) on limited VRAM.

**Key differences from `train.py`:**
- **Pre-tokenizes data before model loading** — prevents OOM by tokenizing with a tiny tokenizer first, then loading the 64GB model into freed memory
- Model loaded on CPU (not GPU) — FSDP shards from CPU to each GPU
- `dist.init_process_group(backend="nccl")` called explicitly
- LoRA applied before FSDP wrapping (PEFT + FSDP requires `use_orig_params=True`)
- Uses `adamw_torch` optimizer instead of `paged_adamw_8bit` (FSDP/DTensor incompatible with 8-bit)
- Only rank 0 saves checkpoints and logs
- Pre-tokenized Arrow cache avoids re-tokenization on restart

**Flow:**
1. `dist.init_process_group(backend="nccl")`, set CUDA device
2. Load and merge config
3. Load tokenizer only (tiny)
4. Format + tokenize data, save Arrow cache to `tokenize_cache/`
5. Load model on CPU, patch NemotronH, apply LoRA
6. Create `SFTConfig` with FSDP settings (`fsdp: "full_shard auto_wrap"`)
7. Pass pre-tokenized data to `SFTTrainer` (no `dataset_text_field`, `processing_class=tokenizer`)
8. `SFTTrainer` handles FSDP wrapping automatically via Accelerate
9. Train, only rank 0 saves

**Launch:**
```bash
torchrun --nproc_per_node=2 train_fsdp.py --config configs/cpt_fsdp.yaml
```

## Source Modules (`src/nemotron_finetune/`)

### `config.py` — Configuration System

Three-layer OmegaConf merge:
1. `configs/base.yaml` — all defaults
2. User YAML — your run config
3. CLI dotlist overrides — `key=value` arguments

Key functions:
- `load_config(config_path, overrides)` — Loads, merges, validates, returns `DictConfig`
- `validate_config(cfg)` — Checks required fields, enum values, mode/format consistency
- `save_resolved(cfg, path)` — Writes fully-resolved config for reproducibility

### `callbacks.py` — Metrics Logging

`JsonlMetricsCallback` (extends `TrainerCallback`):
- Writes per-step metrics to `<output_dir>/logs/metrics.jsonl`
- Enriches with: perplexity, throughput (steps/sec, tokens/sec), GPU memory stats
- Events: `train_begin`, `log`, `evaluate`, `save`, `train_end`

### `env.py` — Environment Capture

- `collect_environment()` — Returns JSON snapshot: hostname, Python/packages/CUDA versions, GPU info
- `gpu_memory_stats()` — Current/peak GPU memory in GiB
- `reset_peak_gpu_memory()` — Resets CUDA peak memory tracking

### `logging_utils.py` — Logging

- `setup_logging(output_dir, level)` — Configures logger to stdout + `<output_dir>/logs/train.log`
- `write_json(path, data)` — Writes JSON document

## NemotronH Compatibility

The Nemotron-30B model uses a custom `NemotronHOutput` class that lacks `past_key_values` (uses `cache_params` instead). TRL 1.8.0's `_chunked_ce_forward` accesses `outputs.past_key_values`, causing `AttributeError`.

**Solution:** A `__getattr__` monkey-patch on `NemotronHOutput` and `NemotronHCausalLMOutput` that maps `past_key_values` → `cache_params`.

This patch is applied in both `train.py` and `train_fsdp.py` during model loading.

## LoRA Target Modules

NemotronH has these `nn.Linear` layers:

| Module | Location | Dimensions |
|--------|----------|------------|
| `in_proj` | Mamba blocks | 2688 × 10304 |
| `out_proj` | Mamba blocks | 4096 × 2688 |
| `q_proj` | Attention blocks (layers 5,12,19,26,33,42) | 2688 × 4096 |
| `k_proj` | Attention blocks | 2688 × 256 |
| `v_proj` | Attention blocks | 2688 × 256 |
| `o_proj` | Attention blocks | 4096 × 2688 |
| `up_proj` | MOE expert blocks | 2688 × 1856 |
| `down_proj` | MOE expert blocks | 1856 × 2688 |

PEFT cannot auto-detect these for NemotronH when `target_modules=None`. Both entrypoints auto-discover them by iterating `model.named_modules()`.

**LoRA stats (r=8, 8 targets):**
- Trainable params: ~4.5M (~0.015% of 30B total)

**LoRA stats (r=16, 8 targets):**
- Trainable params: 441,936,896 (1.38% of 32B total)

## FSDP Configuration

When using `train_fsdp.py`, the following FSDP settings are passed to SFTTrainer:

```python
fsdp = "full_shard auto_wrap"
fsdp_config = {
    "transformer_layer_cls_to_wrap": "NemotronHBlock",
    "backward_prefetch": "backward_pre",
    "forward_prefetch": "true",
    "use_orig_params": "true",          # Required for PEFT/LoRA
    # NOTE: activation_checkpointing removed — causes DTensor mismatch with NemotronH
}
```

- **`FULL_SHARD`**: Shards params, gradients, and optimizer states across GPUs
- **`NemotronHBlock`**: Each of the 52 blocks is a FSDP unit
- **`use_orig_params=True`**: Required for PEFT to access original parameter names
- **`activation_checkpointing`**: **REMOVED** — causes DTensor shape mismatch on forward pass. Gradient checkpointing is disabled via `gradient_checkpointing: false` in SFTConfig.

**Known FSDP issue:** `activation_checkpointing: true` in `fsdp_config` causes a DTensor shape mismatch error. The workaround is to disable both `gradient_checkpointing` and `activation_checkpointing`. This uses more activation memory but avoids the crash. On 2xA100-80GB with max_seq_length=1024, this is not a problem.

## Model Structure

Nemotron-3-Nano-30B-A3B-BF16:
- **Backbone**: 52 `NemotronHBlock` layers
  - Even layers (0,2,4,...): `NemotronHMamba2Mixer` (state-space model)
  - Odd layers (1,3,5,...): `NemotronHMOE` (Mixture of Experts, 128 experts + shared)
  - Attention layers (5,12,19,26,33,42): `NemotronHAttention` (standard attention)
- **LM Head**: `NemotronHLMHead` (2688 → vocab)
- **Embedding**: `NemotronHEmbedding` (vocab → 2688)
- **Config**: Hybrid Mamba-Transformer with MoE, 30B total / 3B active per token

## Data Pipeline (FSDP path)

The `train_fsdp.py` data pipeline is restructured to avoid OOM:

1. **Tokenize first** — Uses only the tokenizer (tiny) to format + tokenize data. Saves Arrow cache.
2. **Load model** — Only then loads the 64GB model weights into memory.
3. **Pre-tokenized data** — `SFTTrainer` receives Arrow datasets directly, no re-tokenization.

This ordering is critical because:
- Tokenizing requires ~2GB RAM (tokenizer + Arrow datasets)
- Model loading requires ~64GB RAM (bf16 weights)
- Combined ~66GB is fine for the 465GB cgroup limit
- If model loads first, tokenization pushes total over 465GB → OOM/SIGKILL

## W&B Integration

wandb is configured via the `wandb` config section:

```yaml
wandb:
  mode: offline  # buffered to ~/.local/share/wandb/
  project: vaschpforge-llm
  run_name: verilog_cpt_v0.1
```

Environment variables are set before `SFTTrainer` init to ensure TRL passes them to `SFTConfig`:
```python
os.environ["WANDB_PROJECT"] = cfg.wandb.project
os.environ["WANDB_NAME"] = cfg.run.name
os.environ["WANDB_MODE"] = cfg.wandb.mode
```

wandb is authenticated and the project `vaschpforge-llm` exists. Offline mode with 100MB buffer avoids network issues during training.
