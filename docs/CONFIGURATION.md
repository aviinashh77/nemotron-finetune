# Configuration Reference

Nemotron-FineTune uses a three-layer YAML configuration system powered by [OmegaConf](https://omegaconf.readthedocs.io/).

## Config Merge Order

1. **`configs/base.yaml`** — All default values. Loaded first.
2. **User config** — Your `--config path/to/run.yaml`. Deep-merged on top.
3. **CLI dotlist overrides** — `key=value` arguments. Merged last.

```bash
# Example: base defaults -> user config -> override LR
python train.py --config configs/cpt_dummy.yaml training.learning_rate=1e-5
```

The fully-resolved config is saved to `<output_dir>/resolved_config.yaml` at the start of every run for exact reproducibility.

## Full Config Schema

### `run` — Run metadata

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `cpt` \| `dapt` \| `sft` | `sft` | Training mode |
| `name` | string | `base_run` | Run name (used in W&B and output dir) |
| `output_dir` | string | `outputs/base_run` | Where all outputs are written |

### `model` — Model configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | **required** | Local path or HuggingFace repo ID |
| `size` | `nano` \| `super` \| `ultra` | `nano` | Model size category |
| `torch_dtype` | string | `bfloat16` | Model dtype (`bfloat16`, `float16`, `float32`) |
| `attn_implementation` | string | `flash_attention_2` | Attention impl — **must be `eager` for NemotronH** |
| `use_cache` | bool | `false` | Whether to use KV cache (disable for training) |
| `use_mamba_kernels` | bool | not set | Override Mamba Triton kernels (set `false` for quantized models or when kernels unavailable) |

**NemotronH notes:**
- NemotronH does **not** support `flash_attention_2` or `sdpa`. Use `attn_implementation: eager`.
- When `causal_conv1d` is not installed, the model prints a warning and falls back to naive Mamba (3x more activation memory).

### `quantization` — BitsAndBytes quantization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable 4-bit NF4 quantization |
| `bits` | int | `4` | Quantization bits (only 4 supported) |
| `quant_type` | string | `nf4` | Quantization type (`nf4` or `fp4`) |
| `compute_dtype` | string | `bfloat16` | Compute dtype for quantized ops |
| `double_quant` | bool | `false` | Double quantization |

**Memory estimates (30B model):**
- bf16 (no quantization): ~64 GB model weights
- 4-bit NF4: ~16 GB model weights
- Requires `use_mamba_kernels: false`

### `lora` — LoRA adapter configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable LoRA finetuning |
| `r` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha scaling factor |
| `lora_dropout` | float | `0.05` | Dropout on LoRA layers |
| `target_modules` | string \| list | `all-linear` | Which modules to adapt |
| `bias` | string | `none` | Bias type (`none`, `lora_only`, `all`) |
| `task_type` | string | `CAUSAL_LM` | PEFT task type |

**Target modules for NemotronH:**
Both entrypoints auto-discover linear layers. If auto-detection fails, specify explicitly:
```yaml
lora:
  target_modules:
    - in_proj
    - out_proj
    - up_proj
    - down_proj
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

**LoRA stats (r=16, 8 targets):** 441M trainable params (1.38% of 32B total).

### `data` — Data configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_path` | string | **required** | Path to training data (JSON/JSONL) |
| `eval_path` | string | `null` | Path to evaluation data |
| `format` | `chat` \| `prompt_completion` \| `text` | `chat` | Data format |
| `max_seq_length` | int | `2048` | Maximum token sequence length |
| `packing` | bool | `false` | Pack short sequences into single examples |
| `num_proc` | int | `4` | Parallel workers for data formatting |

**Format requirements by mode:**
- `cpt` / `dapt` — must use `text` format
- `sft` — must use `chat` or `prompt_completion` format

**Sequence length limits (2xA100-80GB):**
- DDP: max ~512 tokens
- FSDP: max ~5120 tokens

### `training` — Training hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | string | `${run.output_dir}` | Training output directory |
| `num_train_epochs` | float | `1` | Number of training epochs |
| `max_steps` | int | `-1` | Max training steps (overrides epochs if > 0) |
| `per_device_train_batch_size` | int | `2` | Batch size per GPU |
| `per_device_eval_batch_size` | int | `2` | Eval batch size per GPU |
| `gradient_accumulation_steps` | int | `4` | Gradient accumulation steps |
| `learning_rate` | float | `2e-4` | Peak learning rate |
| `lr_scheduler_type` | string | `cosine` | LR schedule (`cosine`, `linear`, `constant`) |
| `warmup_ratio` | float | `0.05` | Fraction of steps for warmup |
| `weight_decay` | float | `0.01` | Weight decay |
| `max_grad_norm` | float | `1.0` | Gradient clipping max norm |
| `bf16` | bool | `true` | Enable bfloat16 mixed precision |
| `fp16` | bool | `false` | Enable float16 mixed precision |
| `logging_steps` | int | `1` | Log metrics every N steps |
| `save_steps` | int | `500` | Save checkpoint every N steps |
| `save_total_limit` | int | `3` | Max checkpoints to keep |
| `eval_strategy` | string | `no` | Eval strategy (`no`, `steps`, `epoch`) |
| `eval_steps` | int | `500` | Eval every N steps |
| `seed` | int | `42` | Random seed |
| `dataloader_num_workers` | int | `4` | Data loading workers |
| `gradient_checkpointing` | bool | `true` | Enable gradient checkpointing (DDP only; FSDP uses activation_checkpointing) |
| `optim` | string | `paged_adamw_8bit` | Optimizer (`paged_adamw_8bit` for DDP, `adamw_torch` for FSDP) |

**Effective batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` × num GPUs

### `wandb` — Weights & Biases

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `online` \| `offline` \| `disabled` | `disabled` | W&B logging mode |
| `project` | string | `nemotron-finetune` | W&B project name |
| `run_name` | string | `${run.name}` | W&B run name |

**Note:** For FSDP, wandb env vars (`WANDB_PROJECT`, `WANDB_NAME`, `WANDB_MODE`) must be set before `SFTTrainer` init. Both entrypoints handle this automatically.

### `logging` — Console/file logging

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `level` | string | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Config Validation

The config is validated after merging. Errors raised:

- Missing required fields: `run.mode`, `run.name`, `run.output_dir`, `model.path`, `data.train_path`
- Invalid enum values for mode, size, format, wandb mode
- Mode/format mismatch: CPT/DAPT require `text`, SFT requires `chat`/`prompt_completion`
- LoRA: `r` must be positive, `target_modules` must be non-empty
- Training: must set either `max_steps > 0` or `num_train_epochs > 0`

## Included Configs

| Config | Mode | Data | Seq Len | Epochs | Purpose |
|--------|------|------|---------|--------|---------|
| `base.yaml` | sft | — | 2048 | 1 | All defaults |
| `sft_sample.yaml` | sft | 5 chat examples | 1024 | 2 | Quick SFT demo |
| `cpt_dummy.yaml` | cpt | 10 text examples | 512 | 2 | CPT smoke test |
| `cpt_long.yaml` | cpt | 118 text examples | 512 | 5 | Longer CPT run |
| `cpt_fsdp.yaml` | cpt | 200 long text examples | 5120 | 1 | FSDP long-sequence CPT |
| `cpt_verilog_fsdp.yaml` | cpt | 36k Verilog samples | 1024 | 1 | Verilog CPT (production) |

## TRL 1.8.0 Compatibility Notes

| Issue | Detail | Fix |
|-------|--------|-----|
| `max_seq_length` not in `SFTConfig` | `SFTConfig` doesn't accept `max_seq_length` — it's a `SFTTrainer` param | Pass to `SFTTrainer(max_seq_length=...)` |
| `past_key_values` missing | NemotronH uses `cache_params` instead | Monkey-patch `NemotronHOutput.__getattr__` |
| `all-linear` broken on custom models | PEFT can't auto-detect NemotronH linear layers | Manual discovery via `named_modules()` |
| wandb env vars not inherited | `SFTConfig` doesn't read config values for wandb | Set `WANDB_*` env vars before `SFTTrainer` init |
