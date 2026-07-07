# Configuration Reference

Nemotron-FineTune uses a three-layer YAML configuration system powered by [OmegaConf](https://omegaconf.readthedocs.io/).

## Config Merge Order

1. **`configs/base.yaml`** -- All default values. Loaded first.
2. **User config** -- Your `--config path/to/run.yaml`. Deep-merged on top of base.
3. **CLI dotlist overrides** -- `key=value` arguments. Merged last.

```bash
# Example: base defaults -> sft_sample.yaml -> override LR
python train.py --config configs/sft_sample.yaml training.learning_rate=1e-5
```

The fully-resolved config is saved to `<output_dir>/resolved_config.yaml` at the start of every run for exact reproducibility.

## Full Config Schema

### `run` -- Run metadata

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `cpt` \| `dapt` \| `sft` | `sft` | Training mode |
| `name` | string | `base_run` | Run name (used in W&B and output dir) |
| `output_dir` | string | `outputs/base_run` | Where all outputs are written |

### `model` -- Model configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | **required** | Local path or HuggingFace repo ID |
| `size` | `nano` \| `super` \| `ultra` | `nano` | Model size category |
| `torch_dtype` | string | `bfloat16` | Model dtype (`bfloat16`, `float16`, `float32`) |
| `attn_implementation` | string | `flash_attention_2` | Attention impl (`flash_attention_2`, `sdpa`, `eager`) |
| `use_cache` | bool | `false` | Whether to use KV cache (disable for training) |
| `use_mamba_kernels` | bool | not set | Override Mamba Triton kernels (set `false` for quantized models) |

**Notes:**
- NemotronH models do not support Flash Attention 2. Use `sdpa` or `eager`.
- Set `use_mamba_kernels: false` when using 4-bit quantization, as the Mamba Triton kernels call `F.linear` directly and cannot handle quantized weight tensors.

### `quantization` -- BitsAndBytes quantization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable 4-bit NF4 quantization |
| `bits` | int | `4` | Quantization bits (only 4 supported) |
| `quant_type` | string | `nf4` | Quantization type (`nf4` or `fp4`) |
| `compute_dtype` | string | `bfloat16` | Compute dtype for quantized ops |
| `double_quant` | bool | `false` | Double quantization (compresses quantization constants) |

**Memory estimates (30B model):**
- bf16 (no quantization): ~64 GB model weights
- 4-bit NF4: ~16 GB model weights

### `lora` -- LoRA adapter configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable LoRA finetuning |
| `r` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha scaling factor |
| `lora_dropout` | float | `0.05` | Dropout on LoRA layers |
| `target_modules` | string \| list | `all-linear` | Which modules to adapt |
| `bias` | string | `none` | Bias type (`none`, `lora_only`, `all`) |
| `task_type` | string | `CAUSAL_LM` | PEFT task type |

**Target modules:**
- `all-linear` -- PEFT auto-detects all `nn.Linear` layers. Works for standard transformer models.
- Explicit list -- Required for NemotronH hybrid models. Use:
  ```yaml
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - up_proj
    - down_proj
    - in_proj
    - out_proj
  ```

**LoRA parameter summary:**
- Trainable params = `2 * r * (in_features + out_features)` per target layer
- Typical ratio: 0.5-2% of total model params

### `data` -- Data configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_path` | string | **required** | Path to training data (JSON/JSONL) |
| `eval_path` | string | `null` | Path to evaluation data |
| `format` | `chat` \| `prompt_completion` \| `text` | `chat` | Data format |
| `max_seq_length` | int | `2048` | Maximum token sequence length |
| `packing` | bool | `false` | Pack short sequences into single examples |
| `num_proc` | int | `4` | Parallel workers for data formatting |

**Format requirements by mode:**
- `cpt` / `dapt` -- must use `text` format
- `sft` -- must use `chat` or `prompt_completion` format

### `training` -- Training hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_train_epochs` | float | `1` | Number of training epochs |
| `max_steps` | int | `-1` | Max training steps (overrides epochs if > 0) |
| `per_device_train_batch_size` | int | `2` | Batch size per GPU |
| `per_device_eval_batch_size` | int | `2` | Eval batch size per GPU |
| `gradient_accumulation_steps` | int | `4` | Gradient accumulation steps |
| `learning_rate` | float | `2e-4` | Peak learning rate |
| `lr_scheduler_type` | string | `cosine` | LR schedule (`cosine`, `linear`, `constant`, etc.) |
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
| `gradient_checkpointing` | bool | `true` | Enable gradient checkpointing (reduces VRAM) |
| `optim` | string | `paged_adamw_8bit` | Optimizer |

**Effective batch size** = `per_device_train_batch_size` x `gradient_accumulation_steps` x num GPUs

### `wandb` -- Weights & Biases

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `online` \| `offline` \| `disabled` | `disabled` | W&B logging mode |
| `project` | string | `nemotron-finetune` | W&B project name |
| `run_name` | string | `${run.name}` | W&B run name |

### `logging` -- Console/file logging

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
