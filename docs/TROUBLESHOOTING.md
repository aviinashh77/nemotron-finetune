# Troubleshooting

## Common Issues

### 1. `ValueError: NemotronHForCausalLM does not support Flash Attention 2.0 yet`

**Cause:** NemotronH hybrid models don't implement Flash Attention 2.

**Fix:** Use `eager` attention (required for NemotronH):

```yaml
model:
  attn_implementation: eager
```

### 2. `ValueError: Please specify target_modules or target_parameters in peft_config`

**Cause:** PEFT cannot auto-detect linear layers in NemotronH when loaded on CPU (FSDP path) or when the model has custom module types. The `all-linear` auto-detection only works reliably on CUDA-loaded standard transformer models.

**Fix:** Both `train.py` and `train_fsdp.py` auto-discover target modules by iterating `model.named_modules()`. If you see this error, it means the auto-discovery failed. Manually specify targets:

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

### 3. `RuntimeError: mat1 and mat2 shapes cannot be multiplied` in Mamba layers

**Cause:** Mamba Triton kernels (`mamba_ssm`) call `F.linear` directly, which is incompatible with BitsAndBytes 4-bit quantized weights.

**Fix:** Disable Mamba kernels when using quantization:

```yaml
model:
  use_mamba_kernels: false
```

### 4. TRL `past_key_values` AttributeError

**Cause:** TRL 1.8.0's `_chunked_ce_forward` accesses `outputs.past_key_values`, but NemotronH returns `NemotronHOutput` which only has `cache_params`.

**Fix:** Both entrypoints monkey-patch `NemotronHOutput.__getattr__` to map `past_key_values` → `cache_params`. This is applied automatically during model loading.

### 5. `ValueError: The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg can't be set to True simultaneously`

**Cause:** Transformers 5.x doesn't allow both `gradient_checkpointing=True` and `fsdp_config.activation_checkpointing=True`.

**Fix:** `train_fsdp.py` sets `gradient_checkpointing=False` and uses FSDP-native `activation_checkpointing=True` instead.

### 6. FSDP `activation_checkpointing` causes DTensor shape mismatch

**Cause:** FSDP's `activation_checkpointing: true` in `fsdp_config` causes a DTensor shape mismatch error during forward pass with NemotronH's custom architecture. The error manifests as a hang (process stuck in D state) or crash.

**Fix:** Remove `activation_checkpointing` from `fsdp_config` and disable `gradient_checkpointing`:

```python
fsdp_config = {
    "transformer_layer_cls_to_wrap": "NemotronHBlock",
    "backward_prefetch": "backward_pre",
    "forward_prefetch": "true",
    "use_orig_params": "true",
    # activation_checkpointing REMOVED
}
```

**Impact:** Uses ~20% more activation memory. On 2xA100-80GB with max_seq_length=1024, this is not a problem (63GB / 80GB used).

### 7. `paged_adamw_8bit` incompatible with FSDP/DTensor

**Cause:** BitsAndBytes `paged_adamw_8bit` optimizer creates DTensor-like state shards that conflict with FSDP's own sharding. Causes crash or hang during optimizer step.

**Fix:** Use `adamw_torch` for FSDP training:

```yaml
training:
  optim: adamw_torch
```

### 8. wandb project not passed to SFTConfig

**Cause:** TRL's `SFTConfig` doesn't inherit wandb settings from the config object. Environment variables must be set before `SFTTrainer` init.

**Fix:** Set env vars explicitly in `train_fsdp.py`:

```python
os.environ["WANDB_PROJECT"] = cfg.wandb.project
os.environ["WANDB_NAME"] = cfg.run.name
os.environ["WANDB_MODE"] = cfg.wandb.mode
```

### 9. `SFTConfig` rejects `max_seq_length`

**Cause:** `SFTConfig` in TRL 1.8.0 does NOT accept `max_seq_length`. That's an `SFTTrainer` parameter. `SFTConfig` accepts: `max_length`, `packing`, `packing_strategy`, `eval_packing`.

**Fix:** Remove `max_seq_length` from `SFTConfig` and pass it to `SFTTrainer` directly:

```python
trainer = SFTTrainer(
    ...,
    max_seq_length=cfg.data.max_seq_length,
)
```

### 10. Out of Memory (OOM)

**DDP OOM (seq_len > 512):**
DDP replicates the full model on each GPU. For 30B bf16 (~64 GB), only ~16 GB remains for activations after model + optimizer. Switch to FSDP:

```bash
torchrun --nproc_per_node=2 train_fsdp.py --config configs/cpt_fsdp.yaml
```

**FSDP OOM (seq_len > 5120 on 2xA100):**
At the hardware limit. Options:
- Add more GPUs: `torchrun --nproc_per_node=4`
- Use 4-bit quantization: `quantization.enabled=true` (reduces model from ~64 GB to ~16 GB)
- Reduce batch size: `training.per_device_train_batch_size=1`
- Reduce seq_len: `data.max_seq_length=4096`

**500GB cgroup OOM (container limit):**
If training processes are killed with SIGKILL (exit code -9) during tokenization, it means the model loaded BEFORE data tokenization, exceeding the 500GB container limit.

**Fix:** Pre-tokenize data before model loading (already implemented in `train_fsdp.py`):

```python
# 1. Load tokenizer only (tiny)
# 2. Format + tokenize data, save Arrow cache
# 3. THEN load model + LoRA
# 4. Pass pre-tokenized data to SFTTrainer
```

**General OOM checklist:**
1. `training.per_device_train_batch_size=1`
2. `training.gradient_accumulation_steps=8` (keeps effective batch size)
3. `data.max_seq_length=512` (for DDP)
4. `training.gradient_checkpointing=true` (DDP) or FSDP native activation_checkpointing
5. `lora.r=8` (reduces trainable params)
6. `quantization.enabled=true` + `model.use_mamba_kernels=false`

### 11. Training loss is NaN or very high

**Possible causes:**
- Learning rate too high — try `training.learning_rate=1e-5`
- Data format mismatch — ensure `data.format` matches your data
- Tokenization issues — check chat template is applied correctly
- Numerical instability — reduce `lora.r` or increase `lora.lora_alpha`

### 12. `mamba-ssm` or `causal-conv1d` build failures

**Cause:** Precompiled wheels may not be available for your CUDA/PyTorch combination.

**Fix:** These are optional. Training works with `use_mamba_kernels: false` (naive Mamba path, ~3x more activation memory).

For `mamba-ssm`:
```bash
pip install mamba-ssm --no-build-isolation --no-deps
```

For `causal-conv1d`: May fail on CUDA 13.0. The model falls back automatically.

### 13. `torchrun` errors with FSDP

**Cause:** `torchrun` requires the script to call `dist.init_process_group()` early.

**Fix:** `train_fsdp.py` handles this. Don't use `accelerate launch` with `train_fsdp.py` — use `torchrun` only:

```bash
# Correct
torchrun --nproc_per_node=2 train_fsdp.py --config configs/cpt_fsdp.yaml

# Wrong — will conflict with manual FSDP setup
accelerate launch --num_processes 2 train_fsdp.py --config configs/cpt_fsdp.yaml
```

### 14. FSDP warning: "1 of the 2 modules passed to fully_shard did not run forward before backward"

**Cause:** The `lm_head` (output projection) is wrapped as a separate FSDP unit but doesn't participate in every forward pass.

**Impact:** Warning only — training proceeds correctly. Can be ignored.

### 15. Process stuck in D state (disk sleep) after "Starting FSDP training..."

**Cause:** Multiple possible causes:
1. **Model loading after tokenization** — If the model is loaded BEFORE data is tokenized, the combined memory (64GB model + tokenizer + Arrow datasets) exceeds the 500GB cgroup limit. The OS starts swapping to disk, causing D state.
2. **FSDP `activation_checkpointing`** — DTensor shape mismatch causes hang.
3. **Naive Mamba first forward pass** — Without `causal_conv1d`, the first forward pass with packed sequences is extremely slow (can appear stuck for 10+ minutes).

**Fixes:**
1. Pre-tokenize data BEFORE model loading (already implemented)
2. Remove `activation_checkpointing` from `fsdp_config`
3. Be patient — first forward pass with naive Mamba + packed sequences can take 10-15 minutes

### 16. Some Verilog samples cause extreme tokenization slowdown

**Cause:** Some samples in `verilog_db_v0.1.jsonl` are extremely long (589k tokens, 55M chars). Tokenizing these through the Nemotron tokenizer can stall for minutes per sample.

**Fix:** Filter the dataset by character count before training:

```python
# Filter: ~10k tokens ≈ 30k chars
MAX_CHARS = 30000
filtered = [d for d in data if len(d.get("text", "")) <= MAX_CHARS]
```

This keeps ~97% of samples and removes the problematic long sequences.

## GPU Memory Usage

Nemotron-30B bf16 on A100-80GB:

| Config | Model Weights | Peak VRAM | Max Seq Len |
|--------|--------------|-----------|-------------|
| bf16, DDP | ~64 GB (replicated) | 70.5 GB | ~512 |
| bf16, FSDP (2 GPU) | ~32 GB (sharded) | 78 GB | ~5120 |
| 4-bit NF4, DDP | ~16 GB (replicated) | ~20 GB | ~2048 |

With LoRA (r=8, 8 targets): ~4.5M trainable params (~0.015%), negligible memory.

## Output Files Reference

| File | Description |
|------|-------------|
| `resolved_config.yaml` | Complete config snapshot used for the run |
| `training_summary.json` | Final metrics (loss, runtime, FLOPs) |
| `adapter_model.safetensors` | LoRA adapter weights |
| `adapter_config.json` | LoRA configuration |
| `logs/metrics.jsonl` | Per-step metrics (loss, perplexity, GPU memory, throughput) |
| `logs/train.log` | Human-readable training log |
| `logs/environment.json` | Python/CUDA/GPU/package snapshot |
| `checkpoint-N/` | Checkpoint at step N (adapter, optimizer, scheduler, RNG) |
| `tokenize_cache/` | Pre-tokenized Arrow cache (train.arrow, eval.arrow) |
