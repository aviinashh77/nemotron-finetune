# Troubleshooting

## Common Issues

### 1. `ValueError: NemotronHForCausalLM does not support Flash Attention 2.0 yet`

**Cause:** NemotronH hybrid models don't implement Flash Attention 2.

**Fix:** Use `sdpa` or `eager` attention:

```yaml
model:
  attn_implementation: sdpa
```

### 2. `ValueError: Please specify target_modules or target_parameters in peft_config`

**Cause:** PEFT cannot auto-detect linear layers in the NemotronH hybrid architecture. The `all-linear` setting requires standard transformer modules.

**Fix:** Explicitly list target modules:

```yaml
lora:
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

### 3. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (N x M and 1 x K)` in Mamba layers

**Cause:** The Mamba Triton kernels (`mamba_ssm`) call `F.linear` directly, which is incompatible with BitsAndBytes 4-bit quantized weights. The quantized weight tensor is flattened to shape `(1, K)` instead of the expected 2D shape.

**Fix:** Disable Mamba kernels when using quantization:

```yaml
model:
  use_mamba_kernels: false
```

Or disable quantization entirely for models that fit in GPU memory in bf16.

### 4. `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`

**Cause:** TRL 0.21+ renamed `max_seq_length` to `max_length` in `SFTConfig`.

**Fix:** Use `max_length` in your config (the codebase handles this automatically).

### 5. `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'max_seq_length'`

**Cause:** In TRL 0.21+, `max_seq_length` moved to `SFTConfig` (not `SFTTrainer`).

**Fix:** Set `max_length` in the training config section. The codebase handles this automatically.

### 6. `ConfigError: Invalid configuration: lora.target_modules must be 'all-linear' or a non-empty list`

**Cause:** The config validator uses `isinstance(tm, (list, tuple))` which doesn't recognize OmegaConf's `ListConfig`.

**Fix:** Already patched in `config.py:139` to use `hasattr(tm, '__len__')`.

### 7. Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `training.per_device_train_batch_size=1`
2. Increase gradient accumulation: `training.gradient_accumulation_steps=8`
3. Reduce sequence length: `data.max_seq_length=512`
4. Enable gradient checkpointing: `training.gradient_checkpointing=true` (default)
5. Reduce LoRA rank: `lora.r=8`
6. Enable 4-bit quantization: `quantization.enabled=true` (not compatible with Mamba kernels)

### 8. Training loss is NaN or very high

**Possible causes:**
- Learning rate too high -- try `training.learning_rate=1e-5`
- Data format mismatch -- ensure your data matches `data.format`
- Tokenization issues -- check that chat template is applied correctly
- Numerical instability -- reduce `lora.r` or increase `lora.lora_alpha`

### 9. `mamba-ssm` or `causal-conv1d` build failures

**Cause:** Precompiled wheels may not be available for your CUDA/PyTorch combination. Source builds require compatible ninja and C++ toolchains.

**Fix:** These are optional dependencies. If not installed, the Mamba Triton kernels won't be available, but training still works with `use_mamba_kernels: false`.

## GPU Memory Usage

Typical memory usage for Nemotron-Nano-30B on A100-80GB:

| Config | Model Weights | Peak VRAM |
|--------|--------------|-----------|
| bf16, no quantization | ~64 GB | ~64 GB |
| 4-bit NF4 (with `use_mamba_kernels: false`) | ~16 GB | ~20 GB |

With LoRA (rank 8, 8 target modules):
- Trainable params: ~221M (0.69% of total)
- Adapter size: ~0.9 GB

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
| `logs/run_output.log` | Full stdout/stderr capture |
| `checkpoint-N/` | Checkpoint at step N (adapter, optimizer, scheduler, RNG) |
