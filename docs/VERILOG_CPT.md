# Verilog CPT Project Log

## Objective

Continued Pre-Training (CPT) Nemotron-30B on Verilog HDL data to teach the model HDL for use in an agentic flow. Using FSDP on 2xA100-80GB with LoRA, wandb logging, and iterative hyperparameter tuning.

## Environment

| Component | Value |
|-----------|-------|
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Total params | 30B (3B active via MoE) |
| Model size (bf16) | ~64 GB |
| Architecture | Hybrid Mamba-Transformer with MoE, 52 layers |
| GPU | 2x NVIDIA A100-SXM4-80GB |
| VRAM per GPU | 80 GB |
| Container RAM limit | 500 GB (cgroup) |
| CUDA | 13.0 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.13.1 |
| TRL | 1.8.0 |
| PEFT | 0.19.1 |

## Timeline

### Phase 1: Infrastructure Setup (Day 1)

1. Cloned repo, installed all dependencies
2. Downloaded Nemotron-30B model (custom modeling code, not standard transformers)
3. Built `causal_conv1d` — FAILED (CUDA 13.0 vs 12.8 mismatch)
4. Installed `mamba-ssm==2.3.2.post1` via `--no-build-isolation --no-deps`
5. Verified 10-sample test: FSDP works, loss drops, packing 1.34x speedup
6. Created comprehensive docs: README.md, SETUP.md, ARCHITECTURE.md, etc.

### Phase 2: Data Preparation (Day 1)

1. Source data: `verilog_db_v0.1.jsonl` — 38,417 Verilog modules
2. Converted to text format: `train_full.json` (37,417 samples) + `val_full.json` (1,000 samples)
3. Discovered: some samples are 589k tokens (16M tokens max!)
4. Filtered dataset: removed samples >30k chars (~10k tokens)
   - Final: 36,321 train, 971 val
   - Kept 97.1% of samples
   - Max chars: 29,754 (train), 29,638 (val)

### Phase 3: Bug Hunting (Day 1-2)

Encountered and fixed 8+ bugs:

1. **PEFT `all-linear` broken on NemotronH** — auto-detect raises ValueError. Fix: manual discovery via `named_modules()`.
2. **FSDP `activation_checkpointing` causes DTensor mismatch** — removed from config.
3. **`paged_adamw_8bit` incompatible with FSDP/DTensor** — switched to `adamw_torch`.
4. **wandb project not passed to SFTConfig** — set env vars before trainer init.
5. **`load_data` dropped eval split** — removed `split` param.
6. **TRL `past_key_values` AttributeError** — monkey-patched `NemotronHOutput.__getattr__`.
7. **SFTConfig rejects `max_seq_length`** — removed from SFTConfig, pass to SFTTrainer.
8. **500GB cgroup OOM** — restructured to pre-tokenize before model loading.

### Phase 4: Training Attempts (Day 2)

**Run 1 (OOM):** Model loaded before tokenization → combined memory exceeded 500GB → SIGKILL.

**Run 2 (OOM):** Same issue, different config. Confirmed: model MUST load AFTER tokenization.

**Run 3 (Python error):** `SFTConfig.__init__() got unexpected keyword argument 'max_seq_length'`. Fixed by removing it.

**Run 4 (D state hang):** Pre-tokenization restructure succeeded but training stuck in D state after "Starting FSDP training...". Root cause: FSDP `activation_checkpointing` DTensor mismatch.

**Run 5 (current — SUCCESS):** Removed `activation_checkpointing`, removed long samples. Training running at 28% progress.

### Phase 5: Current Training Status

```
Step 1968/7090 (28%)
Epoch 0.55
Loss: 0.35-0.55
Accuracy: 87-91%
Speed: ~8.3s/it
VRAM: 63GB / 80GB per GPU
ETA: ~12 hours
```

## Key Code Changes

### `train_fsdp.py` — Pre-Tokenization Restructure

```python
# OLD (OOM):
model = load_model()          # 64GB loaded first
trainer = SFTTrainer(
    train_dataset=train_data,  # Tokenizes here → OOM
    ...
)

# NEW (works):
tokenizer = load_tokenizer()   # Tiny
train_data = format_and_tokenize()  # ~2GB
save_arrow_cache(train_data)
model = load_model()           # 64GB loaded into freed memory
trainer = SFTTrainer(
    train_dataset=load_arrow_cache(),  # Pre-tokenized, no re-tokenization
    processing_class=tokenizer,
    ...
)
```

### `train_fsdp.py` — FSDP Config Fix

```python
# OLD (DTensor mismatch):
fsdp_config = {
    "activation_checkpointing": True,  # REMOVED
    ...
}

# NEW (works):
fsdp_config = {
    "transformer_layer_cls_to_wrap": "NemotronHBlock",
    "backward_prefetch": "backward_pre",
    "forward_prefetch": "true",
    "use_orig_params": "true",
    # activation_checkpointing REMOVED
}
```

### `train_fsdp.py` — Optimizer Fix

```python
# OLD (FSDP/DTensor incompatible):
training_args = SFTConfig(optim="paged_adamw_8bit", ...)

# NEW (works):
training_args = SFTConfig(optim="adamw_torch", ...)
```

### `train_fsdp.py` — wandb Env Vars

```python
# Must set BEFORE SFTTrainer init
os.environ["WANDB_PROJECT"] = cfg.wandb.project
os.environ["WANDB_NAME"] = cfg.run.name
os.environ["WANDB_MODE"] = cfg.wandb.mode
```

## Data Distribution

```
Char threshold    ~Tokens    Samples    Percentage
    4,096         ~1,200     31,301     83.7%
    8,192         ~2,500     33,849     90.5%
   16,384         ~4,900     35,378     94.6%
   30,000         ~9,000     36,321     97.1%  ← chosen threshold
   40,000        ~12,000     36,578     97.8%
   65,536        ~19,700     37,000     98.9%
  100,000        ~30,000     37,056     99.0%
```

Median sample: 686 chars (~200 tokens). Most Verilog modules are short.

## W&B Setup

- Project: `vaschpforge-llm`
- Run name: `verilog_cpt_v0.1`
- Mode: `offline` (buffered to `~/.local/share/wandb/`)
- Buffer size: 100MB
- Authenticated and verified

## Files

| File | Description |
|------|-------------|
| `configs/cpt_verilog_fsdp.yaml` | Training config |
| `train_fsdp.py` | Training script (pre-tokenization, all fixes applied) |
| `data/verilog/train_full.json` | 36,321 training samples |
| `data/verilog/val_full.json` | 971 eval samples |
| `results/output_fsdp_full/train.log` | Training log |
| `results/output_fsdp_full/tokenize_cache/` | Arrow cache |
| `results/output_fsdp_full/resolved_config.yaml` | Resolved config |
| `verilog_db_v0.1.jsonl` | Source data (38,417 modules) |

## Lessons Learned

1. **Pre-tokenize before model loading** — With a 64GB model on a 500GB cgroup, there's no room for tokenization after model load. Always tokenize first.
2. **Filter long samples** — Some Verilog samples are 589k tokens. Filter at ~10k tokens (30k chars) to keep 97% of data.
3. **PEFT `all-linear` broken on custom models** — Must manually discover target modules for NemotronH.
4. **FSDP `activation_checkpointing` broken on NemotronH** — DTensor shape mismatch. Disable it.
5. **`paged_adamw_8bit` incompatible with FSDP** — Use `adamw_torch` instead.
6. **wandb needs env vars set early** — TRL doesn't inherit config values. Set `WANDB_*` env vars before `SFTTrainer` init.
7. **`SFTConfig` != `SFTTrainer` params** — `max_seq_length` goes to `SFTTrainer`, not `SFTConfig`.
8. **Naive Mamba is slow but works** — First forward pass can take 10+ minutes. Subsequent steps are ~8s/it.
9. **Packing is safe for Mamba** — Verified 1.34x speedup with packing=true. Cross-contamination warning can be ignored for CPT.
10. **`causal_conv1d` can't build on CUDA 13.0** — Training still works with naive Mamba fallback.

## Next Steps

After v0.1 completes:
- Analyze loss curves and sample quality
- Experiment with LoRA rank (r=16 vs r=8)
- Try seq_len=2048 with filtered data (should fit in VRAM)
- Add eval loss monitoring
- Experiment with learning rate schedules
- Consider multi-epoch training
- Test on downstream Verilog generation tasks
