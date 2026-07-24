# Verilog CPT v0.1 — Run Report

> **Status:** Complete  
> **Date:** July 14–15, 2026  
> **HF Adapter:** [Ashx098/chipforge-llm/phase_1/v0.1](https://huggingface.co/Ashx098/chipforge-llm/tree/main/phase_1/v0.1)  
> **wandb:** [vaschpforge-llm/runs/c5gid36w](https://wandb.ai/avinash-mynampati-juspay/vaschpforge-llm/runs/c5gid36w)

---

## 1. Objective

Continued Pre-Training (CPT) Nemotron-30B on Verilog HDL data to teach the model hardware description language (HDL) capabilities for use in an agentic flow.

## 2. Configuration

### Model

| Parameter | Value |
|-----------|-------|
| Model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Total params | 30B (3B active via MoE) |
| Architecture | Hybrid Mamba-Transformer, 52 layers |
| Precision | bfloat16 (~64 GB) |
| Attention impl | `eager` (required for NemotronH) |
| Mamba kernels | Disabled (naive fallback, `causal_conv1d` won't build on CUDA 13.0) |

### LoRA

| Parameter | Value |
|-----------|-------|
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Target modules | `in_proj`, `out_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` |
| Trainable params | ~4.5M (0.015% of 30B) |

### Data

| Parameter | Value |
|-----------|-------|
| Source | `verilog_db_v0.1.jsonl` (38,417 Verilog modules) |
| Train samples | 36,321 (filtered ≤30k chars / ~10k tokens) |
| Eval samples | 971 |
| Format | `text` (raw Verilog code) |
| Max seq length | 1024 |
| Packing | true (1.34x speedup verified) |

### Training

| Parameter | Value |
|-----------|-------|
| Strategy | FSDP (`full_shard auto_wrap`) |
| Hardware | 2x A100-SXM4-80GB |
| Epochs | 2 |
| Total steps | 7,090 |
| Batch size | 1 per device × 2 grad accum × 2 GPUs = **effective 4** |
| Learning rate | 1e-4 (cosine, 5% warmup) |
| Optimizer | `adamw_torch` (FSDP-compatible) |
| Max grad norm | 1.0 |
| Seed | 42 |
| Activation checkpointing | Disabled (DTensor mismatch with NemotronH) |

### Logging

| Parameter | Value |
|-----------|-------|
| wandb project | `vaschpforge-llm` |
| wandb mode | `offline` (synced post-run) |
| Save steps | 4,677, 7,090 (end) |
| Eval steps | 2,338, 4,676, 7,014, 7,090 |
| Logging steps | 10 |

---

## 3. Results

### Final Metrics

| Metric | Train | Eval |
|--------|-------|------|
| Loss | 0.4211 | 0.4272 |
| Token accuracy | 89.31% | 88.97% |
| Entropy | — | 0.4225 |

### Loss Curve

```
Step   10: 0.7448 ────────────────────────────────────
Step  410: 0.5324 ──────────────────────
Step 1010: 0.4879 ───────────────────
Step 1610: 0.4421 ──────────────
Step 2210: 0.4712 ────────────────
Step 2810: 0.4039 ─────────────
Step 3410: 0.3756 ────────────
Step 4010: 0.4616 ──────────────
Step 4610: 0.4297 ─────────────
Step 5210: 0.3875 ────────────
Step 5810: 0.3712 ────────────
Step 6410: 0.3626 ───────────
Step 7010: 0.4085 ─────────────
```

Loss dropped **43%** from 0.74 to 0.42 over 2 epochs.

### Accuracy Curve

```
Step   10: 82.93% ────────────────────────────────────
Step 1010: 87.71% ──────────────────────────────────
Step 2010: 87.23% ─────────────────────────────────
Step 3010: 89.18% ──────────────────────────────────
Step 4010: 88.11% ─────────────────────────────────
Step 5010: 89.16% ──────────────────────────────────
Step 6010: 89.73% ──────────────────────────────────
Step 7010: 89.49% ──────────────────────────────────
```

Accuracy rose **6.5%** from 82.9% to 89.5%.

### Eval Loss Over Training

| Step | Eval Loss | Eval Accuracy | Eval Entropy |
|------|-----------|---------------|--------------|
| 2,338 | 0.4650 | 88.11% | 0.4636 |
| 4,676 | 0.4350 | 88.77% | 0.4333 |
| 7,014 | 0.4272 | 88.97% | 0.4225 |
| 7,090 | 0.4272 | 88.97% | 0.4225 |

Eval loss closely tracks train loss — **no overfitting**.

### Throughput

| Metric | Value |
|--------|-------|
| Avg step time | 8.30s/it |
| Throughput | ~247 tokens/sec |
| Total training time | ~16.5 hours |
| Tokens processed | ~8.2M per epoch |

### Resource Usage

| Resource | Value |
|----------|-------|
| GPU 0 VRAM | 63,259 MB / 81,920 MB (77%) |
| GPU 1 VRAM | 61,221 MB / 81,920 MB (75%) |
| GPU utilization | 76–86% |
| GPU temperature | 51–55°C |
| CPU RAM | ~123 GB / 465 GB cgroup |
| Disk | Tokenized data cache ~75 MB |

---

## 4. Checkpoints

| Checkpoint | Step | Epoch | Train Loss | Notes |
|------------|------|-------|------------|-------|
| `checkpoint-4677` | 4,677 | ~1.32 | ~0.43 | Mid-training |
| `checkpoint-7090` | 7,090 | 2.0 | 0.42 | **Final** |

**⚠️ Note:** `save_steps: 4677` was suboptimal — only 2 checkpoints saved for a 7,090-step run. For future runs, use `save_steps: 1000` to get ~7 checkpoints across the run.

**Files per checkpoint:**
- `adapter_model.safetensors` (18 MB) — LoRA weights
- `adapter_config.json` — LoRA config
- `trainer_state.json` — full training log history

---

## 5. Bugs Encountered & Fixes

### 5.1 PEFT `all-linear` Broken on NemotronH

**Symptom:** `ValueError: Please specify target_modules or target_parameters in peft_config`

**Cause:** PEFT's auto-detection can't introspect NemotronH's custom module types.

**Fix:** Manual discovery via `model.named_modules()`:
```python
target_modules = set()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and "lm_head" not in name:
        target_modules.add(name.split(".")[-1])
```

### 5.2 FSDP `activation_checkpointing` DTensor Mismatch

**Symptom:** Training hangs in D state (disk sleep) or crashes with DTensor shape error.

**Cause:** FSDP's activation checkpointing creates DTensor sharding mismatches with NemotronH's custom forward pass.

**Fix:** Remove `activation_checkpointing` from `fsdp_config`. Disable `gradient_checkpointing` in `SFTConfig`. Costs ~20% more activation memory but avoids crash.

### 5.3 `paged_adamw_8bit` Incompatible with FSDP

**Symptom:** Crash during optimizer step.

**Cause:** BitsAndBytes 8-bit optimizer creates state shards conflicting with FSDP's own sharding.

**Fix:** Use `optim: adamw_torch`.

### 5.4 wandb Project Not Passed to SFTConfig

**Symptom:** wandb run appears under wrong project or doesn't log.

**Fix:** Set env vars before `SFTTrainer` init:
```python
os.environ["WANDB_PROJECT"] = cfg.wandb.project
os.environ["WANDB_NAME"] = cfg.run.name
os.environ["WANDB_MODE"] = cfg.wandb.mode
```

### 5.5 `SFTConfig` Rejects `max_seq_length`

**Symptom:** `TypeError: SFTConfig.__init__() got unexpected keyword argument 'max_seq_length'`

**Fix:** `max_seq_length` is an `SFTTrainer` param, not `SFTConfig`. Pass it to `SFTTrainer(max_seq_length=...)`.

### 5.6 OOM During Tokenization (500GB cgroup)

**Symptom:** SIGKILL (exit code -9) during label building at 83%.

**Cause:** Model loaded BEFORE tokenization, combined memory (64GB model + tokenizer + datasets) exceeded 500GB container limit.

**Fix:** Pre-tokenize data BEFORE model loading. Save Arrow cache. Load model into freed memory.

### 5.7 Long Samples Cause Tokenization Stall

**Symptom:** Single samples taking minutes to tokenize (589k tokens).

**Fix:** Filter dataset by character count: `len(text) <= 30000` (~10k tokens). Keeps 97% of samples, removes problematic outliers.

### 5.8 TRL `past_key_values` AttributeError

**Symptom:** `AttributeError: 'NemotronHOutput' object has no attribute 'past_key_values'`

**Fix:** Monkey-patch `NemotronHOutput.__getattr__` to map `past_key_values` → `cache_params`.

---

## 6. Data Pipeline

### Source

`verilog_db_v0.1.jsonl` — 38,417 Verilog modules scraped from OpenRoad and other sources.

### Processing

```python
# 1. Load raw JSONL
data = [json.loads(line) for line in open("verilog_db_v0.1.jsonl")]

# 2. Convert to text format
formatted = [{"text": d["code"]} for d in data]

# 3. Split train/val (97/3)
train = formatted[:37417]
val = formatted[37417:]

# 4. Filter long samples
MAX_CHARS = 30000  # ~10k tokens
train = [d for d in train if len(d["text"]) <= MAX_CHARS]
# Result: 36,321 train, 971 val
```

### Distribution

| Metric | Value |
|--------|-------|
| Median | 686 chars (~200 tokens) |
| P90 | 7,689 chars (~2,300 tokens) |
| P95 | 18,428 chars (~5,500 tokens) |
| Max (filtered) | 29,754 chars (~8,900 tokens) |
| Max (unfiltered) | 55,601,430 chars (~16M tokens!) |

### Filter Rationale

- `max_seq_length: 1024` tokens → samples >3k tokens get truncated anyway
- Filtering at 10k tokens keeps 97% of data while removing outliers that cause:
  - Slow tokenization (minutes per sample)
  - Memory spikes during packing
  - Wasted compute (truncated to 1024 tokens regardless)

---

## 7. Architecture Decisions

### Why Pre-Tokenize Before Model Loading

```
Container RAM: 500 GB cgroup limit
Model: 64 GB (bf16)
Tokenizer + Arrow datasets: ~2 GB
OS + Python: ~4 GB

SAFE ORDER:  Tokenizer (0.1GB) → Data (2GB) → Model (64GB) = ~66GB ✓
UNSAFE ORDER: Model (64GB) → Data (2GB + overhead) → OOM ✗
```

### Why `adamw_torch` Not `paged_adamw_8bit`

FSDP shards optimizer states across GPUs. `paged_adamw_8bit` uses BitsAndBytes paging which conflicts with FSDP's own DTensor sharding. `adamw_torch` is standard PyTorch and FSDP-compatible.

### Why No Activation Checkpointing

FSDP's `activation_checkpointing` creates DTensor wrappers around intermediate activations. NemotronH's custom forward pass (Mamba + MoE + Attention) doesn't produce activations with shapes FSDP expects, causing a silent hang or crash.

### ~~Why Packing Is Safe for Mamba~~ (CORRECTED — this was wrong)

The original claim ("SSM state resets at sequence boundaries, handled by attention mask") does **not** hold in this setup: the naive Mamba path scans straight across packed boundaries, eager attention applies no block-diagonal mask, and v0.1 packed documents with no EOS separators at all. This is root cause #3 of the SystemVerilog pass@1 regression — see [VERILOG_CPT_V0.1_POSTMORTEM.md](VERILOG_CPT_V0.1_POSTMORTEM.md).

---

## 8. Reproducing This Run

### Prerequisites

```bash
# Dependencies
pip install torch==2.8.0+cu128 transformers==5.13.1 peft==0.19.1 \
    trl==1.8.0 accelerate==1.14.0 datasets==5.0.0 omegaconf==2.3.1 \
    wandb==0.28.0 mamba-ssm==2.3.2.post1

# Model
huggingface-cli download NVIDIA/Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# Data (prepare as described in Section 6)
```

### Launch

```bash
cd nemotron-finetune

# Clean previous run
rm -rf results/output_fsdp_full/tokenize_cache results/output_fsdp_full/train.log

# Launch in tmux
tmux new-session -d -s train \
    "NCCL_TIMEOUT=7200 torchrun --nproc_per_node=2 --master_port=29503 \
    train_fsdp.py --config configs/cpt_verilog_fsdp.yaml 2>&1 | \
    tee results/output_fsdp_full/train.log"

# Monitor
tail -f results/output_fsdp_full/train.log
```

### Config File

```yaml
# configs/cpt_verilog_fsdp.yaml
run:
  mode: cpt
  name: verilog_cpt_v0.1
  output_dir: results/output_fsdp_full

model:
  path: models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  attn_implementation: eager
  use_mamba_kernels: false

data:
  train_path: data/verilog/train_full.json
  eval_path: data/verilog/val_full.json
  format: text
  max_seq_length: 1024
  packing: true

lora:
  enabled: true
  r: 8
  lora_alpha: 16
  target_modules:
    - in_proj
    - out_proj
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - up_proj
    - down_proj

training:
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 2
  learning_rate: 1e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  weight_decay: 0.05
  optim: adamw_torch
  save_steps: 1000        # ← FIXED: was 4677
  save_total_limit: 3
  gradient_checkpointing: false

wandb:
  mode: offline
  project: vaschpforge-llm
```

### Resuming

```bash
# The trainer auto-resumes from the latest checkpoint in output_dir
torchrun --nproc_per_node=2 train_fsdp.py --config configs/cpt_verilog_fsdp.yaml
```

---

## 9. Lessons Learned

1. **Pre-tokenize before model loading** — With a 64GB model on a 500GB cgroup, tokenizing after model load causes OOM.
2. **Filter long samples early** — Some code corpora have 500k+ token samples. Filter at 3x `max_seq_length` to keep 97% of data.
3. **PEFT `all-linear` broken on custom architectures** — Must manually discover target modules for NemotronH.
4. **FSDP activation checkpointing broken on NemotronH** — Disable it. Use more VRAM instead.
5. **`paged_adamw_8bit` incompatible with FSDP** — Use `adamw_torch`.
6. **Save checkpoints frequently** — `save_steps: 4677` only gave 2 checkpoints. Use `save_steps: 1000` or less.
7. **wandb needs env vars set early** — TRL doesn't inherit config values.
8. **Naive Mamba is slow but works** — First forward pass takes 10+ minutes, subsequent steps ~8s/it.
9. ~~**Packing is safe for Mamba** — State-space models don't have cross-sequence attention.~~ **CORRECTED:** wrong — see §7 and the [post-mortem](VERILOG_CPT_V0.1_POSTMORTEM.md).
10. **Eval loss tracks train loss** — No overfitting at this scale with LoRA r=8.

---

## 10. Next Steps (v0.2 Ideas)

| Experiment | Change | Expected Impact |
|------------|--------|-----------------|
| Higher LoRA rank | r=16 or r=32 | More capacity, possibly better convergence |
| Longer sequences | max_seq_length=2048 | Capture longer Verilog patterns |
| Multi-epoch | 3–5 epochs | More training signal |
| More frequent saves | save_steps=500 | More checkpoints for analysis |
| Eval loss monitoring | eval_strategy=steps, eval_steps=500 | Better training signal |
| Learning rate sweep | 5e-5, 2e-4 | Find optimal LR |
| DDP with quantization | 4-bit NF4 + DDP | Single-GPU inference testing |
| Instruction tuning | SFT on Verilog QA pairs | Better task-specific performance |

---

## 11. File Reference

```
results/output_fsdp_full/
├── resolved_config.yaml           # Full config snapshot
├── train.log                      # Complete training log
├── checkpoint-4677/               # Mid-training checkpoint
│   ├── adapter_model.safetensors  # LoRA weights (18 MB)
│   ├── adapter_config.json
│   └── trainer_state.json
├── checkpoint-7090/               # Final checkpoint
│   ├── adapter_model.safetensors  # LoRA weights (18 MB)
│   ├── adapter_config.json
│   └── trainer_state.json
└── tokenize_cache/                # Pre-tokenized Arrow cache
    ├── train.arrow
    └── eval.arrow
```

### HF Hub

```
https://huggingface.co/Ashx098/chipforge-llm/tree/main/phase_1/v0.1
├── README.md
├── checkpoint-4677/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── trainer_state.json
└── checkpoint-7090/
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── trainer_state.json
```
