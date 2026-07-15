# Experiments

All experiments use **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** on **2x A100-SXM4-80GB**.

## Experiment 1: DDP LoRA CPT — 2-Step Smoke Test

| Parameter | Value |
|-----------|-------|
| Config | `configs/cpt_dummy.yaml` |
| Script | `train.py` (DDP) |
| Data | `data/dummy/sample_cpt_text.json` (10 examples) |
| Seq len | 512 |
| Batch size | 1 per device |
| Gradient accum | 2 |
| Max steps | 2 |
| LoRA r | 16 |

**Results:**
```
Step 1: loss=1.1448, accuracy=64.06%, grad_norm=2.424
Step 2: loss=0.8711, accuracy=71.88%, grad_norm=2.048
```

**Status:** PASSED — Model trains, loss decreases, LoRA adapters save correctly.

---

## Experiment 2: DDP LoRA CPT — Longer Run (50 steps)

| Parameter | Value |
|-----------|-------|
| Config | `configs/cpt_long.yaml` |
| Script | `train.py` (DDP) |
| Data | `data/dummy/sample_cpt_100.json` (118 examples) |
| Seq len | 512 |
| Max steps | 75 (attempted) |

**Results:**
```
Step 10: loss=1.4314, accuracy=64.06%
Step 20: loss=0.6577, accuracy=79.69%
Step 30: loss=0.4250, accuracy=85.16%
Step 40: loss=0.3345, accuracy=89.06%
Step 50: OOM — torch.OutOfMemoryError: CUDA out of memory
```

**Peak VRAM:** ~70.5 GB per GPU at seq_len=512

**Analysis:** DDP replicates the full 64 GB bf16 model on each GPU. After model + LoRA + optimizer states, only ~10 GB remains for activations. With naive Mamba (no `causal_conv1d`), activation memory is ~3x larger. OOM occurs during backward pass when activation memory peaks.

**Status:** PARTIAL — Training works but OOMs at step 50 with batch_size=1, seq_len=512.

---

## Experiment 3: SFT — 5 Chat Examples

| Parameter | Value |
|-----------|-------|
| Config | `configs/sft_sample.yaml` |
| Script | `train.py` (DDP) |
| Data | `data/dummy/sample_chat_5.json` (5 examples) |
| Seq len | 1024 |
| Epochs | 2 |

**Results:**
```
4 training steps, final loss=6.32
Training time: 168 seconds
```

**Status:** PASSED — SFT works end-to-end on chat data.

---

## Experiment 4: FSDP — Smoke Test (seq_len=512)

| Parameter | Value |
|-----------|-------|
| Config | `configs/cpt_fsdp.yaml` |
| Script | `train_fsdp.py` (FSDP) |
| Data | `data/dummy/sample_cpt_100.json` |
| Seq len | 512 |
| Max steps | 5 |

**Results:**
```
Step 1: loss=1.781, accuracy=64.06%, grad_norm=1.814
Step 2: loss=1.551, accuracy=47.66%, grad_norm=1.712
Step 3: loss=1.018, accuracy=75.39%, grad_norm=1.546
Step 4: loss=1.266, accuracy=69.14%, grad_norm=1.592
Step 5: loss=1.275, accuracy=66.02%, grad_norm=1.679
```

**Peak VRAM:** ~46.8 GB per GPU (vs 70.5 GB with DDP)

**Status:** PASSED — FSDP reduces per-GPU memory by ~24 GB at seq_len=512.

---

## Experiment 5: FSDP — Long Sequence (seq_len=3072)

| Parameter | Value |
|-----------|-------|
| Script | `train_fsdp.py` (FSDP) |
| Data | `data/dummy/sample_cpt_long.json` (200 examples, ~4k tokens each) |
| Seq len | 3072 |
| Max steps | 5 |
| Activation checkpointing | FSDP-native |

**Results:**
```
Step 1: loss=0.249, accuracy=92.58%, grad_norm=0.089
Step 2: loss=0.249, accuracy=92.58%, grad_norm=0.090
Step 3: loss=0.202, accuracy=93.75%, grad_norm=0.085
Step 4: loss=0.130, accuracy=96.48%, grad_norm=0.085
Step 5: loss=0.094, accuracy=97.27%, grad_norm=0.079
```

**Peak VRAM:** 67.7 GB per GPU (12.3 GB headroom)

**Status:** PASSED — 6x sequence length improvement over DDP (512 → 3072).

---

## Experiment 6: FSDP — Maximum Sequence (seq_len=4096)

| Parameter | Value |
|-----------|-------|
| Script | `train_fsdp.py` (FSDP) |
| Data | `data/dummy/sample_cpt_long.json` |
| Seq len | 4096 |
| Max steps | 3 |

**Results:**
```
Step 1: loss=0.1875, accuracy=94.53%
Step 2: loss=0.1875, accuracy=94.53%
Step 3: loss=0.1514, accuracy=95.70%
```

**Peak VRAM:** 76.9 GB per GPU (3.1 GB headroom)

**Status:** PASSED — 8x sequence length improvement over DDP.

---

## Experiment 7: FSDP — Target Sequence (seq_len=5120)

| Parameter | Value |
|-----------|-------|
| Script | `train_fsdp.py` (FSDP) |
| Data | `data/dummy/sample_cpt_long.json` |
| Seq len | 5120 |
| Max steps | 2 |

**Results:**
```
Step 1: loss=0.1924, accuracy=94.92%
Step 2: loss=0.1924, accuracy=94.92%
```

**Peak VRAM:** 78.0 GB per GPU (2.0 GB headroom)

**Status:** PASSED — 10x sequence length improvement over DDP. This is the practical limit on 2xA100-80GB.

---

## Experiment 8: Verilog CPT v0.1 — Full Run (in progress)

| Parameter | Value |
|-----------|-------|
| Config | `configs/cpt_verilog_fsdp.yaml` |
| Script | `train_fsdp.py` (FSDP) |
| Data | `data/verilog/train_full.json` (36,321 samples, filtered ≤30k chars) |
| Eval | `data/verilog/val_full.json` (971 samples) |
| Seq len | 1024 |
| Epochs | 1 |
| Batch size | 2 per device |
| Gradient accum | 4 |
| Effective batch | 16 (2 × 4 × 2 GPUs) |
| Max steps | 7,090 |
| LoRA r | 8 |
| Learning rate | 1e-4 (cosine, 5% warmup) |
| Optimizer | adamw_torch |
| wandb | offline, project `vaschpforge-llm` |
| Packing | true (1.34x speedup) |

**Status:** RUNNING — 28% complete (1,968/7,090 steps), ETA ~12 hours

**Results so far:**
```
Step 1780: loss=0.4858, accuracy=88.03%, epoch=0.50
Step 1800: loss=0.4005, accuracy=89.57%, epoch=0.51
Step 1820: loss=0.4462, accuracy=88.85%, epoch=0.51
Step 1840: loss=0.3463, accuracy=90.96%, epoch=0.52
Step 1860: loss=0.4731, accuracy=87.89%, epoch=0.53
Step 1880: loss=0.4755, accuracy=87.95%, epoch=0.55
Step 1900: loss=0.4349, accuracy=88.71%, epoch=0.54
Step 1920: loss=0.3807, accuracy=90.35%, epoch=0.54
Step 1940: loss=0.3689, accuracy=90.74%, epoch=0.55
Step 1960: loss=0.4286, accuracy=88.98%, epoch=0.55
```

**Peak VRAM:** 63GB / 80GB per GPU
**Speed:** ~8.3s/it

**Data stats:**
- Original: 37,417 samples
- After filtering (≤30k chars / ~10k tokens): 36,321 train, 971 val
- Dropped: 1,096 samples (2.9%) — mostly >10k token boilerplate/duplicates
- Max chars in filtered set: 29,754 (train), 29,638 (val)

**Lessons learned:**
1. Tokenizing BEFORE model loading prevents OOM (see ARCHITECTURE.md)
2. Some Verilog samples are 589k tokens — must filter long samples
3. `SFTConfig` in TRL 1.8.0 does NOT accept `max_seq_length` (use `max_length`)
4. `paged_adamw_8bit` incompatible with FSDP/DTensor — use `adamw_torch`
5. FSDP `activation_checkpointing` causes DTensor mismatch — remove from config

---

## Summary: DDP vs FSDP Memory Scaling

| Seq Len | DDP Peak VRAM | FSDP Peak VRAM | FSDP Headroom |
|---------|---------------|----------------|---------------|
| 512 | 70.5 GB (OOM) | ~46 GB | 34 GB |
| 2048 | — | ~47 GB | 33 GB |
| 3072 | — | 67.7 GB | 12.3 GB |
| 4096 | — | 76.9 GB | 3.1 GB |
| 5120 | — | 78.0 GB | 2.0 GB |

**Key insight:** FSDP shards the 64 GB model across GPUs (~32 GB each), freeing ~32 GB per GPU for activations. This enables 10x longer sequences.

## Scaling Projections

| Hardware | Max Seq Len (FSDP, bf16) | Notes |
|----------|--------------------------|-------|
| 2xA100-80GB | ~5120 | Current limit (2 GB headroom) |
| 4xA100-80GB | ~8192 | Each GPU holds ~16 GB model shard |
| 8xA100-80GB | ~12288 | Each GPU holds ~8 GB model shard |
| 2xA100-80GB + 4-bit | ~7168 | Quantization frees ~48 GB total |

## Known Limitations

1. **Naive Mamba path**: Without `causal_conv1d`, activation memory is ~3x larger. Installing it would roughly double the achievable seq_len.
2. **FSDP activation_checkpointing**: Broken with NemotronH DTensor — must disable. Uses more activation memory.
3. **No eval loss**: Verilog CPT uses `eval_strategy: no`. Loss is monitored via train logs only.
4. **Single training mode**: Only CPT has been extensively tested. SFT and DAPT are structurally identical but not tested at long sequences.
5. **500GB cgroup limit**: Container has 500GB RAM limit. Tokenizing after model load causes OOM. Pre-tokenization solves this.
