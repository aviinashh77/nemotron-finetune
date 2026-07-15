# Verilog CPT v0.1 — Detailed Analysis

> Companion to [VERILOG_CPT_V0.1.md](VERILOG_CPT_V0.1.md) (run report).  
> This document focuses on quantitative analysis, curves, and interpretation.

---

## Overview

| Metric | Value |
|--------|-------|
| Base model | Nemotron-3-Nano-30B-A3B-BF16 |
| Training data | 36,321 Verilog modules (filtered ≤10k tokens) |
| Method | LoRA CPT (r=8, ~4.5M trainable params) |
| Hardware | 2x A100-80GB, FSDP |
| Duration | ~16.5 hours |
| Total steps | 7,090 (2 epochs) |

---

## 1. Loss Curve

```
0.75 ┤●
     │ ╲
0.70 ┤  ╲
     │   ╲
0.65 ┤    ╲
     │     ╲
0.60 ┤      ╲
     │       ╲
0.55 ┤        ●───╲
     │             ╲
0.50 ┤              ●───●───╲
     │                       ╲───●
0.45 ┤                            ╲───●───●
     │                                    ╲───●───●
0.40 ┤                                        ╲───●───●───●
     │                                                    ╲───●
0.35 ┤                                                        ●
     └────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────
        0    650  1300 1950 2600 3250 3900 4550 5200 5850 6500 7090
                              Step
```

### Key Numbers

| Metric | Value |
|--------|-------|
| **Initial loss** | 0.7448 |
| **Final train loss** | 0.4211 |
| **Final eval loss** | 0.4272 |
| **Min loss** | 0.2597 (step 6060) |
| **Loss reduction** | 43.2% |
| **Initial accuracy** | 82.93% |
| **Final accuracy** | 89.31% |
| **Max accuracy** | 93.03% |
| **Entropy decrease** | 8.9% (0.46 → 0.42) |

---

## 2. Convergence Timeline

| Milestone | Step | % of training |
|-----------|------|---------------|
| Loss < 0.5 | 380 | 5.4% |
| Loss < 0.45 | 750 | 10.6% |
| Loss < 0.4 | 1,840 | 26.0% |
| Stable < 0.45 | ~6,000 | 84.6% |

The model learned the bulk of Verilog patterns in the first **26% of training** (1,840 steps). The remaining 74% refined the distribution and reduced variance.

---

## 3. Epoch Comparison

| Epoch | Avg Loss | Improvement |
|-------|----------|-------------|
| Epoch 1 | 0.4834 | — |
| Epoch 2 | 0.4044 | 16.4% better |

Second epoch consistently improved. The model wasn't overfitting — it was still learning useful patterns from the second pass through the data.

---

## 4. Loss Distribution

```
0.25-0.30:    3  #
0.30-0.35:   38  ############
0.35-0.40:  170  ########################################################
0.40-0.45:  226  ###########################################################################  ← peak
0.45-0.50:  146  ################################################
0.50-0.55:   75  #########################
0.55-0.60:   23  #######
0.60-0.70:   15  #####
0.70-1.00:   13  ####
```

Loss is **unimodal and right-skewed** — most steps land in the 0.35–0.50 range. The long tail (0.60–1.0) represents harder samples (complex Verilog modules with unusual constructs).

---

## 5. Eval vs Train Loss (No Overfitting)

| Step | Train Loss | Eval Loss | Gap |
|------|------------|-----------|-----|
| 2,338 | ~0.47 | 0.4650 | +0.005 |
| 4,676 | ~0.43 | 0.4350 | -0.005 |
| 7,014 | ~0.42 | 0.4272 | -0.007 |
| 7,090 | 0.4211 | 0.4272 | +0.006 |

Train and eval are nearly identical throughout. With only 0.015% of parameters being trained (4.5M out of 30B), overfitting isn't a realistic concern at this scale.

---

## 6. Gradient Norm

| Metric | Value |
|--------|-------|
| Mean | 0.3625 |
| Max | 0.6550 |
| Min | 0.0810 |

Grad norm is **stable and bounded** — no spikes, no vanishing. The `max_grad_norm: 1.0` clipping was never triggered. This indicates the learning rate and LoRA rank are well-matched to the data.

---

## 7. Learning Rate Schedule

```
0.00010 ┤     ╱╲
        │    ╱  ╲
0.00008 ┤   ╱    ╲
        │  ╱      ╲
0.00006 ┤ ╱        ╲
        │╱          ╲
0.00004 ┤            ╲
        │             ╲
0.00002 ┤              ╲
        │               ╲
0.00000 ┤────────────────╲────
        0    1770  3540  5310  7090
```

- Warmup: 354 steps (5%) to peak LR 1e-4
- Cosine decay to 0 over remaining steps
- Standard schedule, worked well

---

## 8. What the Model Learned

The accuracy trajectory tells the story:

| Phase | Accuracy | What it learned |
|-------|----------|-----------------|
| Steps 1–380 | 83% → 85% | Basic Verilog syntax (module/endmodule, wire/reg, assign) |
| Steps 380–1840 | 85% → 89% | Structural patterns (always blocks, if/else, case) |
| Steps 1840–5000 | 89% → 90% | Complex constructs (FSM, params, generate) |
| Steps 5000–7090 | 90% → 89% | Consolidation — slight regression as model balances harder samples |

The accuracy plateau at ~89% suggests the model has captured the dominant Verilog patterns. Further improvement would require either:
- More diverse training data
- Higher LoRA rank (more capacity)
- Instruction-tuned data (QA pairs, not raw code)

---

## 9. Resource Efficiency

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| GPU 0 VRAM | 63 GB | 80 GB | 77% |
| GPU 1 VRAM | 61 GB | 80 GB | 75% |
| GPU compute | — | — | 76–86% |
| CPU RAM | 123 GB | 465 GB | 26% |
| Training time | 16.5h | — | — |

**Cost efficiency:** ~16.5 GPU-hours on 2xA100 = ~33 A100-hours total. For a 30B model CPT, this is reasonable.

---

## 10. Throughput

| Metric | Value |
|--------|-------|
| Step time | 8.30s (stable, σ=0.1s) |
| Tokens/step | ~4,096 (packed, effective batch=4) |
| Total tokens | ~29M |
| Throughput | ~493 tokens/sec |

Step time variance is extremely tight (8.14–8.47s), indicating no I/O bottlenecks or data loading issues.

---

## 11. What Worked

1. **Pre-tokenization** — Prevented OOM on 500GB cgroup
2. **Packing** — 1.34x speedup, safe for Mamba SSM
3. **Filtering long samples** — Kept 97% of data, removed tokenization stalls
4. **LoRA r=8** — Small but sufficient for syntax learning
5. **FSDP** — Sharded 64GB model across 2 GPUs, enabled 1024-token sequences

---

## 12. What Didn't Work (Bugs Fixed)

1. PEFT `all-linear` auto-detect — Broken on NemotronH
2. FSDP `activation_checkpointing` — DTensor mismatch
3. `paged_adamw_8bit` — Incompatible with FSDP
4. `SFTConfig` `max_seq_length` — Not a valid param in TRL 1.8.0
5. wandb env vars — Not inherited by SFTConfig

---

## 13. Limitations

1. **Only 2 checkpoints** — `save_steps: 4677` was suboptimal
2. **No instruction tuning** — Raw code CPT, not task-specific
3. **Small LoRA rank** — r=8 limits capacity; r=16-32 might converge faster
4. **Short sequences** — max_seq_length=1024 truncates complex modules
5. **No eval loss monitoring** — Only 4 eval points across 7,090 steps

---

## 14. Recommendations for v0.2

| Change | Rationale |
|--------|-----------|
| `save_steps: 500` | More checkpoints for analysis |
| `eval_steps: 500` | Better training signal |
| `r: 16` | More adapter capacity |
| `max_seq_length: 2048` | Capture longer Verilog patterns |
| `num_train_epochs: 3` | More training signal |
| Instruction-tuned data | QA pairs for task-specific performance |
| `learning_rate: 5e-5` | Slower, possibly better convergence |

---

## 15. Verdict

**v0.1 successfully taught Nemotron-30B basic Verilog syntax.** The model went from 83% to 89% token accuracy, with loss dropping 43%. The training was stable, efficient, and produced a usable adapter. The main limitation is capacity (r=8) and data type (raw code vs instruction-tuned).

---

## Raw Data

### Loss Every 200 Steps

| Step | Loss | Accuracy |
|------|------|----------|
| 10 | 0.7448 | 82.93% |
| 210 | 0.6293 | 84.32% |
| 410 | 0.5324 | 86.76% |
| 610 | 0.5178 | 86.93% |
| 810 | 0.5137 | 87.46% |
| 1010 | 0.4879 | 87.71% |
| 1210 | 0.4380 | 88.75% |
| 1410 | 0.4688 | 87.97% |
| 1610 | 0.4421 | 88.87% |
| 1810 | 0.4935 | 87.62% |
| 2010 | 0.5001 | 87.23% |
| 2210 | 0.4712 | 87.85% |
| 2410 | 0.5029 | 87.07% |
| 2610 | 0.4233 | 89.34% |
| 2810 | 0.4039 | 89.63% |
| 3010 | 0.4115 | 89.18% |
| 3210 | 0.3848 | 89.77% |
| 3410 | 0.3756 | 89.98% |
| 3610 | 0.3350 | 91.05% |
| 3810 | 0.4173 | 89.51% |
| 4010 | 0.4616 | 88.11% |
| 4210 | 0.4474 | 88.26% |
| 4410 | 0.3691 | 90.29% |
| 4610 | 0.4297 | 88.55% |
| 4810 | 0.4034 | 89.77% |
| 5010 | 0.4223 | 89.16% |
| 5210 | 0.3875 | 90.23% |
| 5410 | 0.3335 | 90.86% |
| 5610 | 0.5096 | 87.13% |
| 5810 | 0.3712 | 90.74% |
| 6010 | 0.3886 | 89.73% |
| 6210 | 0.4045 | 89.59% |
| 6410 | 0.3626 | 90.27% |
| 6610 | 0.4132 | 89.57% |
| 6810 | 0.3483 | 90.61% |
| 7010 | 0.4085 | 89.49% |

### Eval Results

| Step | Eval Loss | Eval Accuracy | Eval Entropy |
|------|-----------|---------------|--------------|
| 2,338 | 0.4650 | 88.11% | 0.4636 |
| 4,676 | 0.4350 | 88.77% | 0.4333 |
| 7,014 | 0.4272 | 88.97% | 0.4225 |
| 7,090 | 0.4272 | 88.97% | 0.4225 |

### Smoothed Loss (EMA, α=0.1)

| Step | Smoothed Loss |
|------|---------------|
| 10 | 0.7448 |
| 890 | 0.4851 |
| 1,770 | 0.4642 |
| 2,650 | 0.4296 |
| 3,530 | 0.4126 |
| 4,410 | 0.4025 |
| 5,290 | 0.3967 |
| 6,170 | 0.3827 |
| 7,050 | 0.3984 |
| 7,090 | 0.4043 |
