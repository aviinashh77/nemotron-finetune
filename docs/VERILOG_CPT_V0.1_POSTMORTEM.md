# Verilog CPT v0.1 Post-Mortem — Why pass@1 Dropped While Token Accuracy Rose

> **Status:** Root-cause analysis complete (July 2026)
> **Companion docs:** [VERILOG_CPT_V0.1.md](VERILOG_CPT_V0.1.md) (run report), [VERILOG_CPT_V0.1_ANALYSIS.md](VERILOG_CPT_V0.1_ANALYSIS.md) (curves)
> **Fixes applied:** `train.py`, `train_fsdp.py`, `configs/base.yaml`, new [`configs/cpt_verilog_v0.2.yaml`](../configs/cpt_verilog_v0.2.yaml)

---

## 1. The symptom

| Metric | Base | After CPT LoRA v0.1 |
|---|---|---|
| SystemVerilog eval pass@1 | **56%** | **42%** |
| SystemVerilog eval pass@5 | 72% | 71% |
| Training token accuracy | 81% | 89% |

This pattern — **pass@1 collapses, pass@5 intact, token accuracy up** — is the signature of a model whose *capability survived* but whose *output distribution got skewed*. The knowledge to solve the tasks is still in the weights (pass@5 proves it); the most-likely completion the model now emits first is more often wrong. Token accuracy measures "predict the next token of the training corpus," which the run genuinely improved — it is **not** a proxy for generation quality on a benchmark, and v0.1 is the demonstration.

Also note the eval entropy dropped 8.9% during training: the distribution sharpened toward the CPT corpus.

## 2. Root causes, ranked

### #1 — LoRA on the Mamba SSM backbone (`in_proj`, `out_proj`)

v0.1 targeted `in_proj`/`out_proj` (the Mamba2 mixer projections) alongside attention and MLP. External evidence says this is the highest-risk choice available:

- **"Where Should LoRA Go?" ([arXiv 2604.22127](https://arxiv.org/html/2604.22127))**: on *sequential* hybrid models (NemotronH is sequential), attention-only LoRA consistently outperformed broader placement with 5–10× fewer parameters, while adapting the recurrent (SSM) backbone caused **catastrophic degradation** (−14.8pp GSM8K in their study). Recommendation: default to attention-only; avoid recurrent-backbone adaptation.
- **MambaPEFT ([arXiv 2411.03855](https://arxiv.org/pdf/2411.03855))**: tuning SSM internals together with projections degrades performance; avoid `conv1d`/`A_log`/`D`.
- **Fused-kernel bypass ([huggingface/peft#2274](https://github.com/huggingface/peft/issues/2274))**: in the fused `mamba_split_conv1d_scan_combined` training path, `out_proj` is passed as a raw weight (`outproj_weight=self.out_proj.weight`) — the module's forward is never called, so a PEFT adapter on it is **silently skipped**. v0.1 ran the naive path (`use_mamba_kernels: false`), so its adapter *was* active — but any future run that enables kernels while targeting `out_proj` trains an adapter that inference bypasses (or vice versa), and the path taken even flips batch-by-batch depending on padding.
- NVIDIA's own Megatron-Bridge LoRA defaults for hybrid Nemotron do include `in_proj`/`out_proj` — but inside Megatron's kernels ([docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html)), which does not validate the same targeting under HF PEFT.

**Fix (applied):** attention-only default (`q_proj,k_proj,v_proj,o_proj`); loud warning if a config targets Mamba modules.

### #2 — Narrow-corpus raw-text CPT with zero replay, on an aligned model, at LR 1e-4

The base model is a post-trained (aligned/reasoning) model. Two epochs of raw Verilog files with no general-data mixing pushes it toward "raw file continuation" mode and erodes the instruction-following that the benchmark harness depends on.

- **ChipNeMo ([arXiv 2311.00176](https://arxiv.org/pdf/2311.00176))** — NVIDIA's own chip-domain adaptation — used a *small* LR (large LR caused degradation across all non-coding benchmarks) and blended ~9% general data (Wikipedia + GitHub code) specifically to preserve general ability.
- CPT literature standard: 10–30% general-data replay; >90% domain data risks forgetting ([CMR scaling law, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.903.pdf)).
- **"LoRA Learns Less and Forgets Less" ([arXiv 2405.09673](https://arxiv.org/abs/2405.09673))**: LoRA forgets less than full FT, but nothing says narrow LoRA CPT doesn't degrade adjacent-domain generation — it commonly does.

**Fix (applied):** `data.replay_path`/`data.replay_ratio` mixing support; v0.2 LR 2e-5.

### #3 — Packing with no document boundaries

`_tokenize_fn` tokenized raw text **without appending EOS**, then TRL packed the pre-tokenized ids. Consequences:

- No separator token between documents → the model never learns where a module ends → damaged termination behavior at generation time.
- With `attn_implementation: eager` there is no varlen/block-diagonal masking, and the naive Mamba path carries SSM state straight across packed boundaries. Median doc ≈ 200 tokens packed into 1024-token rows → each training row was ~5 unrelated modules bleeding into each other.
- ~17% of docs exceeded 1024 tokens and were truncated mid-module *before* packing → "module cut off, unrelated module follows" was a pervasive training pattern.

The v0.1 docs claimed "Packing is safe for Mamba — SSM state resets at sequence boundaries" and "cross-contamination warning can be ignored for CPT." **Both claims were wrong for this setup** (corrected in those docs). Packing itself is fine — pretraining pipelines pack — but only with document boundaries present.

**Fix (applied):** `data.append_eos` (default true), EOS appended per document before packing.

### #4 — Corpus/dialect mismatch

The corpus is plain Verilog (OpenROAD-scraped, composition unaudited) while the benchmark is **SystemVerilog**. Curated Verilog corpora (e.g., [VerilogDB, arXiv 2507.13369](https://arxiv.org/html/2507.13369v1)) explicitly filter SystemVerilog out — CPT on such a corpus actively pulls the model toward a different dialect than the one being evaluated. If the corpus contains synthesized netlists or tool-generated code (typical of OpenROAD flows), it also teaches flat gate-level style rather than behavioral RTL.

**Action (open, before v0.2):** audit a sample of `verilog_db_v0.1.jsonl` for netlist/tool-generated share; add SystemVerilog data given the eval target.

### #5 — Sequence length 1024 (contributor, not primary)

Nemotron 3 Nano was pretrained at seq len 8192 (25T tokens), with a long-context phase mixing 4k/512k ([tech report, arXiv 2512.20848](https://arxiv.org/pdf/2512.20848)). NVIDIA's report documents that CPT at a *single out-of-distribution length* shifted benchmark behavior until lengths were mixed. 1024 is 8× below pretraining length and truncates real RTL files badly. SSM state statistics are length-dependent ([arXiv 2509.19633](https://arxiv.org/pdf/2509.19633)).

**Fix (applied):** v0.2 at 4096 (8192 if VRAM allows after kernel install).

## 3. Memory & throughput (separate problem, not a cause of the regression)

63 GB/GPU at seq 1024, batch 1, and ~493 tok/s on 2×A100 came from a stack of known issues:

| Factor | Evidence |
|---|---|
| FSDP fails to shard Mamba2/hybrid layers | [transformers#36982](https://github.com/huggingface/transformers/issues/36982) — identical 70–80GB signature on a 2.7B Mamba2; the new `_make_sharding_diag_callback` in `train_fsdp.py` reports per-rank param bytes at step 1 so this is visible immediately |
| Naive Mamba fallback (`use_mamba_kernels: false`) | `torch_forward` upcasts to fp32 and materializes chunked-scan intermediates — ~3× activation memory and dominant slowdown; the CUDA 13.0 `causal_conv1d` build failure is a version-pinning problem ([Unsloth Nemotron 3 guide](https://unsloth.ai/docs/models/nemotron-3) has known-good pins), or downgrade the container to CUDA 12.8 to match `torch 2.8.0+cu128` |
| Activation checkpointing fully disabled | v0.1 workaround for the FSDP-native AC DTensor crash; v0.2 uses HF non-reentrant checkpointing instead (`gradient_checkpointing_kwargs={"use_reentrant": False}`) — a different mechanism, to be validated on GPU |
| Eager attention | No flash-attn2 for NemotronH; unavoidable for now |
| No QLoRA escape hatch | bitsandbytes 4-bit breaks on the mamba/expert layers of this hybrid MoE ([details](https://github.com/unslothai/unsloth/discussions/3810)) — full BF16 must be loaded |

transformers is already 5.13.1 ≥ 5.11.0, so the NemotronH `out_proj` dtype bug (fixed in [PR #46487](https://github.com/huggingface/transformers/releases/tag/v5.11.0)) was **not** in play.

If HF-stack memory remains hostile after the kernel install + checkpointing, the supported alternative is NVIDIA's NeMo AutoModel / Megatron-Bridge path (TP=4, EP=8, sequence parallel for the hybrid MoE) — [docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/nemotron3.html), [blog](https://huggingface.co/blog/nvidia/accelerating-fine-tuning-nvidia-nemo-automodel).

## 4. GPU-session validation checklist (run BEFORE the full v0.2)

1. **Kernels**: install pinned `mamba-ssm` + `causal-conv1d`; import both; run 1 batch and assert the log shows no naive/`torch_forward` fallback.
2. **Sharding**: 50-step smoke run at seq 4096 — check the `[FSDP diag]` line: per-rank param bytes should be ≈ model_size/2, and VRAM well under v0.1's 63GB; step time should be ≪ 8.3s/it.
3. **Checkpointing**: confirm non-reentrant gradient checkpointing runs (no DTensor hang); if it hangs, set `training.gradient_checkpointing: false` and drop seq len.
4. **Replay data**: prepare `data/replay/general_code.json` (~20% of mix; general code + some instruction/chat-formatted text) — the v0.2 config will not run without it.
5. **Mid-run decode**: at the first checkpoint, decode ~5 prompts — does it still follow instructions? Do generated modules terminate cleanly at `endmodule`?
6. **Benchmark early**: run the SystemVerilog benchmark base-vs-adapter at ~25% of training before committing to full epochs.

## 5. Corrected mental model

- **Token accuracy / train loss measure corpus fit, not downstream quality.** v0.1's "83%→89%, no overfitting, verdict: success" read was wrong precisely because the only metrics tracked were corpus-fit metrics. Always benchmark base-vs-adapter on the actual downstream task before declaring a run good.
- **pass@1 down + pass@5 flat = distribution skew, not capability loss** — look for causes that reshape the output distribution (alignment erosion, dialect shift, boundary artifacts), not destroyed weights.
- **A hybrid SSM model is not a drop-in transformer for PEFT.** Module choice, kernel paths, packing semantics, and FSDP wrapping all behave differently.
