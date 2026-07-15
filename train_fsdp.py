#!/usr/bin/env python
"""FSDP-based LoRA finetuning entrypoint for Nemotron models.

Parallel path to train.py (DDP). Shards model weights across GPUs via
Fully Sharded Data Parallel, enabling long-sequence CPT (5k-10k seq_len)
on 2xA100-80GB by distributing ~64GB bf16 weights (~32GB each).

Usage:
    torchrun --nproc_per_node=2 train_fsdp.py --config configs/cpt_fsdp.yaml
    torchrun --nproc_per_node=2 train_fsdp.py --config configs/cpt_fsdp.yaml \\
        training.per_device_train_batch_size=1 \\
        data.max_seq_length=7168
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description="Nemotron LoRA finetuning (FSDP)")
    p.add_argument("--config", required=True, help="Path to run YAML config")
    p.add_argument("overrides", nargs="*", help="Dotlist overrides")
    return p.parse_args()


def load_data(cfg):
    from datasets import load_dataset

    train_path = cfg.data.train_path
    if train_path is None:
        raise ValueError("data.train_path is required")

    data_files = {"train": str(train_path)}
    if cfg.data.get("eval_path"):
        data_files["eval"] = str(cfg.data.eval_path)

    raw = load_dataset("json", data_files=data_files)
    return raw


def format_chat(example, tokenizer):
    messages = example.get("messages", [])
    if not messages:
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_prompt_completion(example, tokenizer):
    prompt = example.get("prompt", "")
    completion = example.get("completion", "")
    text = f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"
    return {"text": text}


def setup_model(cfg):
    """Load model on CPU with LoRA applied. SFTTrainer handles FSDP wrapping."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = cfg.model.path
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0
    if is_main:
        print(f"Loading model from: {model_path}")

    # Load on CPU — FSDP will shard from here to each GPU
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": getattr(torch, cfg.model.get("torch_dtype", "bfloat16")),
        "device_map": "cpu",
    }
    if cfg.model.get("attn_implementation"):
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if not cfg.model.get("use_cache", True):
        model.config.use_cache = False

    if cfg.model.get("use_mamba_kernels") is not None:
        model.config.use_mamba_kernels = cfg.model.use_mamba_kernels
        if is_main:
            print(f"use_mamba_kernels set to {cfg.model.use_mamba_kernels}")

    # Patch NemotronHOutput for TRL past_key_values compat
    _patch_nemotron_output(is_main)

    # Apply LoRA before FSDP
    if cfg.lora.get("enabled", True):
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.get("lora_dropout", 0.05),
            bias=cfg.lora.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )

        if cfg.lora.target_modules == "all-linear":
            import torch.nn as nn
            # PEFT auto-detect fails on custom NemotronH — discover manually
            linear_names = sorted({
                name.split(".")[-1]
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear) and name.split(".")[-1] != "lm_head"
            })
            # Attention + Mamba projections only (~92 layers) — skip MoE experts
            priority = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]
            targets = [n for n in priority if n in linear_names]
            if len(targets) < 3:
                targets += [n for n in linear_names if n not in targets][:3]
            lora_config.target_modules = targets
            if is_main:
                print(f"LoRA targeting: {targets}")
        else:
            lora_config.target_modules = list(cfg.lora.target_modules)

        model = get_peft_model(model, lora_config)
        if is_main:
            model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _patch_nemotron_output(is_main: bool):
    """Patch NemotronHOutput to expose past_key_values for TRL."""
    try:
        _patched_classes = set()
        for _mod_name in list(sys.modules.keys()):
            _mod = sys.modules.get(_mod_name)
            if _mod is None:
                continue
            for _cls_name in ("NemotronHOutput", "NemotronHCausalLMOutput"):
                _cls = getattr(_mod, _cls_name, None)
                if _cls is not None and id(_cls) not in _patched_classes:
                    _patched_classes.add(id(_cls))
                    _orig_getattr = getattr(_cls, '__getattr__', None)
                    def _make_pkv_getattr(orig_ga):
                        def _pkv_getattr(self, name):
                            if name == 'past_key_values':
                                return getattr(self, 'cache_params', None)
                            if orig_ga is not None:
                                return orig_ga(self, name)
                            raise AttributeError(name)
                        return _pkv_getattr
                    _cls.__getattr__ = _make_pkv_getattr(_orig_getattr)
                    if is_main:
                        print(f"Patched {_cls_name} for TRL past_key_values compatibility")
    except Exception as e:
        if is_main:
            print(f"Warning: Could not patch NemotronH output classes: {e}")


def main():
    args = parse_args()

    # Initialize distributed early with 1-hour NCCL timeout
    os.environ.setdefault("NCCL_TIMEOUT", "3600")
    timeout = timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", "3600")))
    dist.init_process_group(backend="nccl", timeout=timeout)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    from nemotron_finetune.config import load_config
    cfg = load_config(args.config, args.overrides if args.overrides else None)

    # Set wandb env vars before trainer init (so wandb.init picks them up)
    if cfg.wandb.get("mode", "disabled") != "disabled":
        os.environ["WANDB_PROJECT"] = cfg.wandb.get("project", "nemotron-finetune")
        if cfg.wandb.get("run_name"):
            os.environ["WANDB_NAME"] = cfg.wandb["run_name"]
        # Offline mode with large buffer so logs persist through crashes
        if cfg.wandb.get("mode") == "offline":
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_BUFFER_SIZE"] = "100"  # MB — flush frequently

    from nemotron_finetune.logging_utils import setup_logging
    logger = setup_logging(cfg.run.output_dir, cfg.logging.get("level", "INFO"))
    logger.info("Config loaded (FSDP): %s", cfg.run.config_path)

    from nemotron_finetune.config import save_resolved
    save_resolved(cfg, os.path.join(cfg.run.output_dir, "resolved_config.yaml"))

    # ---- Phase 1: Tokenize data BEFORE loading model (500GB container limit) ----
    # Tokenizing 37k samples + building labels while model is in memory OOMs.
    # Fix: load tokenizer only, format + tokenize, save Arrow cache, then load model.
    logger.info("Loading tokenizer for pre-tokenization...")
    from transformers import AutoTokenizer
    model_path = cfg.model.path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading data...")
    raw_dataset = load_data(cfg)
    logger.info("Dataset loaded: %d examples", len(raw_dataset))

    # Format data (chat template or prompt_completion)
    data_format = cfg.data.get("format", "chat")
    num_proc = min(cfg.data.get("num_proc", 1), 2)
    if data_format == "chat":
        raw_dataset = raw_dataset.map(
            lambda ex: format_chat(ex, tokenizer),
            num_proc=num_proc,
            desc="Formatting chat data",
        )
    elif data_format == "prompt_completion":
        raw_dataset = raw_dataset.map(
            lambda ex: format_prompt_completion(ex, tokenizer),
            num_proc=num_proc,
            desc="Formatting prompt/completion data",
        )

    train_dataset = raw_dataset
    eval_dataset = None
    if "eval" in raw_dataset:
        eval_dataset = raw_dataset["eval"]
        train_dataset = raw_dataset["train"]

    # Pre-tokenize to Arrow cache on disk (avoids OOM during SFTTrainer init)
    max_seq_len = cfg.data.get("max_seq_length", 2048)
    packing = cfg.data.get("packing", False)
    cache_dir = os.path.join(cfg.run.output_dir, "tokenize_cache")
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("Pre-tokenizing train dataset (max_seq_length=%d, packing=%s)...", max_seq_len, packing)
    from datasets import Dataset
    def _tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
    tokenized_train = train_dataset.map(
        _tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
        cache_file_name=os.path.join(cache_dir, "train.arrow"),
    )
    logger.info("Train tokenized: %d examples", len(tokenized_train))

    tokenized_eval = None
    if eval_dataset is not None:
        logger.info("Pre-tokenizing eval dataset...")
        tokenized_eval = eval_dataset.map(
            _tokenize_fn,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval",
            cache_file_name=os.path.join(cache_dir, "eval.arrow"),
        )
        logger.info("Eval tokenized: %d examples", len(tokenized_eval))

    # Free raw text data from memory
    del raw_dataset

    # ---- Phase 2: Load model + LoRA (after tokenization to stay under 500GB) ----
    logger.info("Loading model + LoRA...")
    model, _ = setup_model(cfg)

    from trl import SFTConfig, SFTTrainer

    # Determine dataloader workers — fewer for eval to save memory
    dl_workers = min(cfg.training.get("dataloader_num_workers", 4), 2)

    training_kwargs = {
        "output_dir": str(cfg.training.output_dir),
        "num_train_epochs": cfg.training.num_train_epochs,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "lr_scheduler_type": cfg.training.lr_scheduler_type,
        "warmup_ratio": cfg.training.get("warmup_ratio", 0.05),
        "weight_decay": cfg.training.get("weight_decay", 0.01),
        "max_grad_norm": cfg.training.get("max_grad_norm", 1.0),
        "bf16": cfg.training.get("bf16", True),
        "fp16": cfg.training.get("fp16", False),
        "logging_steps": cfg.training.get("logging_steps", 1),
        "save_steps": cfg.training.get("save_steps", 500),
        "save_total_limit": cfg.training.get("save_total_limit", 3),
        "seed": cfg.training.get("seed", 42),
        "dataloader_num_workers": dl_workers,
        "gradient_checkpointing": cfg.training.get("gradient_checkpointing", False),
        "optim": cfg.training.get("optim", "adamw_torch"),
        "report_to": "none" if cfg.wandb.get("mode") == "disabled" else "wandb",
        "packing": packing,
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": "NemotronHBlock",
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "true",
            "use_orig_params": "true",
        },
    }

    max_steps = cfg.training.get("max_steps", -1)
    if max_steps and max_steps > 0:
        training_kwargs["max_steps"] = max_steps
        training_kwargs.pop("num_train_epochs", None)

    eval_strategy = cfg.training.get("eval_strategy", "no")
    if eval_strategy and eval_strategy != "no":
        training_kwargs["eval_strategy"] = eval_strategy
        training_kwargs["eval_steps"] = cfg.training.get("eval_steps", 500)

    sft_config = SFTConfig(**training_kwargs)

    from nemotron_finetune.callbacks import JsonlMetricsCallback
    callbacks = [JsonlMetricsCallback(cfg.run.output_dir)]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    from nemotron_finetune.env import collect_environment
    from nemotron_finetune.logging_utils import write_json
    if local_rank == 0:
        env_info = collect_environment()
        env_path = os.path.join(cfg.run.output_dir, "logs", "environment.json")
        write_json(env_path, env_info)
        logger.info("Environment info saved to %s", env_path)

    logger.info("Starting FSDP training...")
    start_time = time.time()

    # Resume from latest checkpoint if available
    resume_checkpoint = None
    output_dir = str(cfg.run.output_dir)
    if os.path.isdir(output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            resume_checkpoint = os.path.join(output_dir, checkpoints[-1])
            logger.info("Resuming from checkpoint: %s", resume_checkpoint)

    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed = time.time() - start_time

    if local_rank == 0:
        logger.info("Training completed in %.1f seconds", elapsed)
        logger.info("Train metrics: %s", json.dumps(train_result.metrics, indent=2))

        trainer.save_model(cfg.run.output_dir)
        logger.info("Model saved to %s", cfg.run.output_dir)

        summary = {
            "train_metrics": train_result.metrics,
            "elapsed_seconds": round(elapsed, 2),
            "config_path": str(cfg.run.config_path),
        }
        summary_path = os.path.join(cfg.run.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Training summary saved to %s", summary_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
