#!/usr/bin/env python
"""Config-driven LoRA finetuning entrypoint for Nemotron models.

Supports SFT, CPT, and DAPT modes via YAML configuration. The config is
assembled by deep-merging: base.yaml -> user YAML -> CLI dotlist overrides.

Usage:
    # Basic run
    python train.py --config configs/sft_sample.yaml

    # Override config fields from CLI
    python train.py --config configs/sft_sample.yaml \\
        training.learning_rate=1e-4 \\
        training.num_train_epochs=5

    # Use a named config from configs/
    python train.py --config sft_sample.yaml

See README.md for full usage and docs/CONFIGURATION.md for config reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omegaconf import DictConfig, OmegaConf


def parse_args():
    p = argparse.ArgumentParser(description="Nemotron LoRA finetuning")
    p.add_argument("--config", required=True, help="Path to run YAML config")
    p.add_argument("overrides", nargs="*", help="Dotlist overrides, e.g. training.learning_rate=1e-4")
    return p.parse_args()


def load_data(cfg: DictConfig):
    """Load training (and optional eval) data as HuggingFace Dataset."""
    from datasets import load_dataset

    train_path = cfg.data.train_path
    if train_path is None:
        raise ValueError("data.train_path is required")

    data_files = {"train": str(train_path)}
    if cfg.data.get("eval_path"):
        data_files["eval"] = str(cfg.data.eval_path)

    raw = load_dataset("json", data_files=data_files, split="train")
    return raw


def format_chat(example, tokenizer):
    """Convert chat-format data to model input using the chat template."""
    messages = example.get("messages", [])
    if not messages:
        # Fallback: try prompt/completion keys
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def format_prompt_completion(example, tokenizer):
    """Format prompt_completion data."""
    prompt = example.get("prompt", "")
    completion = example.get("completion", "")
    text = f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"
    return {"text": text}


def setup_model(cfg: DictConfig):
    """Load model with optional quantization and apply LoRA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = cfg.model.path
    print(f"Loading model from: {model_path}")

    # Quantization config
    quant_config = None
    if cfg.quantization.get("enabled", False):
        compute_dtype = torch.bfloat16 if cfg.quantization.compute_dtype == "bfloat16" else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.bits == 4,
            bnb_4bit_quant_type=cfg.quantization.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.quantization.get("double_quant", False),
        )

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, cfg.model.get("torch_dtype", "bfloat16")),
    }
    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    if cfg.model.get("attn_implementation"):
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Disable cache for training
    if not cfg.model.get("use_cache", True):
        model.config.use_cache = False

    # Override use_mamba_kernels if specified (needed for 4-bit quantization compat)
    if cfg.model.get("use_mamba_kernels") is not None:
        model.config.use_mamba_kernels = cfg.model.use_mamba_kernels
        print(f"use_mamba_kernels set to {cfg.model.use_mamba_kernels}")

    # Patch NemotronHOutput to include past_key_values for TRL compatibility
    # TRL 1.8.0's _chunked_ce_forward accesses outputs.past_key_values, but
    # NemotronHOutput uses cache_params instead.
    try:
        import types as _types
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
                    print(f"Patched {_cls_name} for TRL past_key_values compatibility")
    except Exception as e:
        print(f"Warning: Could not patch NemotronH output classes: {e}")

    # Apply LoRA
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
            # Limit to attention + Mamba projections only (~92 layers) — skip MoE
            # experts (~5934 layers) which bloat PEFT init time.
            priority = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"]
            targets = [n for n in priority if n in linear_names]
            # Include up/down_proj only if explicitly requested or few targets found
            remaining = [n for n in linear_names if n not in targets]
            if len(targets) < 3:
                targets += remaining[:3]
            lora_config.target_modules = targets
            print(f"LoRA targeting: {targets}")
        else:
            target = list(cfg.lora.target_modules)
            lora_config.target_modules = target

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    args = parse_args()

    # Load config
    from nemotron_finetune.config import load_config
    cfg = load_config(args.config, args.overrides if args.overrides else None)

    # Setup logging
    from nemotron_finetune.logging_utils import setup_logging
    logger = setup_logging(cfg.run.output_dir, cfg.logging.get("level", "INFO"))
    logger.info("Config loaded: %s", cfg.run.config_path)

    # Save resolved config for reproducibility
    from nemotron_finetune.config import save_resolved
    save_resolved(cfg, os.path.join(cfg.run.output_dir, "resolved_config.yaml"))
    logger.info("Resolved config saved")

    # Load data
    logger.info("Loading data...")
    raw_dataset = load_data(cfg)
    logger.info("Dataset loaded: %d examples", len(raw_dataset))

    # Setup model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = setup_model(cfg)

    # Format data based on mode
    data_format = cfg.data.get("format", "chat")
    if data_format == "chat":
        raw_dataset = raw_dataset.map(
            lambda ex: format_chat(ex, tokenizer),
            num_proc=cfg.data.get("num_proc", 4),
            desc="Formatting chat data",
        )
    elif data_format == "prompt_completion":
        raw_dataset = raw_dataset.map(
            lambda ex: format_prompt_completion(ex, tokenizer),
            num_proc=cfg.data.get("num_proc", 4),
            desc="Formatting prompt/completion data",
        )
    # For text format, data is already in the right shape

    # Split train/eval if eval data provided
    train_dataset = raw_dataset
    eval_dataset = None
    if "eval" in raw_dataset:
        eval_dataset = raw_dataset["eval"]
        train_dataset = raw_dataset["train"]

    # Build SFTConfig and SFTTrainer
    from trl import SFTConfig, SFTTrainer

    training_kwargs = {
        "output_dir": str(cfg.training.output_dir),
        "num_train_epochs": cfg.training.num_train_epochs,
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.get("per_device_eval_batch_size", 2),
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
        "dataloader_num_workers": cfg.training.get("dataloader_num_workers", 4),
        "gradient_checkpointing": cfg.training.get("gradient_checkpointing", True),
        "optim": cfg.training.get("optim", "paged_adamw_8bit"),
        "report_to": "none" if cfg.wandb.get("mode") == "disabled" else "wandb",
        "max_length": cfg.data.get("max_seq_length", 2048),
        "packing": cfg.data.get("packing", False),
        "dataset_text_field": "text",
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

    # Callbacks
    from nemotron_finetune.callbacks import JsonlMetricsCallback
    callbacks = [JsonlMetricsCallback(cfg.run.output_dir)]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Log environment info
    from nemotron_finetune.env import collect_environment
    from nemotron_finetune.logging_utils import write_json
    env_info = collect_environment()
    env_path = os.path.join(cfg.run.output_dir, "logs", "environment.json")
    write_json(env_path, env_info)
    logger.info("Environment info saved to %s", env_path)

    # Train
    logger.info("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    logger.info("Training completed in %.1f seconds", elapsed)
    logger.info("Train metrics: %s", json.dumps(train_result.metrics, indent=2))

    # Save model
    trainer.save_model(cfg.run.output_dir)
    logger.info("Model saved to %s", cfg.run.output_dir)

    # Save training summary
    summary = {
        "train_metrics": train_result.metrics,
        "elapsed_seconds": round(elapsed, 2),
        "config_path": str(cfg.run.config_path),
    }
    summary_path = os.path.join(cfg.run.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Training summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
