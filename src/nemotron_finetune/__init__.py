"""Config-driven LoRA finetuning for NVIDIA Nemotron models.

Supports LoRA CPT (continued pretraining), LoRA DAPT (domain-adaptive
pretraining) and LoRA SFT (supervised finetuning) via a single config-driven
entrypoint. Features:

- Three-layer config merging: base defaults -> user YAML -> CLI overrides
- PEFT/LoRA for parameter-efficient finetuning
- TRL SFTTrainer as the training backbone
- BitsAndBytes 4-bit NF4 quantization (optional)
- Local JSONL metrics logging with perplexity, throughput, and GPU memory
- Optional Weights & Biases integration
- Full environment reproducibility via saved configs and environment snapshots

See README.md for usage and docs/CONFIGURATION.md for config reference.
"""

__version__ = "0.1.0"
