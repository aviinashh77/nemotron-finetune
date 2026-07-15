# Setup Guide

## Environment

- **OS**: Ubuntu 24.04.3 LTS (Noble Numbat)
- **Python**: 3.12.3
- **CUDA**: 13.0
- **GPU**: 2x NVIDIA A100-SXM4-80GB (81920 MiB each)
- **Container RAM limit**: 500 GB (cgroup)

## 1. Clone and install

```bash
git clone <repo-url> nemotron-finetune
cd nemotron-finetune

# Install core dependencies
pip install -r requirements.txt

# Optional: install Mamba kernels for faster training
# Note: causal_conv1d may fail to build on CUDA 13.0 — training still works without it
pip install mamba-ssm --no-build-isolation --no-deps
```

### Dependency notes

| Package | Install issue | Impact |
|---------|--------------|--------|
| `mamba-ssm` | Requires `--no-build-isolation --no-deps` | Falls back to naive Mamba (3x more activation memory) |
| `causal_conv1d` | Fails to build on CUDA 13.0 | Same as above — naive Mamba path used |
| `bitsandbytes` | Works out of the box | Required for 4-bit quantization |
| `wandb` | Not installed by default | Uncomment in requirements.txt if needed |

## 2. Model preparation

The model must be downloaded or linked locally. The codebase expects the model directory to contain:
- `config.json` with `auto_map` pointing to local modeling files
- `modeling_nemotron_h.py` — Custom model code
- `configuration_nemotron_h.py` — Custom config class
- Model weight files (`.safetensors` or `.bin`)

### Download the model

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download NVIDIA/Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir /path/to/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

### Update config paths

Edit your run config to point to the correct model path:

```yaml
model:
  path: /path/to/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

## 3. Attention implementation

NemotronH does **not** support Flash Attention 2 or SDPA. Use `eager`:

```yaml
model:
  attn_implementation: eager  # Required for NemotronH
```

## 4. Mamba kernels

The model falls back to a naive (slow) Mamba implementation when `causal_conv1d` and `mamba_ssm` selective ops are unavailable. This uses ~3x more activation memory.

To enable fast Mamba kernels:
```bash
pip install mamba-ssm --no-build-isolation --no-deps
pip install causal-conv1d  # May fail on CUDA 13.0
```

If kernels are not available, set in config:
```yaml
model:
  use_mamba_kernels: false
```

## 5. Training data

Place your training data in `data/` as JSON files. See [DATA_FORMATS.md](DATA_FORMATS.md) for format specs.

### Verilog data preparation

Source data: `verilog_db_v0.1.jsonl` — 38,417 Verilog modules.

```python
import json

# Load source data
with open("/workspace/verilog_db_v0.1.jsonl") as f:
    data = [json.loads(line) for line in f]

# Split into train/val (98/2)
train = [{"text": d["code"]} for d in data[:37417]]
val = [{"text": d["code"]} for d in data[37417:]]

# Save
with open("data/verilog/train_full.json", "w") as f:
    json.dump(train, f)

with open("data/verilog/val_full.json", "w") as f:
    json.dump(val, f)

# Filter long samples (>30k chars ≈ >10k tokens)
MAX_CHARS = 30000
train_filtered = [d for d in train if len(d["text"]) <= MAX_CHARS]
# Result: 36,321 train, 971 val
```

### Quick test with included dummy data
- `data/dummy/sample_chat_5.json` — 5 chat examples for SFT
- `data/dummy/sample_cpt_100.json` — 118 text examples for CPT
- `data/dummy/sample_cpt_long.json` — 200 long text examples (~4k tokens each)

## 6. Running training

### DDP (single GPU)

```bash
python train.py --config configs/cpt_dummy.yaml
```

### DDP (multi-GPU)

```bash
accelerate launch --num_processes 2 --multi_gpu \
    train.py --config configs/cpt_dummy.yaml
```

### FSDP (multi-GPU, long sequences)

```bash
torchrun --nproc_per_node=2 \
    train_fsdp.py --config configs/cpt_fsdp.yaml
```

### FSDP with custom settings

```bash
torchrun --nproc_per_node=2 \
    train_fsdp.py --config configs/cpt_fsdp.yaml \
    data.max_seq_length=4096 \
    training.per_device_train_batch_size=1 \
    training.gradient_accumulation_steps=8
```

### Verilog CPT (current active run)

```bash
# Clean old cache
rm -rf results/output_fsdp_full/tokenize_cache results/output_fsdp_full/train.log

# Launch in tmux
tmux new-session -d -s train \
    "torchrun --nproc_per_node=2 --master_port=29503 \
    train_fsdp.py --config configs/cpt_verilog_fsdp.yaml 2>&1 | \
    tee results/output_fsdp_full/train.log"
```

## 7. Outputs

All outputs are written to `<output_dir>/` (default: `outputs/<run_name>/`):

```
outputs/my_run/
├── adapter_model.safetensors    # LoRA weights
├── adapter_config.json          # LoRA config
├── resolved_config.yaml         # Full config snapshot
├── training_summary.json        # Final metrics
├── logs/
│   ├── metrics.jsonl            # Per-step metrics
│   ├── train.log                # Human-readable log
│   └── environment.json         # HW/SW snapshot
├── tokenize_cache/              # Pre-tokenized Arrow cache
│   ├── train.arrow
│   └── eval.arrow
└── checkpoint-N/                # Checkpoints
```

## 8. Resume from checkpoint

```bash
# DDP
python train.py --config configs/cpt_long.yaml \
    training.output_dir=outputs/cpt_long_100dp_5ep

# The trainer auto-resumes from the latest checkpoint in output_dir
```

## 9. Weights & Biases

Enable W&B logging:

```yaml
wandb:
  mode: online       # or 'offline' for buffered logging
  project: nemotron-finetune
  run_name: my_run
```

```bash
wandb login  # Enter your API key
```

**Note:** For FSDP, wandb env vars must be set before `SFTTrainer` init. Both `train.py` and `train_fsdp.py` handle this automatically.

## 10. Quantization

For memory-constrained setups, enable 4-bit NF4 quantization:

```yaml
quantization:
  enabled: true
  bits: 4
  quant_type: nf4
  compute_dtype: bfloat16
  double_quant: false

model:
  use_mamba_kernels: false  # Required with quantization
```

**Note**: Quantization reduces model weights from ~64 GB to ~16 GB, but Mamba Triton kernels are incompatible with quantized weights. The model falls back to the naive Mamba path.

## 11. Container RAM considerations

The training container has a 500GB RAM cgroup limit. The Nemotron-30B model is ~64GB in bf16. Combined with tokenization, data loading, and OS overhead, this can be tight.

**Rule of thumb:** Always tokenize data BEFORE loading the model.

```python
# SAFE order:
tokenizer = AutoTokenizer.from_pretrained(model_path)  # ~100MB
data = format_and_tokenize(tokenizer)                   # ~2GB
model = AutoModelForCausalLM.from_pretrained(model_path) # ~64GB
# Total: ~66GB — well within 500GB limit

# UNSAFE order:
model = AutoModelForCausalLM.from_pretrained(model_path) # ~64GB
data = format_and_tokenize(tokenizer)                   # ~2GB + overhead
# Can exceed 500GB when Arrow datasets + OS cache are counted
```
