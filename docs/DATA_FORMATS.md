# Data Formats

Nemotron-FineTune supports three data formats. The format must match the training mode.

## Format Summary

| Format | Mode | File Type | Required Keys |
|--------|------|-----------|---------------|
| `chat` | SFT | JSON/JSONL | `messages` |
| `prompt_completion` | SFT | JSON/JSONL | `prompt`, `completion` |
| `text` | CPT, DAPT | JSON/JSONL | `text` |

All formats are loaded as JSON arrays or JSONL (one JSON object per line).

## Chat Format

Used for **SFT** mode. Each example is a conversation with multiple turns.

### Structure

```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<user message>"},
    {"role": "assistant", "content": "<assistant response>"},
    {"role": "user", "content": "<follow-up question>"},
    {"role": "assistant", "content": "<follow-up response>"}
  ]
}
```

### Roles

| Role | Purpose |
|------|---------|
| `system` | Sets the model's behavior/persona |
| `user` | Human input |
| `assistant` | Model's expected response |

### Example

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a math tutor."},
      {"role": "user", "content": "What is the quadratic formula?"},
      {"role": "assistant", "content": "The quadratic formula is x = (-b +/- sqrt(b^2 - 4ac)) / 2a, used to solve ax^2 + bx + c = 0."}
    ]
  },
  {
    "messages": [
      {"role": "system", "content": "You are a math tutor."},
      {"role": "user", "content": "Can you solve x^2 - 5x + 6 = 0?"},
      {"role": "assistant", "content": "Using the quadratic formula: x = (5 +/- sqrt(25-24)) / 2 = (5 +/- 1) / 2. So x = 3 or x = 2."}
    ]
  }
]
```

### How it's processed

The tokenizer's `apply_chat_template()` method converts the messages array into the model's expected input format. This handles special tokens, role markers, and conversation structure automatically.

### Tips

- Include a `system` message to set consistent behavior across examples
- Ensure every user message has a corresponding assistant response
- Multi-turn conversations are supported and encouraged
- Keep examples within `data.max_seq_length` tokens

## Prompt/Completion Format

Used for **SFT** mode. A simpler format for single-turn instruction-following.

### Structure

```json
{
  "prompt": "<input instruction>",
  "completion": "<expected output>"
}
```

### Example

```json
[
  {
    "prompt": "Summarize the theory of relativity in one sentence.",
    "completion": "Einstein's theory of relativity describes how space and time are relative to the observer, with special relativity governing objects moving at constant speed and general relativity describing gravity as spacetime curvature."
  },
  {
    "prompt": "Write a Python function to check if a number is prime.",
    "completion": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"
  }
]
```

### How it's processed

The prompt and completion are concatenated with role markers:

```
<|user|>
{prompt}
<|assistant|>
{completion}
```

## Text Format

Used for **CPT** and **DAPT** modes. Raw text for continued pre-training.

### Structure

```json
{
  "text": "<raw text content>"
}
```

### Example

```json
[
  {
    "text": "NVIDIA Nemotron models are a family of large language models designed for various natural language processing tasks. The Nemotron-H architecture combines Mamba state-space models with transformer attention layers..."
  },
  {
    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Neural networks, a key ML technique, are inspired by biological brain structure..."
  }
]
```

### How it's processed

Text examples are tokenized directly without any role markers or conversation structure. They are packed into fixed-length sequences using the tokenizer.

## File Formats

### JSON Array

```json
[
  {"messages": [...]},
  {"messages": [...]},
  {"messages": [...]}
]
```

### JSONL (one JSON per line)

```jsonl
{"messages": [...]}
{"messages": [...]}
{"messages": [...]}
```

Both formats are supported. JSONL is recommended for large datasets as it allows streaming and partial loading.

## Data Preparation

### Validation checklist

1. All examples must have the correct top-level keys for your format
2. Strings must not be empty (at least one turn/message per example)
3. No null values in message content
4. File must be valid JSON or JSONL
5. Character encoding: UTF-8
6. **Filter long samples** — Samples >30k chars (~10k tokens) should be filtered to avoid slow tokenization and OOM

### Filtering long samples

Some datasets (especially code corpora) contain extremely long samples that cause:
- Slow tokenization (minutes per sample)
- Memory issues during training
- Wasted compute (truncated to `max_seq_length` anyway)

```python
import json

MAX_CHARS = 30000  # ~10k tokens

with open("data/train.json") as f:
    data = json.load(f)

filtered = [d for d in data if len(d.get("text", "")) <= MAX_CHARS]
print(f"Filtered: {len(data)} → {len(filtered)} ({100*len(filtered)/len(data):.1f}%)")

with open("data/train_filtered.json", "w") as f:
    json.dump(filtered, f)
```

### Example script to validate data

```python
import json

with open("data/train.json") as f:
    data = json.load(f)

for i, example in enumerate(data):
    assert "messages" in example, f"Example {i} missing 'messages' key"
    for j, msg in enumerate(example["messages"]):
        assert "role" in msg, f"Example {i}, message {j} missing 'role'"
        assert "content" in msg, f"Example {i}, message {j} missing 'content'"
        assert msg["role"] in ("system", "user", "assistant"), f"Invalid role: {msg['role']}"

print(f"Validated {len(data)} examples")
```

## Sample Data

A sample dataset with 5 chat examples is included at `data/dummy/sample_chat_5.json` for testing.

## Verilog HDL Data

The Verilog CPT project uses text-format data derived from `verilog_db_v0.1.jsonl`:

```python
# Source: verilog_db_v0.1.jsonl — 38,417 Verilog modules
# Each line: {"code": "module foo(...); ... endmodule", "metadata": {...}}

# Convert to training format:
data = [{"text": d["code"]} for d in source_data]

# Split: 37,417 train + 1,000 val
# Filter: remove samples >30k chars (~10k tokens)
# Final: 36,321 train + 971 val
```

**Data stats:**
- Median: 686 chars (~200 tokens)
- P90: 7,689 chars (~2,300 tokens)
- P95: 18,428 chars (~5,500 tokens)
- Max (filtered): 29,754 chars (~8,900 tokens)
- Unfiltered max: 55,601,430 chars (~16M tokens!) — must be filtered

**Filter threshold:** 30,000 chars (~10,000 tokens) keeps 97.1% of samples.
