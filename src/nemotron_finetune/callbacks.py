"""Trainer callbacks for detailed local metric logging.

``JsonlMetricsCallback`` appends one JSON object per Trainer ``log`` event to
``<output_dir>/logs/metrics.jsonl``, enriched with derived metrics the base
Trainer does not emit: perplexity, GPU memory, wall-clock and throughput
(tokens/second). This gives a complete, machine-readable local record that sits
alongside (and does not depend on) Weights & Biases.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback

from .env import gpu_memory_stats
from .logging_utils import get_logger


def _safe_ppl(loss: Optional[float]) -> Optional[float]:
    if loss is None:
        return None
    try:
        if loss > 30:  # exp overflow guard
            return float("inf")
        return round(math.exp(loss), 4)
    except (OverflowError, ValueError):
        return None


class JsonlMetricsCallback(TrainerCallback):
    """Persist enriched training / evaluation metrics as JSON lines."""

    def __init__(self, output_dir: str | Path, tokens_per_step: Optional[int] = None):
        self.path = Path(output_dir) / "logs" / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tokens_per_step = tokens_per_step
        self.logger = get_logger()
        self._last_time: Optional[float] = None
        self._last_step: int = 0

    def _append(self, record: dict) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_time = time.time()
        self._last_step = int(state.global_step)
        self._append(
            {
                "event": "train_begin",
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "num_train_epochs": args.num_train_epochs,
                "tokens_per_step": self.tokens_per_step,
                **gpu_memory_stats(),
            }
        )
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = dict(logs or {})
        now = time.time()
        step = int(state.global_step)

        record = {
            "event": "log",
            "global_step": step,
            "epoch": round(float(state.epoch), 4) if state.epoch is not None else None,
        }
        record.update(logs)

        if "loss" in logs:
            record["perplexity"] = _safe_ppl(logs.get("loss"))

        # Throughput since the previous log event.
        if self._last_time is not None and step > self._last_step:
            dt = max(now - self._last_time, 1e-9)
            steps_done = step - self._last_step
            record["steps_per_sec"] = round(steps_done / dt, 4)
            if self.tokens_per_step:
                record["tokens_per_sec"] = round(steps_done * self.tokens_per_step / dt, 1)
        self._last_time = now
        self._last_step = step

        record.update(gpu_memory_stats())
        self._append(record)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metrics = dict(metrics or {})
        record = {"event": "evaluate", "global_step": state.global_step}
        record.update(metrics)
        if "eval_loss" in metrics:
            record["eval_perplexity"] = _safe_ppl(metrics.get("eval_loss"))
            self.logger.info(
                "eval @ step %s | eval_loss=%.4f | eval_ppl=%s",
                state.global_step,
                metrics["eval_loss"],
                record["eval_perplexity"],
            )
        record.update(gpu_memory_stats())
        self._append(record)
        return control

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        self.logger.info("checkpoint saved: %s", ckpt)
        self._append(
            {"event": "save", "global_step": state.global_step, "checkpoint": str(ckpt)}
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._append(
            {
                "event": "train_end",
                "global_step": state.global_step,
                **gpu_memory_stats(),
            }
        )
        return control
