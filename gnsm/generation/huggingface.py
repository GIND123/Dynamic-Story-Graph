"""Optional frozen Hugging Face generator wrapper.

This file intentionally owns all Transformers/PEFT imports so the symbolic and
state-plane tests never require model downloads. Loading adapts to the runtime:
4-bit quantization when bitsandbytes + CUDA are present, otherwise bf16/fp16 on
GPU, and float32 on CPU. That keeps `gnsm generate` runnable on a Colab T4 and on
a plain laptop without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gnsm.exceptions import OptionalDependencyError
from gnsm.generation.base import GenerationRequest
from gnsm.generation.conditioning import build_conditioning_packet

# Root searched for already-downloaded weights before falling back to the Hub.
MODELS_DIR_ENV = "GNSM_MODELS_DIR"


def local_or_hub(reference: str) -> str:
    """Local weights under `models/` when present, else the reference unchanged.

    The Colab download step writes each checkpoint to the lowercased repo
    basename, so `meta-llama/Llama-3.1-8B-Instruct` loads from
    `models/llama-3.1-8b-instruct` instead of re-downloading 15 GB. An explicit
    path is returned untouched, and an unknown repo id still resolves via the Hub.
    """
    if Path(reference).expanduser().is_dir():
        return reference
    root = os.environ.get(MODELS_DIR_ENV) or Path(__file__).resolve().parents[2] / "models"
    candidate = Path(root).expanduser() / reference.rsplit("/", 1)[-1].lower()
    return str(candidate) if candidate.is_dir() else reference


@dataclass(slots=True)
class HuggingFaceFrozenGenerator:
    model_name: str
    max_new_tokens: int = 1024
    quantize: bool = True
    dtype: str = "auto"
    _tokenizer: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _load_summary: dict[str, Any] = field(default_factory=dict, repr=False)

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "Hugging Face generation requires `pip install -e '.[training]'`."
            ) from exc

        has_cuda = bool(torch.cuda.is_available())
        kwargs: dict[str, Any] = {"device_map": "auto" if has_cuda else None}
        summary: dict[str, Any] = {"cuda": has_cuda, "quantized": False}

        quant_config = self._quantization_config(torch) if (self.quantize and has_cuda) else None
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            summary["quantized"] = True
        else:
            kwargs["torch_dtype"] = self._resolve_dtype(torch, has_cuda)
            summary["torch_dtype"] = str(kwargs["torch_dtype"])

        weights = local_or_hub(self.model_name)
        summary["weights"] = weights
        summary["local_weights"] = weights != self.model_name

        self._tokenizer = AutoTokenizer.from_pretrained(weights)
        self._model = AutoModelForCausalLM.from_pretrained(weights, **kwargs)
        if not has_cuda:
            self._model = self._model.to("cpu")
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        summary["device"] = str(next(self._model.parameters()).device)
        self._load_summary = summary

    def _quantization_config(self, torch: Any) -> Any:
        """4-bit NF4 config, or None if bitsandbytes is unavailable."""
        try:
            import bitsandbytes  # noqa: F401  # presence check only
            from transformers import BitsAndBytesConfig
        except ImportError:
            return None
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def _resolve_dtype(self, torch: Any, has_cuda: bool) -> Any:
        if self.dtype == "float32" or not has_cuda:
            return torch.float32
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def generate(self, request: GenerationRequest) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()
        import torch

        packet = build_conditioning_packet(request.state, request.action.participants)
        prompt = self._prompt(request, packet.symbolic_constraints)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        continuation = outputs[0, inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(continuation, skip_special_tokens=True))

    @staticmethod
    def _prompt(request: GenerationRequest, constraints: tuple[str, ...]) -> str:
        canon = "\n".join(f"- {constraint}" for constraint in constraints)
        return (
            "Write the next story scene. Preserve every canon constraint.\n"
            f"CANON:\n{canon}\n"
            f"ROLLING SUMMARY:\n{request.rolling_summary}\n"
            f"SCENE INTENT:\n{request.action.intent}\n"
            f"CORRECTION:\n{request.corrective_constraint}\n"
            "SCENE:\n"
        )
