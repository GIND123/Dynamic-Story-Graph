"""Optional frozen Hugging Face generator wrapper.

This file intentionally owns all Transformers/PEFT imports so the symbolic and
state-plane tests never require model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gnsm.exceptions import OptionalDependencyError
from gnsm.generation.base import GenerationRequest
from gnsm.generation.conditioning import build_conditioning_packet


@dataclass(slots=True)
class HuggingFaceFrozenGenerator:
    model_name: str
    max_new_tokens: int = 1024
    _tokenizer: Any = None
    _model: Any = None

    def load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "Hugging Face generation requires `python -m pip install -e '.[training]'`."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            load_in_4bit=True,
        )
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def generate(self, request: GenerationRequest) -> str:
        if self._model is None or self._tokenizer is None:
            self.load()
        packet = build_conditioning_packet(request.state, request.action.participants)
        prompt = self._prompt(request, packet.symbolic_constraints)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
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
