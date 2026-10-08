"""The Modal generation loop, driven off-GPU with stand-ins for vLLM and Modal.

The loop only ever runs on a paid GPU, so a wiring mistake -- a prompt built
for the wrong template, a truncation flag attached to the wrong run -- would
otherwise surface as a wasted run. These stand-ins answer every call
deterministically so the real ``_run_generation`` can be exercised on a CPU.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from unittest.mock import MagicMock

import pytest

from dsg.generate.canon import build_premises, rotate_beats
from dsg.generate.conditions import PROSE_LADDER, chapter_budget, template_of


def _is_extraction(prompt: str) -> bool:
    return "\nFACTS:" in prompt


class _Tokenizer:
    @classmethod
    def from_pretrained(cls, _model_id):
        return cls()

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        turns = "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)
        return turns + "<|assistant|>"

    def __call__(self, text):
        return types.SimpleNamespace(input_ids=text.split())


class _SamplingParams:
    def __init__(self, **kw):
        self.n = kw.get("n", 1)


class _LLM:
    """Answers chapters, extractions and repairs; remembers every prompt."""

    seen: list[str] = []

    def __init__(self, **_kw):
        type(self).seen = []

    def generate(self, prompts, params, use_tqdm=False, lora_request=None):
        out = []
        for prompt in prompts:
            type(self).seen.append(prompt)
            if _is_extraction(prompt):
                # Chapter 1 sets grey eyes; every later chapter contradicts it,
                # so the guarded arm is always sent back for repair.
                colour = "grey" if "(draft for chapter 1)" in prompt else "blue"
                texts = [f"FACT | Hesper | eye_colour | {colour}"]
                finish = "stop"
            else:
                n = int(re.search(r"chapter (\d+)", prompt, re.IGNORECASE).group(1))
                text = (
                    f"Hesper waited by the door. (draft for chapter {n})\n\n"
                    f'"Not yet," said Jarrow.\n\nThe fog lifted at noon, chapter {n}.'
                )
                prose = "<|system|>" in prompt
                # Prose chapter 3 hits the cap, and so does every repair. The
                # guard fires at chapter 2 only (the state then takes "blue"),
                # so a cut there can only come from the repaired chapter.
                repair = "contradicted the established canon" in prompt
                finish = "length" if prose and (n == 3 or repair) else "stop"
                texts = [text]
                if params.n > 1:
                    # Best-of-k: tell the candidates apart, and cut the odd ones,
                    # so the record shows whose flag it kept.
                    texts = [f"{text} (candidate {j})" for j in range(params.n)]
                    finish = ["length" if j % 2 else "stop" for j in range(params.n)]
            finishes = finish if isinstance(finish, list) else [finish] * len(texts)
            out.append(types.SimpleNamespace(outputs=[
                types.SimpleNamespace(text=t, finish_reason=f)
                for t, f in zip(texts, finishes, strict=True)
            ]))
        return out


@pytest.fixture
def loop(monkeypatch, tmp_path):
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = _LLM, _SamplingParams
    lora = types.ModuleType("vllm.lora")
    request = types.ModuleType("vllm.lora.request")
    request.LoRARequest = MagicMock()
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _Tokenizer
    for name, module in {
        "modal": MagicMock(), "vllm": vllm, "vllm.lora": lora,
        "vllm.lora.request": request, "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "dsg.infra.modal_generate", raising=False)
    module = importlib.import_module("dsg.infra.modal_generate")
    monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
    yield module
    sys.modules.pop("dsg.infra.modal_generate", None)


CONDITIONS = PROSE_LADDER + ("base:dsg-hybrid-repair~prose",)
EXAMPLE = " ".join(["word"] * 200)


def _run(loop, conditions=CONDITIONS, example=EXAMPLE, chapters=3):
    premises = rotate_beats(build_premises(n=2), 2)
    payload = loop._run_generation(
        [p.to_json() for p in premises], "stub-model", conditions, chapters=chapters,
        words=500, max_model_len=4096, run_id="t", example=example, beat_rotation=2,
    )
    return premises, payload


def test_meta_records_templates_example_and_rotation(loop, tmp_path):
    _, payload = _run(loop)
    assert any(_is_extraction(p) for p in _LLM.seen), "the stub must see extractions"
    meta = payload["meta"]
    assert meta["templates"] == ["classic", "prose"]
    assert len(meta["example_sha256"]) == 64 and meta["beat_rotation"] == 2
    assert (tmp_path / "t-checkpoint.json").exists()
    assert len(payload["records"]) == 2 * len(CONDITIONS) * 3


def test_each_template_reaches_the_model_in_its_own_shape(loop):
    _run(loop)
    chapters = [p for p in _LLM.seen if not _is_extraction(p)]
    classic = [p for p in chapters if "<|system|>" not in p]
    prose = [p for p in chapters if "<|system|>" in p]
    assert classic and prose
    assert all(p.startswith("<|user|>You are writing chapter") for p in classic)
    assert all("<<<\n" + EXAMPLE + "\n>>>" in p for p in prose), "example in system turn"
    assert not any("<<<" in p for p in classic)


def test_records_carry_budget_truncation_and_continuity(loop):
    premises, payload = _run(loop)
    by_id = {p.story_id: p for p in premises}
    for rec in payload["records"]:
        prose = template_of(rec["condition"]) == "prose"
        assert rec["template"] == ("prose" if prose else "classic")
        run = types.SimpleNamespace(condition=rec["condition"])
        assert rec["word_budget"] == chapter_budget(
            by_id[rec["story_id"]], run, rec["chapter"], 500
        )
        repaired = rec["condition"].startswith("base:dsg-hybrid-repair") and rec["chapter"] == 2
        cut = prose and (rec["chapter"] == 3 or repaired)
        assert rec["truncated"] is cut, (rec["condition"], rec["chapter"])
        expected = "cue" if prose and rec["chapter"] > 1 else "none"
        assert rec["continuity"] == expected, rec["condition"]


def test_rotation_reaches_the_prompts(loop):
    premises, _ = _run(loop)
    assert premises[0].beats[0] != premises[1].beats[0]
    first = [p for p in _LLM.seen if "chapter 1 of" in p]
    assert any(premises[1].beats[0] in p for p in first)


def test_a_prose_repair_lands_in_the_user_turn(loop):
    _, payload = _run(loop)
    assert payload["meta"]["repairs_attempted"] > 0
    repairs = [p for p in _LLM.seen if "contradicted the established canon" in p]
    assert repairs and all("<|system|>" in p for p in repairs)
    for prompt in repairs:
        user = prompt.split("<|user|>", 1)[1]
        assert user.index("Begin chapter") < user.index("contradicted the established")
        assert user.index("contradicted the established") < user.index("<|assistant|>")


def test_the_tuned_writer_is_never_served_the_prose_template(loop):
    with pytest.raises(ValueError, match="trained on the classic"):
        _run(loop, conditions=("tuned:dsg-state~prose",))


def test_best_of_k_keeps_the_chosen_candidates_truncation_flag(loop):
    """The flag must follow the pick, not candidate 0 or any other draw."""
    _, payload = _run(loop, conditions=("base:last-chapter@bo4-random",), example="")
    picks = set()
    for rec in payload["records"]:
        pick = int(re.search(r"\(candidate (\d)\)", rec["text"]).group(1))
        picks.add(pick)
        assert rec["truncated"] is bool(pick % 2), (rec["chapter"], pick)
    assert {p % 2 for p in picks} == {0, 1}, "the run must pick both kinds to prove it"
