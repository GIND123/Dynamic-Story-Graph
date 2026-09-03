"""Deterministic backend used by the tests and by ``--backend mock``.

It runs the entire pipeline with no weights and no network, so CI exercises
windowing, parsing, the calculus, invariants and metrics end to end. It is a
plumbing check, never a source of reported numbers.
"""

from __future__ import annotations

import json
import re

_NAME_RE = re.compile(r"\b(?:Mr\.|Mrs\.|Miss|Dr\.|Sir|Lady)\s+[A-Z][a-z]+|\b[A-Z][a-z]{2,}\b")
_STOP = {
    "The", "And", "But", "She", "Her", "His", "They", "That", "This", "When",
    "Then", "There", "What", "With", "For", "Not", "You", "Was", "Had", "Chapter",
}


class MockBackend:
    name = "mock"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        body = prompt.split("<<<WINDOW>>>")[-1].split("<<<END>>>")[0]
        names, seen = [], set()
        for m in _NAME_RE.finditer(body):
            s = m.group(0).strip()
            if s in _STOP or s in seen:
                continue
            seen.add(s)
            names.append(s)
        names = names[:6]
        facts = []
        for i, n in enumerate(names[:2]):
            loc = re.search(r"\bin ([A-Z][a-z]+)", body)
            if loc:
                facts.append(
                    {"subject": n, "predicate": "location", "object": loc.group(1),
                     "certainty": "narrated", "evidence": loc.group(0)}
                )
        speech = []
        for m in list(re.finditer(r"[“\"]([^”\"]{8,120})[”\"]", body))[:3]:
            speech.append({"quote": m.group(1)[:40], "speaker": names[0] if names else ""})
        return json.dumps(
            {
                "entities": [{"surface": n, "kind": "name"} for n in names],
                "facts": facts,
                "speech": speech,
                "links": [],
            }
        )
