"""The extraction pass: what the language model is allowed to do.

The model never writes to the store. It reads one window plus a bounded,
**policy-agnostic** digest of proper names seen so far, and returns typed
*proposals*. Because the digest depends on nothing but the cached proposal
stream itself, every policy in the study consumes a byte-identical extraction,
so any measured difference between policies is a difference of representation,
not of extractor quality.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dsg.lexicon import is_person_description, is_proper_name
from dsg.matching import classify_surface
from dsg.schemas import Span, Window
from dsg.windows import iter_windows

MAX_ENTITIES = 10
MAX_FACTS = 8
MAX_SPEECH = 6

PROMPT = """You annotate a novel one passage at a time, as a first-time reader who has not read ahead.

Characters already met:
{digest}

Read the passage below and output ONE LINE PER ITEM, using these forms exactly:

CHAR | <exact string from the passage> | name
CHAR | <exact string from the passage> | description
LINK | <string in this passage> | <a character already met>
FACT | <who> | <predicate> | <what> | narrated | <<= 10 words quoted from the passage>
SAID | <first few words of a quoted line> | <who said it>

Predicates allowed: location, resides_in, possesses, alive, married_to, engaged_to,
stance_toward, emotion, occupation, travelling_to, knows, member_of, parent_of,
child_of, sibling_of, relative_of, identity_of, true_name, gender

Rules:
- Use ONLY what this passage says. You have not read the rest of the book.
- CHAR is for people and places only. NEVER put dialogue, a sentence, or a
  pronoun (he, she, they) in a CHAR line.
- Use "description" for unnamed references such as "the stranger", "an old man".
- Write LINK when this passage reveals that one reference is another, e.g.
  if it becomes clear that "the young American" is Winterbourne, write
  LINK | the young American | Winterbourne
- Write "reported" instead of "narrated" when a character says it rather than the narration.
- Never repeat the same line twice. At most {max_entities} CHAR, {max_facts} FACT, {max_speech} SAID lines.
- Output nothing except these lines. Stop when done.

Previous passage ended:
{lead_in}

<<<WINDOW>>>
{window}
<<<END>>>

LINES:"""


@dataclass(slots=True)
class FactProposal:
    subject: str
    predicate: str
    object: str
    certainty: str = "narrated"
    evidence: str = ""


@dataclass(slots=True)
class SpeechProposal:
    quote: str
    speaker: str


@dataclass(slots=True)
class WindowProposal:
    index: int
    start: int
    end: int
    entities: list[dict[str, str]] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)
    facts: list[FactProposal] = field(default_factory=list)
    speech: list[SpeechProposal] = field(default_factory=list)
    parse_ok: bool = True
    raw: str = ""

    @property
    def span(self) -> Span:
        return Span(self.start, self.end)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("raw", None)
        return d

    @classmethod
    def from_json(cls, d: dict) -> WindowProposal:
        return cls(
            index=d["index"], start=d["start"], end=d["end"],
            entities=d.get("entities", []), links=d.get("links", []),
            facts=[FactProposal(**f) for f in d.get("facts", [])],
            speech=[SpeechProposal(**s) for s in d.get("speech", [])],
            parse_ok=d.get("parse_ok", True),
        )


class SurfaceLedger:
    """Policy-free running tally used only to build the prompt digest.

    Unnamed figures are tracked alongside names: a reader can only be told that
    'the young American' is Winterbourne if the earlier description is still
    available to refer to, and that is the evidence a deferred identity
    resolution depends on.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.descriptions: dict[str, int] = {}

    def update(self, proposal: WindowProposal) -> None:
        for e in proposal.entities:
            s = (e.get("surface") or "").strip()
            if not s:
                continue
            if is_proper_name(s):
                self.counts[s] = self.counts.get(s, 0) + 1
            elif e.get("kind") == "description":
                self.descriptions[s] = self.descriptions.get(s, 0) + 1

    def digest(self, k: int = 20, k_desc: int = 6) -> str:
        if not self.counts and not self.descriptions:
            return "(none yet -- this is the start of the book)"
        lines = [
            f"- {name}"
            for name, _ in sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        ]
        unnamed = sorted(self.descriptions.items(), key=lambda kv: (-kv[1], kv[0]))[:k_desc]
        if unnamed:
            lines.append("Unnamed figures met so far (link one to a name if this passage reveals it):")
            lines += [f"- {name}" for name, _ in unnamed]
        return "\n".join(lines)



_PRONOUNS = frozenset({
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "i", "me", "my", "we", "us", "our", "you", "your", "who", "whom",
    "himself", "herself", "themselves", "myself", "this", "that", "these", "those",
})
_SENTENCE_CHARS = frozenset('?!;:"\u201c\u201d\u2018\u2019')


def is_plausible_entity(surface: str) -> bool:
    """Reject dialogue fragments, bare pronouns and whole sentences.

    Small models put quoted speech in entity slots. Filtering here rather than
    inside a policy keeps the proposal stream identical for every policy, so the
    comparison stays controlled.
    """
    s = (surface or "").strip()
    if not s or len(s) > 40:
        return False
    if any(c in _SENTENCE_CHARS for c in s):
        return False
    tokens = s.split()
    if not tokens or len(tokens) > 5:
        return False
    if s.lower().strip(".,") in _PRONOUNS:
        return False
    # A trailing period after a word is a sentence end, not an abbreviation.
    if s.endswith(".") and len(tokens[-1]) > 3:
        return False
    return any(c.isalpha() for c in s)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|```\s*$", re.MULTILINE)


def _coerce_str(v: object) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return " ".join(_coerce_str(x) for x in v).strip()
    if v is None:
        return ""
    return str(v).strip()


def _clean(cell: str) -> str:
    return _coerce_str(cell).strip(" \t\"'`*-").strip()


def parse_proposal(raw: str, window: Window) -> WindowProposal:
    """Parse the line format, falling back to JSON if a model emits that instead.

    Line-oriented output is used because small instruction-tuned models emit it
    far more reliably than nested JSON: a truncated or repeated line costs one
    item, whereas one unclosed brace costs the entire window.
    """
    out = WindowProposal(index=window.index, start=window.start, end=window.end, raw=raw)
    text = _FENCE_RE.sub("", raw or "").strip()
    seen: set[str] = set()
    found_any = False

    for line in text.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        cells = [_clean(c) for c in line.split("|")]
        tag = cells[0].upper().lstrip("- ").strip()

        if tag == "CHAR" and len(cells) >= 2 and cells[1]:
            surface = cells[1]
            if not is_plausible_entity(surface):
                continue
            if any(e["surface"].lower() == surface.lower() for e in out.entities):
                continue
            kind = (cells[2].lower() if len(cells) > 2 and cells[2] else "")
            if kind not in ("name", "description"):
                kind = classify_surface(surface)
            if is_proper_name(surface):
                # A model that labels 'Mrs. Miller' a description would put
                # her into the identity-resolution question and get her
                # merged with Miss Miller. Names are decided here, not there.
                kind = "name"
            if len(out.entities) < MAX_ENTITIES:
                out.entities.append({"surface": surface, "kind": kind})
            found_any = True
        elif tag == "LINK" and len(cells) >= 3 and cells[1] and cells[2]:
            if not (is_plausible_entity(cells[1]) and is_plausible_entity(cells[2])):
                continue
            if cells[1].lower() != cells[2].lower() and len(out.links) < MAX_ENTITIES:
                out.links.append({"surface": cells[1], "same_as": cells[2]})
            found_any = True
        elif tag == "FACT" and len(cells) >= 4 and cells[1] and cells[2] and cells[3]:
            if not is_plausible_entity(cells[1]):
                continue
            certainty = cells[4].lower() if len(cells) > 4 else "narrated"
            if certainty not in ("narrated", "reported", "implied"):
                certainty = "narrated"
            evidence = cells[5][:200] if len(cells) > 5 else ""
            if len(out.facts) < MAX_FACTS:
                out.facts.append(
                    FactProposal(
                        subject=cells[1], predicate=cells[2], object=cells[3],
                        certainty=certainty, evidence=evidence,
                    )
                )
            found_any = True
        elif tag == "SAID" and len(cells) >= 3 and cells[1] and cells[2]:
            if not is_plausible_entity(cells[2]):
                continue
            if len(out.speech) < MAX_SPEECH:
                out.speech.append(SpeechProposal(quote=cells[1], speaker=cells[2]))
            found_any = True

    if found_any:
        return out
    return _parse_json_fallback(text, out)


def _parse_json_fallback(text: str, out: WindowProposal) -> WindowProposal:
    m = _JSON_RE.search(text)
    if not m:
        out.parse_ok = False
        return out
    blob = _TRAILING_COMMA_RE.sub(r"\1", m.group(0))
    data = None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        depth, cut = 0, None
        for i, ch in enumerate(blob):
            depth += (ch == "{") - (ch == "}")
            if depth == 0 and ch == "}":
                cut = i + 1
        if cut:
            try:
                data = json.loads(blob[:cut])
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        out.parse_ok = False
        return out

    for e in (data.get("entities") or [])[:MAX_ENTITIES]:
        if isinstance(e, str) and is_plausible_entity(e):
            out.entities.append({"surface": e.strip(), "kind": classify_surface(e)})
        elif isinstance(e, dict):
            s = _coerce_str(e.get("surface") or e.get("name") or e.get("text"))
            if s and is_plausible_entity(s):
                out.entities.append(
                    {"surface": s, "kind": _coerce_str(e.get("kind")) or classify_surface(s)}
                )
    for item in (data.get("links") or [])[:MAX_ENTITIES]:
        if isinstance(item, dict):
            src = _coerce_str(item.get("surface"))
            dst = _coerce_str(item.get("same_as"))
            if src and dst and src.lower() != dst.lower():
                out.links.append({"surface": src, "same_as": dst})
    for f in (data.get("facts") or [])[:MAX_FACTS]:
        if isinstance(f, dict):
            subj, pred = _coerce_str(f.get("subject")), _coerce_str(f.get("predicate"))
            obj = _coerce_str(f.get("object"))
            if subj and pred and obj:
                out.facts.append(
                    FactProposal(
                        subject=subj, predicate=pred, object=obj,
                        certainty=(_coerce_str(f.get("certainty")) or "narrated").lower(),
                        evidence=_coerce_str(f.get("evidence"))[:200],
                    )
                )
    for s in (data.get("speech") or [])[:MAX_SPEECH]:
        if isinstance(s, dict):
            q, sp = _coerce_str(s.get("quote")), _coerce_str(s.get("speaker"))
            if q and sp:
                out.speech.append(SpeechProposal(quote=q, speaker=sp))
    if not (out.entities or out.facts or out.speech):
        out.parse_ok = False
    return out



# Identity resolution gets its own call. Asked as one of five things inside a
# larger schema, a 7B model simply never emits it; asked on its own, against an
# explicit list of open references, it answers reliably. Deferred commitment is
# only useful if the questions it leaves open eventually get answered, so this
# pass is what makes the mechanism live rather than decorative.
RESOLVE_PROMPT = """In this passage from a novel, some people are referred to without being named.

Unnamed references to resolve:
{unnamed}

People whose names are known so far:
{names}

Passage:
<<<WINDOW>>>
{window}
<<<END>>>

For each unnamed reference above, decide whether THIS passage makes clear which
named person it is. Output one line each, nothing else:

SAME | <unnamed reference> | <name from the list>
UNKNOWN | <unnamed reference>

Only write SAME when the passage itself makes the identity clear. If in doubt, write UNKNOWN.

LINES:"""


def build_resolve_prompt(window: Window, unnamed: list[str], names: list[str]) -> str:
    return RESOLVE_PROMPT.format(
        unnamed="\n".join(f"- {u}" for u in unnamed),
        names="\n".join(f"- {n}" for n in names),
        window=window.text,
    )


def needs_resolution(
    proposal: WindowProposal, ledger: SurfaceLedger, max_items: int = 4, max_names: int = 12
) -> tuple[list[str], list[str]]:
    """Which open references to ask about, and which names to offer.

    Policy-agnostic by construction: it reads only this window's proposal and
    the shared ledger, so the second pass can be cached and replayed alongside
    the first for every policy alike.
    """
    seen: set[str] = set()
    unnamed: list[str] = []
    for entity in proposal.entities:
        surface = (entity.get("surface") or "").strip()
        key = surface.lower()
        if entity.get("kind") != "description" or not surface or key in seen:
            continue
        if is_proper_name(surface) or not is_person_description(surface):
            continue
        seen.add(key)
        unnamed.append(surface)
    names = [n for n, _ in sorted(ledger.counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    for entity in proposal.entities:
        surface = (entity.get("surface") or "").strip()
        if entity.get("kind") == "name" and surface and surface not in names:
            names.append(surface)
    if not unnamed or not names:
        return [], []
    return unnamed[:max_items], names[:max_names]


def parse_resolve(raw: str, unnamed: list[str], names: list[str]) -> list[dict[str, str]]:
    """Accept only links whose two sides were actually offered in the prompt."""
    offered = {u.lower(): u for u in unnamed}
    known = {n.lower(): n for n in names}
    links: list[dict[str, str]] = []
    for line in _FENCE_RE.sub("", raw or "").splitlines():
        cells = [_clean(c) for c in line.split("|")]
        if len(cells) < 3 or cells[0].upper().lstrip("- ").strip() != "SAME":
            continue
        source, target = offered.get(cells[1].lower()), known.get(cells[2].lower())
        if source and target and source.lower() != target.lower():
            if not any(existing["surface"] == source for existing in links):
                links.append({"surface": source, "same_as": target})
    return links


def build_prompt(window: Window, lead_in: str, digest: str) -> str:
    return PROMPT.format(
        digest=digest,
        lead_in=(lead_in or "(start of book)")[-400:],
        window=window.text,
        max_entities=MAX_ENTITIES,
        max_facts=MAX_FACTS,
        max_speech=MAX_SPEECH,
    )


def extract_book(
    text: str,
    backend,
    window_chars: int = 4800,
    lead_chars: int = 500,
    max_windows: int | None = None,
    max_tokens: int = 700,
    progress=None,
) -> list[WindowProposal]:
    ledger = SurfaceLedger()
    out: list[WindowProposal] = []
    for window, lead in iter_windows(text, window_chars, lead_chars):
        if max_windows is not None and window.index >= max_windows:
            break
        prompt = build_prompt(window, lead, ledger.digest())
        raw = backend.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        proposal = parse_proposal(raw, window)
        unnamed, names = needs_resolution(proposal, ledger)
        if unnamed:
            resolve_raw = backend.generate(
                build_resolve_prompt(window, unnamed, names),
                max_tokens=120, temperature=0.0,
            )
            proposal.links.extend(parse_resolve(resolve_raw, unnamed, names))
        ledger.update(proposal)
        out.append(proposal)
        if progress is not None:
            progress(proposal)
    return out


def save_proposals(path: Path, proposals: Iterable[WindowProposal], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for p in proposals:
            fh.write(json.dumps(p.to_json()) + "\n")


def load_proposals(path: Path) -> tuple[dict, list[WindowProposal]]:
    lines = Path(path).read_text().splitlines()
    meta = json.loads(lines[0]).get("meta", {})
    return meta, [
        WindowProposal.from_json(json.loads(row)) for row in lines[1:] if row.strip()
    ]
