"""Planted canon: gold that exists by construction.

The hard part of measuring generated-story consistency is that there is no gold
to measure against -- the story did not exist before the model wrote it. Human
judgement is expensive and an LLM judge is the thing we are trying not to
depend on.

So we plant the gold. Each story premise fixes a small set of **canon facts**
drawn from closed vocabularies whose contradictions are enumerable: eye colour,
metal, kinship, status. A violation is then a deterministic string test -- a
character established with green eyes described as blue-eyed, a silver locket
turned gold, a brother become a husband, a dead man speaking. These are exactly
the continuity errors long-form generation is accused of, and they are checkable
without a model in the loop.

The premises are synthetic, which is the price of controlled gold; the
limitation is stated rather than hidden.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# Closed vocabularies. Every value's contradiction set is every *other* value,
# which is what makes a violation decidable by string test.
VOCAB: dict[str, tuple[str, ...]] = {
    "eyes": ("green", "blue", "grey", "hazel", "amber"),
    "hair": ("red", "black", "blonde", "auburn", "silver-white"),
    "metal": ("silver", "gold", "iron", "brass", "pewter"),
    "kin": ("brother", "sister", "father", "mother", "cousin", "uncle", "aunt"),
    "trade": ("blacksmith", "apothecary", "cartographer", "clockmaker",
              "bookbinder", "glassblower"),
    "home": ("Ashgrove", "Ravensmoor", "Thornbury", "Kilbride", "Alderney"),
}

# Surface forms that count as asserting each value, beyond the value itself.
_ALIASES: dict[str, tuple[str, ...]] = {
    "blonde": ("blond", "fair-haired"),
    "silver-white": ("white-haired", "silver-haired"),
    "gold": ("golden",),
    "silver": ("silvery",),
}


def _forms(value: str) -> tuple[str, ...]:
    return (value,) + _ALIASES.get(value, ())


@dataclass(frozen=True, slots=True)
class CanonFact:
    """One planted, checkable fact."""

    fact_id: str
    subject: str            # character name, or object name for `metal`
    kind: str               # key into VOCAB
    value: str              # the true value
    statement: str          # how it is stated in the outline
    window: int = 220       # chars either side of the subject mention to search

    def contradictions(self) -> tuple[str, ...]:
        out: list[str] = []
        for other in VOCAB[self.kind]:
            if other == self.value:
                continue
            out.extend(_forms(other))
        return tuple(out)

    def _pattern(self, words: tuple[str, ...]) -> re.Pattern[str]:
        alt = "|".join(re.escape(w) for w in words)
        return re.compile(rf"(?<!\w)({alt})(?!\w)", re.IGNORECASE)

    def check(self, text: str) -> tuple[bool, str]:
        """Return (violated, evidence).

        A violation requires a contradicting value *near a mention of the
        subject*: proximity is what makes the test about this character rather
        than about the chapter's vocabulary at large.
        """
        subject_re = re.compile(rf"(?<!\w){re.escape(self.subject)}(?!\w)", re.IGNORECASE)
        bad = self._pattern(self.contradictions())
        good = self._pattern(_forms(self.value))
        for m in subject_re.finditer(text):
            lo = max(0, m.start() - self.window)
            hi = min(len(text), m.end() + self.window)
            span = text[lo:hi]
            hit = bad.search(span)
            if hit is None:
                continue
            # A nearby correct restatement dominates: "not gold but silver".
            near = span[max(0, hit.start() - 40) : hit.end() + 40]
            if good.search(near):
                continue
            return True, " ".join(span[max(0, hit.start() - 90) : hit.end() + 90].split())
        return False, ""

    def mentioned(self, text: str) -> bool:
        """Whether the story restates the fact correctly at all."""
        subject_re = re.compile(rf"(?<!\w){re.escape(self.subject)}(?!\w)", re.IGNORECASE)
        good = self._pattern(_forms(self.value))
        for m in subject_re.finditer(text):
            lo, hi = max(0, m.start() - self.window), min(len(text), m.end() + self.window)
            if good.search(text[lo:hi]):
                return True
        return False


@dataclass(slots=True)
class Premise:
    story_id: str
    title: str
    setting: str
    characters: list[str]
    objects: list[str]
    canon: list[CanonFact] = field(default_factory=list)
    beats: list[str] = field(default_factory=list)

    def canon_block(self) -> str:
        return "\n".join(f"- {f.statement}" for f in self.canon)

    def to_json(self) -> dict:
        return {
            "story_id": self.story_id, "title": self.title, "setting": self.setting,
            "characters": self.characters, "objects": self.objects,
            "beats": self.beats,
            "canon": [
                {"fact_id": f.fact_id, "subject": f.subject, "kind": f.kind,
                 "value": f.value, "statement": f.statement}
                for f in self.canon
            ],
        }

    @classmethod
    def from_json(cls, d: dict) -> Premise:
        return cls(
            story_id=d["story_id"], title=d["title"], setting=d["setting"],
            characters=d["characters"], objects=d["objects"], beats=d["beats"],
            canon=[
                CanonFact(
                    fact_id=c["fact_id"], subject=c["subject"], kind=c["kind"],
                    value=c["value"],
                    statement=c.get("statement", f"{c['subject']}: {c['value']}"),
                )
                for c in d["canon"]
            ],
        )


_GIVEN = (
    "Miriam", "Halloran", "Tobias", "Ceridwen", "Ambrose", "Isolde", "Barnaby",
    "Rowena", "Silas", "Maud", "Cormac", "Perpetua", "Hesper", "Jarrow",
    "Lavinia", "Odell", "Winifred", "Gideon", "Araminta", "Thaddeus",
)
_OBJECTS = ("locket", "compass", "key", "flask", "ring", "lantern", "seal", "cipher-wheel")
_SETTINGS = (
    "a fog-bound harbour town losing its shipping trade",
    "a hill village whose only road washes out each spring",
    "a cathedral city where the bell foundry has just closed",
    "a moorland estate being sold off in parcels",
    "a canal port whose locks are silting up",
)
_BEATS = (
    "Introduce the household and the trouble that has arrived.",
    "A stranger's request forces a decision no one wants to make.",
    "An old debt surfaces and someone is asked to pay it.",
    "A journey is undertaken; something is left behind.",
    "A confidence is broken, and the wrong person learns of it.",
    "A search turns up the opposite of what was wanted.",
    "An accusation is made in public.",
    "Two people who have avoided each other are forced to speak.",
    "A hidden arrangement is exposed.",
    "An attempt at repair makes matters worse.",
    "Someone returns who was thought gone for good.",
    "A choice is made that cannot be taken back.",
    "The cost of the earlier decision becomes plain.",
    "An alliance shifts.",
    "What was buried is dug up.",
    "The account is settled, though not as anyone intended.",
    "A quiet aftermath, and one thing still unresolved.",
    "The unresolved thing resolves, at a price.",
    "A last reckoning between the two who began it.",
    "The town settles into what it will now be.",
)


def build_premises(n: int = 30, n_canon: int = 8, seed: int = 0) -> list[Premise]:
    """Deterministic premises with a fixed number of checkable planted facts."""
    rng = random.Random(seed)
    premises: list[Premise] = []
    for i in range(n):
        names = rng.sample(_GIVEN, 5)
        objects = rng.sample(_OBJECTS, 2)
        setting = _SETTINGS[i % len(_SETTINGS)]
        canon: list[CanonFact] = []

        def add(subject: str, kind: str, template: str,
                _canon: list[CanonFact] = canon, _i: int = i) -> None:
            value = rng.choice(VOCAB[kind])
            _canon.append(
                CanonFact(
                    fact_id=f"s{_i:02d}f{len(_canon):02d}",
                    subject=subject, kind=kind, value=value,
                    statement=template.format(subject=subject, value=value),
                )
            )

        add(names[0], "eyes", "{subject} has {value} eyes.")
        add(names[1], "hair", "{subject} has {value} hair.")
        add(objects[0], "metal", "The {subject} is {value}.")
        add(names[2], "trade", "{subject} is the village {value}.")
        add(names[3], "home", "{subject} was born in {value} and has never left it.")
        add(names[4], "eyes", "{subject} has {value} eyes.")
        add(objects[1], "metal", "The {subject} is {value}.")
        add(names[0], "trade", "{subject} works as a {value}.")
        canon = canon[:n_canon]

        premises.append(
            Premise(
                story_id=f"story{i:02d}",
                title=f"The {objects[0].title()} of {names[0]}",
                setting=setting,
                characters=names,
                objects=objects,
                canon=canon,
                beats=list(_BEATS),
            )
        )
    return premises


@dataclass(slots=True)
class CanonReport:
    chapter: int
    violations: list[str]
    restated: list[str]
    evidence: dict[str, str]

    @property
    def n_violations(self) -> int:
        return len(self.violations)


def check_chapter(premise: Premise, text: str, chapter: int) -> CanonReport:
    violations, restated, evidence = [], [], {}
    for fact in premise.canon:
        bad, why = fact.check(text)
        if bad:
            violations.append(fact.fact_id)
            evidence[fact.fact_id] = why
        if fact.mentioned(text):
            restated.append(fact.fact_id)
    return CanonReport(chapter=chapter, violations=violations,
                       restated=restated, evidence=evidence)
