"""Scoring generated stories against their planted canon.

The primary quantity is **cumulative canon violation**: by chapter c, what
fraction of the facts established in chapter 1 has the story contradicted at
least once? A fact once contradicted stays contradicted -- continuity, once
broken, is broken -- so the curve is monotone in c and its slope is the rate at
which a condition forgets.

Everything here is deterministic string matching against facts we planted, so
no model and no human judgement enters the measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dsg.eval.bootstrap import paired_bootstrap
from dsg.generate.canon import Premise, beat_size


def on_premise(text: str, premise) -> bool:
    """Does this chapter actually write about the story it was asked for?

    Canon violations are only counted near a mention of their subject, so a
    model that wanders off the premise entirely scores a perfect violation rate
    while producing nothing usable. Without this denominator that artefact
    reads as a win.
    """
    import re

    for name in premise.characters:
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
            return True
    return False


# ------------------------------------------------------- readability proxies
#
# Deterministic, like the rest of this module: no model reads the prose. They
# are diagnostics of the reviewers' complaints -- stiff, padded, flat -- not a
# measure of quality, and the blinded human read is the outcome they explain.

_WORD = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z]+)?")
# A quotation never spans a paragraph here: an unmatched quote at the end of a
# paragraph is left out rather than pairing with the next one and inverting
# everything after it.
_QUOTED = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d')
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
    "for", "with", "from", "into", "onto", "as", "is", "was", "were", "be", "been",
    "being", "it", "its", "he", "she", "they", "them", "his", "her", "their", "him",
    "i", "you", "we", "us", "our", "your", "my", "me", "that", "this", "these", "those",
    "there", "here", "then", "than", "so", "not", "no", "do", "did", "does", "had",
    "has", "have", "would", "could", "should", "will", "can", "may", "might", "must",
    "up", "down", "out", "over", "all", "what", "which", "who", "whom", "when", "where",
    "why", "how"
})


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def dialogue_share(text: str) -> float:
    """Share of a chapter's words that sit inside quotation marks."""
    total = word_count(text)
    if not total:
        return 0.0
    return sum(word_count(q) for q in _QUOTED.findall(text)) / total


def phrase_repeats(text: str, seen: set[tuple[str, ...]], n: int = 4) -> tuple[int, int]:
    """(repeated, counted) word n-grams, against everything in ``seen``.

    ``seen`` is updated in place, so feeding a story chapter by chapter counts
    a phrase that recurs within a chapter or anywhere earlier in the story.
    Grams made only of function words ("and then he was") repeat in any prose
    and are not counted.
    """
    tokens = words(text)
    repeated = counted = 0
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        if all(t in _STOPWORDS for t in gram):
            continue
        counted += 1
        if gram in seen:
            repeated += 1
        seen.add(gram)
    return repeated, counted


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


@dataclass(slots=True)
class RunScore:
    story_id: str
    condition: str
    n_canon: int
    chapters: int
    total_chars: int
    violated: set[str] = field(default_factory=set)
    first_violation: dict[str, int] = field(default_factory=dict)
    restated_ever: set[str] = field(default_factory=set)
    cumulative: list[float] = field(default_factory=list)   # per chapter
    per_chapter: list[int] = field(default_factory=list)
    on_premise_chapters: int = 0
    n_chapters_seen: int = 0
    prompt_tokens_total: int = 0
    prompt_tokens_last: int = 0
    state_facts_final: int = 0
    state_entities_final: int = 0
    canon_in_state_final: int = 0
    rollbacks_final: int = 0
    # Readability proxies, accumulated chapter by chapter.
    words_total: int = 0
    dialogue_words: float = 0.0
    phrases_repeated: int = 0
    phrases_counted: int = 0
    seen_phrases: set[tuple[str, ...]] = field(default_factory=set)
    words_by_size: dict[str, list[int]] = field(default_factory=dict)
    budget_ratios: list[float] = field(default_factory=list)
    restated_later: list[int] = field(default_factory=list)
    truncated_chapters: int = 0

    @property
    def violation_rate(self) -> float:
        return len(self.violated) / self.n_canon if self.n_canon else 0.0

    @property
    def retention(self) -> float:
        return 1.0 - self.violation_rate

    @property
    def restatement_rate(self) -> float:
        return len(self.restated_ever) / self.n_canon if self.n_canon else 0.0

    def as_dict(self) -> dict:
        return {
            "story_id": self.story_id, "condition": self.condition,
            "n_canon": self.n_canon, "chapters": self.chapters,
            "total_chars": self.total_chars,
            "violation_rate": self.violation_rate, "retention": self.retention,
            "restatement_rate": self.restatement_rate,
            "n_violated": len(self.violated),
            "mean_first_violation": (
                sum(self.first_violation.values()) / len(self.first_violation)
                if self.first_violation else float(self.chapters + 1)
            ),
            "on_premise": (
                self.on_premise_chapters / self.n_chapters_seen
                if self.n_chapters_seen else 0.0
            ),
            "prompt_tokens_total": float(self.prompt_tokens_total),
            "prompt_tokens_last": float(self.prompt_tokens_last),
            "canon_capture": (
                self.canon_in_state_final / self.n_canon if self.n_canon else 0.0
            ),
            "state_facts_final": float(self.state_facts_final),
            "state_entities_final": float(self.state_entities_final),
            "rollbacks_final": float(self.rollbacks_final),
            **self.prose_dict(),
        }

    def prose_dict(self) -> dict:
        """The readability proxies. ``None`` where a story has nothing to measure."""
        n = self.n_chapters_seen
        sized = {
            f"words_per_beat_{size}": _mean([float(w) for w in self.words_by_size.get(size, [])])
            for size in ("small", "medium", "large")
        }
        return {
            "dialogue_share": (
                self.dialogue_words / self.words_total if self.words_total else 0.0
            ),
            "repeated_phrase_rate": (
                self.phrases_repeated / self.phrases_counted if self.phrases_counted else 0.0
            ),
            "words_per_beat": self.words_total / n if n else 0.0,
            **sized,
            "budget_ratio": _mean(self.budget_ratios),
            "restated_after_ch1": _mean([float(r) for r in self.restated_later]),
            "truncated_rate": self.truncated_chapters / n if n else 0.0,
        }


def score_runs(payload: dict) -> list[RunScore]:
    premises = {p["story_id"]: Premise.from_json(p) for p in payload["premises"]}
    n_chapters = payload["meta"]["chapters"]

    scores: dict[tuple[str, str], RunScore] = {}
    for rec in sorted(payload["records"], key=lambda r: r["chapter"]):
        key = (rec["story_id"], rec["condition"])
        score = scores.get(key)
        if score is None:
            score = RunScore(
                story_id=rec["story_id"], condition=rec["condition"],
                n_canon=len(premises[rec["story_id"]].canon), chapters=n_chapters,
                total_chars=0,
            )
            scores[key] = score
        score.total_chars += rec["chars"]
        score.n_chapters_seen += 1
        _add_prose(score, rec, premises[rec["story_id"]])
        if rec.get("text") and on_premise(rec["text"], premises[rec["story_id"]]):
            score.on_premise_chapters += 1
        score.prompt_tokens_total += int(rec.get("prompt_tokens", 0))
        score.prompt_tokens_last = int(rec.get("prompt_tokens", 0))
        for fact_id in rec["violations"]:
            if fact_id not in score.violated:
                score.first_violation[fact_id] = rec["chapter"]
            score.violated.add(fact_id)
        score.restated_ever.update(rec["restated"])
        score.per_chapter.append(len(rec["violations"]))
        score.cumulative.append(
            len(score.violated) / score.n_canon if score.n_canon else 0.0
        )
        score.state_facts_final = rec.get("state_facts", 0)
        score.canon_in_state_final = len(rec.get("canon_in_state", []))
        score.state_entities_final = rec.get("state_entities", 0)
        score.rollbacks_final = rec.get("rollbacks", 0)
    return list(scores.values())


def _add_prose(score: RunScore, rec: dict, premise: Premise) -> None:
    text = rec.get("text", "")
    n_words = word_count(text)
    score.words_total += n_words
    score.dialogue_words += dialogue_share(text) * n_words
    repeated, counted = phrase_repeats(text, score.seen_phrases)
    score.phrases_repeated += repeated
    score.phrases_counted += counted
    if premise.beats:
        beat = premise.beats[(rec["chapter"] - 1) % len(premise.beats)]
        score.words_by_size.setdefault(beat_size(beat), []).append(n_words)
    if rec.get("word_budget"):
        score.budget_ratios.append(n_words / rec["word_budget"])
    if rec["chapter"] > 1:
        score.restated_later.append(len(rec.get("restated", [])))
    score.truncated_chapters += bool(rec.get("truncated"))


def curves(scores: list[RunScore]) -> dict[str, list[float]]:
    """Mean cumulative violation rate by chapter, per condition."""
    by_condition: dict[str, list[list[float]]] = {}
    for s in scores:
        by_condition.setdefault(s.condition, []).append(s.cumulative)
    out: dict[str, list[float]] = {}
    for condition, series in by_condition.items():
        depth = min(len(x) for x in series)
        out[condition] = [
            sum(x[i] for x in series) / len(series) for i in range(depth)
        ]
    return out


COMPARISONS = (
    # Does any memory help at all?
    ("base:rolling-summary", "base:none"),
    ("base:full-context", "base:rolling-summary"),
    # What does a maintained state buy over the practical alternatives?
    ("base:dsg-state", "base:none"),
    ("base:dsg-state", "base:rolling-summary"),
    ("base:dsg-state", "base:full-context"),
    # Does the state need to be revisable?
    ("base:dsg-state", "base:append-only-state"),
    # What does training the writer to use the state buy?
    ("tuned:dsg-state", "base:dsg-state"),
    ("tuned:full-context", "base:full-context"),
    # And the full system against the strongest practical baseline.
    ("tuned:dsg-state", "tuned:full-context"),
    ("tuned:dsg-repair", "tuned:dsg-state"),
    ("tuned:dsg-repair", "base:full-context"),
    # The hybrid: does a digest plus recent text beat either alone?
    ("base:dsg-hybrid", "base:dsg-state"),
    ("base:dsg-hybrid", "base:last-chapter"),
    ("base:dsg-hybrid", "base:full-context"),
    ("base:dsg-hybrid", "base:none"),
    ("base:dsg-hybrid-repair", "base:dsg-hybrid"),
    # Beat-conditioned retrieval against the incumbent and the failing cell.
    ("base:beat-retrieval", "base:dsg-state"),
    ("base:beat-retrieval", "base:none"),
    ("base:beat-hybrid", "base:last-chapter"),
    ("base:beat-hybrid", "base:full-context"),
    ("base:beat-hybrid", "base:beat-retrieval"),
    ("base:beat-hybrid-repair", "base:beat-hybrid"),
    # Test time selection. The random arm isolates the graph from the
    # extra sampling: both draw four candidates at the same budget.
    ("base:last-chapter@bo4-graph", "base:last-chapter"),
    ("base:last-chapter@bo4-random", "base:last-chapter"),
    ("base:last-chapter@bo4-graph", "base:last-chapter@bo4-random"),
    ("base:last-chapter@bo4-graph", "base:full-context"),
)

METRICS = ("violation_rate", "retention", "restatement_rate", "on_premise",
           "canon_capture",
           "mean_first_violation", "total_chars",
           "prompt_tokens_total", "prompt_tokens_last")


def compare(scores: list[RunScore]) -> dict:
    by_condition: dict[str, dict[str, dict]] = {}
    for s in scores:
        by_condition.setdefault(s.condition, {})[s.story_id] = s.as_dict()
    out: dict[str, dict] = {}
    for a, b in COMPARISONS:
        if a not in by_condition or b not in by_condition:
            continue
        stories = sorted(set(by_condition[a]) & set(by_condition[b]))
        if not stories:
            continue
        entry: dict[str, object] = {"n_stories": len(stories)}
        for metric in METRICS:
            xs = [float(by_condition[a][s][metric]) for s in stories]
            ys = [float(by_condition[b][s][metric]) for s in stories]
            entry[metric] = paired_bootstrap(xs, ys).as_dict()
        out[f"{a}_vs_{b}"] = entry
    return out


def summarize(scores: list[RunScore], order: tuple[str, ...]) -> str:
    cols = ("on_premise", "violation_rate", "retention", "restatement_rate",
            "canon_capture",
            "mean_first_violation", "total_chars", "prompt_tokens_last")
    lines = ["| condition | n | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for condition in order:
        rows = [s.as_dict() for s in scores if s.condition == condition]
        if not rows:
            continue
        cells = []
        for c in cols:
            mean = sum(r[c] for r in rows) / len(rows)
            cells.append(
                f"{mean:,.0f}" if c in ("total_chars", "prompt_tokens_last")
                else f"{mean:.3f}"
            )
        lines.append(f"| {condition} | {len(rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------- prose template vs classic

PROSE_METRICS = (
    "dialogue_share", "repeated_phrase_rate", "words_per_beat",
    "words_per_beat_small", "words_per_beat_medium", "words_per_beat_large",
    "budget_ratio", "truncated_rate",
)
# A prose gain only counts if these hold. Margins are absolute and one-sided:
# the prose arm may not break more canon, or wander off the premise more, by
# more than this at the edge of the 95% interval.
GUARDS = {"violation_rate": +0.05, "on_premise": -0.05}
# Reported beside the guards, with no verdict.
DIAGNOSTICS = ("restatement_rate", "restated_after_ch1")
# Arms that capture different shares of canon are not reading the same state.
CAPTURE_FLAG = 0.10


def classic_twin(condition: str) -> str:
    """``base:dsg-hybrid~prose@bo4-graph`` -> ``base:dsg-hybrid@bo4-graph``."""
    base, at, selection = condition.partition("@")
    return base.split("~", 1)[0] + at + selection


def compare_templates(scores: list[RunScore]) -> dict:
    """Every prose arm against its classic twin, paired within story."""
    from dsg.generate.conditions import STATE_MEMORIES, memory_of, template_of

    by_condition: dict[str, dict[str, dict]] = {}
    for s in scores:
        by_condition.setdefault(s.condition, {})[s.story_id] = s.as_dict()
    out: dict[str, dict] = {}
    for prose in sorted(c for c in by_condition if template_of(c) == "prose"):
        classic = classic_twin(prose)
        if classic not in by_condition:
            continue
        stories = sorted(set(by_condition[prose]) & set(by_condition[classic]))
        if not stories:
            continue
        entry: dict[str, object] = {"prose": prose, "classic": classic,
                                    "n_stories": len(stories), "metrics": {}}
        for metric in PROSE_METRICS + tuple(GUARDS) + DIAGNOSTICS:
            pairs = [
                (by_condition[prose][s][metric], by_condition[classic][s][metric])
                for s in stories
            ]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            entry["metrics"][metric] = paired_bootstrap(
                [float(a) for a, _ in pairs], [float(b) for _, b in pairs]
            ).as_dict()
        verdicts = {}
        for metric, margin in GUARDS.items():
            stat = entry["metrics"][metric]
            edge = stat["ci_high"] if margin > 0 else stat["ci_low"]
            holds = edge <= margin if margin > 0 else edge >= margin
            verdicts[metric] = {"margin": margin, "edge": edge, "holds": bool(holds)}
        entry["guards"] = verdicts
        if memory_of(prose) in STATE_MEMORIES:
            a = sum(by_condition[prose][s]["canon_capture"] for s in stories) / len(stories)
            b = sum(by_condition[classic][s]["canon_capture"] for s in stories) / len(stories)
            entry["canon_capture"] = {"prose": a, "classic": b,
                                      "flagged": abs(a - b) > CAPTURE_FLAG}
        else:
            entry["canon_capture"] = None   # no state in this memory
        out[f"{prose}_vs_{classic}"] = entry
    return out


# ------------------------------------------------------------ example leakage
#
# The example passage is meant to lend a voice, not a plot. Leakage is the share
# of its distinctive content words -- not stopwords, not words the premise
# itself supplies -- that turn up in a story. The classic arm never sees the
# example, so its share is the baseline: what this premise would produce
# anyway.


def distinctive_words(example: str, premise: dict) -> set[str]:
    """The example's content words that this premise does not already supply."""
    supplied = " ".join([
        premise.get("title", ""), premise.get("setting", ""),
        " ".join(premise.get("characters", [])), " ".join(premise.get("objects", [])),
        " ".join(c.get("statement", "") for c in premise.get("canon", [])),
        " ".join(premise.get("beats", [])),
    ])
    return set(words(example)) - _STOPWORDS - set(words(supplied))


def example_leakage(payload: dict, sharp: bool = False) -> tuple[dict, dict]:
    """Per arm and story, the share of the example's distinctive words used.

    ``sharp`` also drops, for each story, every example word the classic arm
    writes in the *other* stories: words this model uses with no example at all
    ("door", "said", "hand"), which otherwise swamp the few that betray the
    example ("porch", "truck"). Leave-one-story-out, so a story's own classic
    chapters still count towards its baseline.

    Returns (shares, words used) keyed by condition then story; both are empty
    when the run carries no example text.
    """
    from dsg.generate.conditions import template_of

    example = payload.get("meta", {}).get("example", "")
    if not example:
        return {}, {}
    premises = {p["story_id"]: p for p in payload["premises"]}
    texts: dict[tuple[str, str], list[str]] = {}
    for rec in payload["records"]:
        texts.setdefault((rec["condition"], rec["story_id"]), []).append(rec.get("text", ""))
    vocab = {key: set(words(" ".join(chapters))) for key, chapters in texts.items()}
    classic_by_story: dict[str, set[str]] = {}
    for (condition, story), used in vocab.items():
        if template_of(condition) == "classic":
            classic_by_story.setdefault(story, set()).update(used)
    shares: dict[str, dict[str, float]] = {}
    leaked: dict[str, dict[str, list[str]]] = {}
    for (condition, story), used in vocab.items():
        distinct = distinctive_words(example, premises[story])
        if sharp:
            for other, seen in classic_by_story.items():
                if other != story:
                    distinct -= seen
        hit = distinct & used
        shares.setdefault(condition, {})[story] = len(hit) / len(distinct) if distinct else 0.0
        leaked.setdefault(condition, {})[story] = sorted(hit)
    return shares, leaked


def compare_leakage(payload: dict) -> dict:
    """Each prose arm's leakage against its classic twin's baseline, paired by story.

    Both measures are reported: ``as_specified`` (stopwords and premise words
    removed) and ``sharp`` (also words the classic arm uses elsewhere), with the
    words only the prose arm used, as evidence.
    """
    from dsg.generate.conditions import template_of

    out: dict[str, dict] = {}
    for name, sharp in (("as_specified", False), ("sharp", True)):
        shares, leaked = example_leakage(payload, sharp=sharp)
        for prose in sorted(c for c in shares if template_of(c) == "prose"):
            classic = classic_twin(prose)
            stories = sorted(set(shares[prose]) & set(shares.get(classic, {})))
            if not stories:
                continue
            entry = out.setdefault(prose, {"classic": classic})
            entry[name] = paired_bootstrap(
                [shares[prose][s] for s in stories], [shares[classic][s] for s in stories]
            ).as_dict()
            if sharp:
                only_prose = set().union(*(set(leaked[prose][s]) - set(leaked[classic][s])
                                           for s in stories))
                entry["prose_only_words"] = sorted(only_prose)
    return out


def example_name_check(payload: dict) -> dict[str, dict[str, list]]:
    """Capitalised example words a prose arm wrote and the classic arm never did.

    The distinctive-word shares dilute a leaked name to almost nothing -- two
    names among a hundred-odd words -- so names are checked on their own.
    Case-sensitive and whole-word; classic, which never sees the example, is
    the baseline across every one of its chapters in the run. Returns, per
    prose arm, each flagged word and the (story, chapter) places it appears.
    """
    from dsg.generate.conditions import template_of

    example = payload.get("meta", {}).get("example", "")
    if not example:
        return {}
    capitals = sorted(set(re.findall(r"\b[A-Z][a-z'\u2019]+\b", example)))
    patterns = {w: re.compile(rf"(?<![\w'\u2019]){re.escape(w)}(?![\w'\u2019])")
                for w in capitals}
    classic_text = " ".join(
        r.get("text", "") for r in payload["records"]
        if template_of(r["condition"]) == "classic"
    )
    in_classic = {w for w, pat in patterns.items() if pat.search(classic_text)}
    out: dict[str, dict[str, list]] = {}
    for rec in payload["records"]:
        if template_of(rec["condition"]) != "prose":
            continue
        arm = out.setdefault(rec["condition"], {})
        for word, pat in patterns.items():
            if word not in in_classic and pat.search(rec.get("text", "")):
                arm.setdefault(word, []).append((rec["story_id"], rec["chapter"]))
    return {arm: words for arm, words in sorted(out.items())}


# ------------------------------------------------------------ verbatim copying
#
# The share and name checks can both pass while a chapter recites its source:
# v2 had one prose chapter reproduce the example nearly line for line, under
# its own beat. A run of shared words cannot hide that.

COPY_FLAG = 8   # a shared run of this many words or more is flagged


def _grams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def longest_shared_run(a: str, b: str) -> int:
    """The longest run of consecutive words (case-folded) that ``a`` and ``b`` share.

    Sharing a run of n words implies sharing one of every shorter length, so
    the length is found by binary search over n-gram sets.
    """
    x, y = words(a), words(b)
    lo, hi = 0, min(len(x), len(y))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _grams(x, mid) & _grams(y, mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def copy_check(payload: dict) -> dict[str, dict]:
    """Per arm: the longest run each chapter shares with the example passage
    (when the run had one) and with its own previous chapter, and which
    chapters reach ``COPY_FLAG`` words.
    """
    example = payload.get("meta", {}).get("example", "")
    texts = {(r["condition"], r["story_id"], r["chapter"]): r.get("text", "")
             for r in payload["records"]}
    out: dict[str, dict] = {}
    for (condition, story, chapter), text in sorted(texts.items()):
        arm = out.setdefault(condition, {
            "chapters": 0,
            "example": {"max": 0, "flagged": []} if example else None,
            "previous": {"max": 0, "flagged": []},
        })
        arm["chapters"] += 1
        sources = [("previous", texts.get((condition, story, chapter - 1)))]
        if example:
            sources.append(("example", example))
        for name, source in sources:
            if not source:
                continue
            run = longest_shared_run(text, source)
            arm[name]["max"] = max(arm[name]["max"], run)
            if run >= COPY_FLAG:
                arm[name]["flagged"].append([story, chapter, run])
    return out
