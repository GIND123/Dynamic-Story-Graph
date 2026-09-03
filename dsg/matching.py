"""Pure surface-matching used for entity resolution.

Kept deliberately conservative. In 19th-century fiction the surname is the
single most dangerous cue -- 'Miss Bennet' and 'Elizabeth Bennet' are different
people -- so a shared surname alone never licenses a merge. It only ever
*raises a question*, which deferred commitment then holds open until naming or
apposition evidence settles it.
"""

from __future__ import annotations

import re

from dsg.lexicon import is_description, is_proper_name

_TITLE_RE = re.compile(
    r"^(mr|mister|mrs|missis|missus|miss|ms|dr|doctor|sir|lady|lord|master|"
    r"mistress|captain|capt|colonel|col|major|general|gen|professor|prof|"
    r"reverend|rev|madame|mme|monsieur|m|aunt|uncle|st|saint)\.?\s+",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s']+")
_WS_RE = re.compile(r"\s+")

# Kinship/role words that look like names after a title but denote a relation.
_ROLE_WORDS = frozenset({"father", "mother", "sister", "brother", "aunt", "uncle",
                         "son", "daughter", "wife", "husband", "cousin", "nurse"})

# Titles are *distinguishing*, not decorative. In the fiction this corpus is
# drawn from, "Mr. Bennet", "Mrs. Bennet" and "Miss Bennet" are three
# different people who share a surname, so stripping the title before
# comparing collapses a household into one node. Two surfaces that both carry
# a title match only when the titles are the same or spelling variants.
_TITLE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"mr", "mister"}),
    frozenset({"mrs", "missis", "missus"}),
    frozenset({"miss", "ms"}),
    frozenset({"dr", "doctor"}),
    frozenset({"professor", "prof"}),
    frozenset({"reverend", "rev"}),
    frozenset({"captain", "capt"}),
    frozenset({"colonel", "col"}),
    frozenset({"major"}),
    frozenset({"general", "gen"}),
    frozenset({"sir"}),
    frozenset({"lady"}),
    frozenset({"lord"}),
    frozenset({"master"}),
    frozenset({"mistress"}),
    frozenset({"madame", "mme"}),
    frozenset({"monsieur", "m"}),
    frozenset({"st", "saint"}),
    frozenset({"aunt"}),
    frozenset({"uncle"}),
)


def title_of(surface: str) -> str:
    """The honorific a surface opens with, normalised, or the empty string."""
    m = _TITLE_RE.match((surface or "").strip())
    return m.group(1).lower().rstrip(".") if m else ""


def titles_conflict(a: str, b: str) -> bool:
    """True when both carry a title and those titles denote different people."""
    ta, tb = title_of(a), title_of(b)
    if not ta or not tb or ta == tb:
        return False
    return not any(ta in group and tb in group for group in _TITLE_GROUPS)


class MatchKind:
    EXACT = "exact"
    NAME_SUBSET = "name_subset"   # 'Elizabeth' vs 'Elizabeth Bennet'
    TITLED = "titled"             # 'Mr. Darcy' vs 'Darcy'
    SURNAME_ONLY = "surname_only" # ambiguous: never auto-merges
    NONE = "none"


def normalize(surface: str) -> str:
    s = _PUNCT_RE.sub(" ", (surface or "").strip().lower())
    return _WS_RE.sub(" ", s).strip()


def strip_title(surface: str) -> str:
    return _TITLE_RE.sub("", (surface or "").strip()).strip()


def tokens(surface: str) -> list[str]:
    return [t for t in normalize(strip_title(surface)).split() if t]


def is_role_reference(surface: str) -> bool:
    toks = tokens(surface)
    return bool(toks) and toks[-1] in _ROLE_WORDS


def match_kind(a: str, b: str) -> str:
    """Classify how two surface forms relate. Symmetric."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return MatchKind.NONE
    if na == nb:
        return MatchKind.EXACT
    if titles_conflict(a, b):
        # Same surname, different honorific: a household, not a person.
        return MatchKind.NONE
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return MatchKind.NONE
    if ta == tb:
        return MatchKind.TITLED
    sa, sb = set(ta), set(tb)
    if not sa & sb:
        return MatchKind.NONE
    if sa <= sb or sb <= sa:
        # A single shared token that is the *last* token of both multi-token
        # names is a bare surname overlap -- the Bennet-sisters trap.
        shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
        if len(shorter) == 1 and len(longer) > 1 and shorter[0] == longer[-1]:
            return MatchKind.SURNAME_ONLY
        return MatchKind.NAME_SUBSET
    return MatchKind.NONE


# How much each match kind counts as evidence that two mentions corefer.
# Only scores at or above ``MERGE_THRESHOLD`` license an automatic merge.
MATCH_SCORE: dict[str, float] = {
    MatchKind.EXACT: 1.0,
    MatchKind.TITLED: 0.9,
    MatchKind.NAME_SUBSET: 0.75,
    MatchKind.SURNAME_ONLY: 0.35,
    MatchKind.NONE: 0.0,
}

MERGE_THRESHOLD = 0.7


def surface_score(a: str, b: str) -> float:
    return MATCH_SCORE[match_kind(a, b)]


def best_surface_score(surfaces_a: dict[str, int] | set[str], surface: str) -> float:
    return max((surface_score(s, surface) for s in surfaces_a), default=0.0)


def classify_surface(surface: str) -> str:
    """``name`` | ``description`` | ``other`` -- decides provisional vs committed."""
    s = (surface or "").strip()
    if not s:
        return "other"
    if is_description(s) or is_role_reference(s):
        return "description"
    if is_proper_name(s):
        return "name"
    return "other"
