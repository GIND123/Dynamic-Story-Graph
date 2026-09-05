"""The fixed, documented lexicon the reconciler arbitrates with.

The single most consequential judgement in the calculus is whether a conflict
means *the world changed* or *the reader was wrong*. That judgement is not left
to the language model: it is decided by a small, inspectable table of predicate
mutability plus a list of surface revelation markers. The model proposes; this
table disposes.
"""

from __future__ import annotations

import re

# Predicates whose truth can legitimately change as the story advances. A later
# conflicting value means the story world moved on, so the earlier value stays
# in the record with a closed validity interval (SUPERSEDE).
MUTABLE: frozenset[str] = frozenset(
    {
        "location",
        "possesses",
        "alive",
        "married_to",
        "engaged_to",
        "stance_toward",
        "emotion",
        "occupation",
        "resides_in",
        "travelling_to",
        "knows",
        "member_of",
        "health",
        "wealth",
    }
)

# Predicates that cannot change within one story world. A later conflicting
# value means the reader's earlier belief was never true (REVISE).
IMMUTABLE: frozenset[str] = frozenset(
    {
        "parent_of",
        "child_of",
        "sibling_of",
        "relative_of",
        "identity_of",
        "true_name",
        "gender",
        "species",
        "birthplace",
        "eye_colour",
        "hair_colour",
        "material",
    }
)

# Predicates that admit many simultaneous values (no slot conflict on a new
# object; only a polarity flip on the *same* object is a conflict).
# Predicates that admit many simultaneous values. "occupation" belongs here:
# a character is legitimately wife, mother and breadwinner at once, and
# counting those as rivals was manufacturing contradictions in professionally
# edited novels.
MULTI_VALUED: frozenset[str] = frozenset(
    {
        "knows", "possesses", "sibling_of", "relative_of", "member_of",
        "stance_toward", "occupation",
    }
)

CANONICAL: frozenset[str] = MUTABLE | IMMUTABLE

# Free-text predicate strings a small model actually emits, mapped onto the
# canonical set. Anything unmatched falls through to ``other:<raw>``.
_SYNONYMS: dict[str, str] = {
    "is_at": "location", "at": "location", "located_at": "location",
    "in": "location", "is_in": "location", "place": "location",
    "lives_in": "resides_in", "lives_at": "resides_in", "home": "resides_in",
    "has": "possesses", "owns": "possesses", "carries": "possesses",
    "holds": "possesses", "possession": "possesses",
    "is_alive": "alive", "dead": "alive", "died": "alive", "is_dead": "alive",
    "wife_of": "married_to", "husband_of": "married_to", "spouse_of": "married_to",
    "married": "married_to", "marries": "married_to",
    "betrothed_to": "engaged_to", "engaged": "engaged_to",
    "loves": "stance_toward", "hates": "stance_toward", "trusts": "stance_toward",
    "distrusts": "stance_toward", "admires": "stance_toward",
    "feels": "emotion", "mood": "emotion", "emotion_of": "emotion",
    "job": "occupation", "works_as": "occupation", "profession": "occupation",
    "goes_to": "travelling_to", "travels_to": "travelling_to",
    "departs_for": "travelling_to", "journeys_to": "travelling_to",
    "aware_of": "knows", "knows_about": "knows", "learns": "knows",
    "belongs_to": "member_of", "member": "member_of",
    "father_of": "parent_of", "mother_of": "parent_of",
    "son_of": "child_of", "daughter_of": "child_of",
    "brother_of": "sibling_of", "sister_of": "sibling_of",
    "cousin_of": "relative_of", "uncle_of": "relative_of", "aunt_of": "relative_of",
    "nephew_of": "relative_of", "niece_of": "relative_of", "family_of": "relative_of",
    "is_really": "identity_of", "same_as": "identity_of", "actually_is": "identity_of",
    "real_name": "true_name", "named": "true_name", "name": "true_name",
    "sex": "gender", "is_a": "occupation",
    "eyes": "eye_colour", "eye_color": "eye_colour", "eye_colour": "eye_colour",
    "hair": "hair_colour", "hair_color": "hair_colour", "hair_colour": "hair_colour",
    "made_of": "material", "metal": "material", "material_of": "material",
    "composed_of": "material", "built_of": "material",
    "sick": "health", "ill": "health", "healthy": "health",
    "rich": "wealth", "poor": "wealth",
}

# Predicates whose *polarity* is one-way: once false, it cannot become true
# again. Used to catch impossible supersessions (resurrection) as violations.
ONE_WAY_FALSE: frozenset[str] = frozenset({"alive"})

# Surface cues that the narration is correcting a belief rather than reporting
# a change. Presence of any of these in the evidence promotes SUPERSEDE to
# REVISE even for a mutable predicate.
_REVELATION_PATTERNS: tuple[str, ...] = (
    r"\bin fact\b", r"\bin truth\b", r"\bturn(?:ed|s) out\b", r"\ball along\b",
    r"\bhad (?:always|never) been\b", r"\bwas (?:never|not really)\b",
    r"\bmistaken\b", r"\bactually\b", r"\breally was\b", r"\bthe truth (?:was|is)\b",
    r"\brevealed?\b", r"\breveal(?:s|ing)\b", r"\bdiscover(?:ed|s)\b",
    r"\blearn(?:ed|s|t) that\b", r"\bconfess(?:ed|es)\b", r"\badmitted\b",
    r"\bno longer believed\b", r"\brealis(?:ed|es)\b", r"\brealiz(?:ed|es)\b",
    r"\bfalsely\b", r"\bpretend(?:ed|ing)\b", r"\bdisguise[d]?\b", r"\bimpost(?:or|er)\b",
)
_REVELATION_RE = re.compile("|".join(_REVELATION_PATTERNS), re.IGNORECASE)

# Object strings that mark a slot as deliberately underspecified, so a later
# concrete value refines rather than contradicts it.
UNDERSPECIFIED_OBJECTS: frozenset[str] = frozenset(
    {
        "", "?", "unknown", "unspecified", "someone", "somebody", "something",
        "somewhere", "a place", "a man", "a woman", "a person", "a stranger",
        "the stranger", "a house", "a room", "elsewhere", "n/a", "none", "null",
        # A model asked for a property it cannot fill will say so rather than
        # omit the line. Such a "fact" carries nothing, crowds the digest, and
        # would teach a writer trained on it to produce the same non-answer.
        "not specified", "not mentioned", "not stated", "not given",
        "not described", "not applicable", "unclear", "undefined", "-",
        "no", "n/a.", "not known", "unnamed",
    }
)

# Generic head nouns: an earlier value that is one of these is refined, not
# contradicted, by a later value containing it ("a house" -> "Satis House").
_GENERIC_HEADS: frozenset[str] = frozenset(
    {"house", "room", "street", "town", "city", "village", "inn", "ship",
     "man", "woman", "girl", "boy", "child", "place", "road", "garden"}
)

# Definite/indefinite descriptions that should create a *provisional* node
# rather than a committed one -- the mechanism that makes a later identity
# reveal a monotone elaboration instead of a rollback.
_DESCRIPTION_RE = re.compile(
    r"^(the|a|an|his|her|their|my|your|that|this|some|another)\b", re.IGNORECASE
)

_TITLES = (
    "mr", "mrs", "miss", "ms", "dr", "sir", "lady", "lord", "master", "captain",
    "colonel", "major", "professor", "reverend", "madame", "monsieur", "aunt",
    "uncle", "st", "señor", "don", "dona",
)


def normalize_predicate(raw: str) -> str:
    """Map a free-text predicate onto the canonical set, or tag it ``other:``."""
    key = re.sub(r"[^a-z_]+", "_", (raw or "").strip().lower()).strip("_")
    if not key:
        return "other:unspecified"
    if key in CANONICAL:
        return key
    if key in _SYNONYMS:
        return _SYNONYMS[key]
    for suffix in ("_of", "_to", "_in", "_at"):
        stem = key[: -len(suffix)] if key.endswith(suffix) else key
        if stem in CANONICAL:
            return stem
        if stem in _SYNONYMS:
            return _SYNONYMS[stem]
    return f"other:{key}"


def is_mutable(predicate: str) -> bool:
    """Unknown predicates are treated as mutable: the conservative choice, since
    SUPERSEDE preserves history whereas REVISE throws a belief away."""
    return predicate not in IMMUTABLE


def is_multi_valued(predicate: str) -> bool:
    return predicate in MULTI_VALUED or predicate.startswith("other:")


def has_revelation_marker(text: str | None) -> bool:
    return bool(text) and _REVELATION_RE.search(text or "") is not None


def is_underspecified(value: str | None) -> bool:
    v = (value or "").strip().lower()
    return v in UNDERSPECIFIED_OBJECTS


def refines(earlier: str, later: str) -> bool:
    """True when ``later`` specifies what ``earlier`` left open.

    Includes plain specification: "dog" -> "Newfoundland dog" names the same
    thing more precisely and is not a contradiction. Extraction returns the same
    fact at different granularities all the time, and treating those as rival
    values is the single largest source of false contradictions.
    """
    before, after = earlier.strip().lower(), later.strip().lower()
    if not before or before == after:
        return False
    if is_underspecified(before):
        return True
    before_tokens, after_tokens = set(before.split()), set(after.split())
    if before_tokens and before_tokens < after_tokens:
        return True
    head = before.split()[-1] if before.split() else ""
    return head in _GENERIC_HEADS and head in after and len(after) > len(before)


def compatible(a: str, b: str) -> bool:
    """Two values that name the same thing at different precision."""
    return refines(a, b) or refines(b, a)


def is_description(surface: str) -> bool:
    """A definite/indefinite description, not a name: 'the stranger', 'an old man'."""
    s = (surface or "").strip()
    if not s:
        return False
    return bool(_DESCRIPTION_RE.match(s))


# Head nouns that make a description denote a person. Identity resolution is
# only ever asked about these: a model offered "the pool" or "a very large jar"
# alongside a cast list will cheerfully link them to a character.
PERSON_NOUNS: frozenset[str] = frozenset(
    {
        "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child",
        "children", "lady", "ladies", "gentleman", "gentlemen", "person", "people",
        "stranger", "strangers", "fellow", "friend", "friends", "companion",
        "father", "mother", "parent", "son", "daughter", "brother", "sister",
        "husband", "wife", "widow", "widower", "aunt", "uncle", "cousin", "nephew",
        "niece", "grandmother", "grandfather", "guest", "visitor", "neighbour",
        "neighbor", "servant", "maid", "butler", "footman", "nurse", "governess",
        "cook", "clerk", "doctor", "physician", "surgeon", "lawyer", "attorney",
        "detective", "inspector", "constable", "officer", "soldier", "sailor",
        "captain", "colonel", "major", "general", "priest", "clergyman", "parson",
        "vicar", "bishop", "professor", "teacher", "student", "scholar", "master",
        "mistress", "landlord", "landlady", "innkeeper", "driver", "coachman",
        "courier", "guide", "prisoner", "convict", "thief", "murderer", "victim",
        "traveller", "traveler", "passenger", "youth", "elder", "creature",
        "figure", "narrator", "speaker", "host", "hostess", "bride", "groom",
        "king", "queen", "prince", "princess", "duke", "duchess", "lord",
    }
)


# Predicates whose object is drawn from a small closed set. A small model will
# happily emit "alive = 1800" or "gender = reported"; such a fact carries no
# information and, worse, generates spurious supersessions when the next junk
# value arrives. Checked at parse time so every policy sees the same stream.
CLOSED_RANGE: dict[str, frozenset[str]] = {
    "alive": frozenset({"true", "false", "yes", "no", "alive", "dead", "living",
                        "deceased", "killed", "died", "survives"}),
    "gender": frozenset({"male", "female", "man", "woman", "boy", "girl",
                         "m", "f", "masculine", "feminine", "non-binary"}),
}


def is_valid_object(predicate: str, obj: str) -> bool:
    """False when a closed-range predicate is given a value outside its range."""
    allowed = CLOSED_RANGE.get(predicate)
    if allowed is None:
        return True
    return (obj or "").strip().lower().rstrip(".") in allowed


def is_person_description(surface: str) -> bool:
    """True when a description plausibly denotes a person rather than a thing.

    Deterministic and inspectable, like the rest of the lexicon: the head noun
    must be a person noun, or the phrase must open with a personal title.
    """
    s = (surface or "").strip().lower().rstrip(".,;:!?")
    if not s:
        return False
    tokens = [t for t in re.split(r"[^a-z'-]+", s) if t]
    if not tokens:
        return False
    if tokens[0] in _TITLES:
        return True
    return tokens[-1] in PERSON_NOUNS


def is_proper_name(surface: str) -> bool:
    """Heuristic proper-name test tuned for 19th/20th-century fiction."""
    s = (surface or "").strip()
    if not s or is_description(s):
        return False
    tokens = [t for t in re.split(r"\s+", s) if t]
    if not tokens or len(tokens) > 4:
        return False
    head = tokens[0].rstrip(".").lower()
    if head in _TITLES:
        return True
    return all(t[:1].isupper() for t in tokens if t[:1].isalpha())
