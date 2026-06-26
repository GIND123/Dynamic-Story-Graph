from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class RelationType(str, Enum):
    """The closed set of typed relationships the graph models.

    14 types — no PRECEDES (event ordering is carried by sequence_index /
    seq_introduced) and no mental-state relations (Belief/Emotion/Intention/
    Desire are intentionally out of scope). Being a str-Enum, each member's
    .value is the literal Neo4j edge type, which graph.merge_relation maps to
    a parameterized query.

    SPEAKS_TO is a thin direct dialogue edge (CHARACTER->CHARACTER) carrying
    quote_type + a source_ref, added as a single-hop view for first-class
    quote-attribution queries. It coexists with the EVENT+INVOLVES dialogue
    representation that ingestion already builds; neither replaces the other.
    """

    LOCATED_AT = "LOCATED_AT"
    PART_OF = "PART_OF"
    POSSESSES = "POSSESSES"
    CONTROLS = "CONTROLS"
    FAMILY_OF = "FAMILY_OF"
    RELATES_TO = "RELATES_TO"
    MEMBER_OF = "MEMBER_OF"
    IDENTIFIED_AS = "IDENTIFIED_AS"
    HAS_TRAIT = "HAS_TRAIT"
    OWES = "OWES"
    KNOWS_ABOUT = "KNOWS_ABOUT"
    INVOLVES = "INVOLVES"
    CAUSED = "CAUSED"
    SPEAKS_TO = "SPEAKS_TO"


class ExtractedRelation(BaseModel):
    """A single typed edge between two named entities.

    Only `type`, `source`, and `target` are required; every discriminator
    field is optional so the same flat model covers all 13 relation types.
    The relevant discriminator per type (kept here for reference):
      FAMILY_OF      -> subtype  (parent|child|sibling|spouse|relative)
      RELATES_TO     -> stance (ally|enemy|neutral), romantic (bool)
      HAS_TRAIT      -> category (physical|personality|skill),
                        trait_key (normalized attribute, e.g. eye_color, height),
                        trait_value (e.g. blue, tall)
      IDENTIFIED_AS  -> kind (alias|title|disguise)
      OWES           -> kind (debt|promise|loyalty|oath)
      MEMBER_OF      -> role
      INVOLVES       -> role (agent|patient|witness)
      SPEAKS_TO      -> quote_type (explicit|implicit|anaphoric)
    """

    type: RelationType
    source: str
    target: str
    subtype: str | None = None
    stance: str | None = None
    romantic: bool | None = None
    category: str | None = None
    kind: str | None = None
    role: str | None = None
    quote_type: str | None = None
    # Structured HAS_TRAIT split: a normalized attribute name and its value, so
    # the trait-contradiction check matches on a stable key (no string parsing).
    trait_key: str | None = None
    trait_value: str | None = None
    confidence: float | None = None


class CharacterSeed(BaseModel):
    name: str
    traits: str
    location: str
    status: Literal["alive", "dead", "missing"] = "alive"


class ManuscriptCreate(BaseModel):
    title: str
    premise: str
    characters: list[CharacterSeed]
    rules: list[str] = Field(default_factory=list)


class ChapterRequest(BaseModel):
    scene_hint: str | None = None


class ExtractedCharacter(BaseModel):
    name: str
    status: Literal["alive", "dead", "missing"]
    note: str = Field(description="What changed for this character in this chapter")


class ExtractedEvent(BaseModel):
    summary: str
    involves: list[str] = Field(description="Character names involved")


class ChapterExtraction(BaseModel):
    characters: list[ExtractedCharacter]
    events: list[ExtractedEvent]
    # Optional, defaults to empty so every existing extraction stays valid and
    # the generation pipeline (which does not yet populate relations) is unchanged.
    relations: list[ExtractedRelation] = Field(default_factory=list)


class Contradiction(BaseModel):
    violated_fact: str = Field(description="The canon fact, quoted verbatim from CANON")
    draft_evidence: str = Field(description="The draft passage that violates it")


class JudgeVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    contradictions: list[Contradiction]
    coherence_score: float = Field(ge=0, le=1)
    reasoning: str
