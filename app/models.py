from typing import Literal
from pydantic import BaseModel, Field


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


class Contradiction(BaseModel):
    violated_fact: str = Field(description="The canon fact, quoted verbatim from CANON")
    draft_evidence: str = Field(description="The draft passage that violates it")


class JudgeVerdict(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    contradictions: list[Contradiction]
    coherence_score: float = Field(ge=0, le=1)
    reasoning: str
