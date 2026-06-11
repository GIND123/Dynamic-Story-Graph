"""Idempotent demo-story seeder.

Seeds "The Ashen Crown" with two static chapters so the demo can show
Brann already dead and chapter-3 generation can demonstrate the gate.

Run with: docker compose exec api python -m app.seed
"""

from app import graph
from app.models import (
    CharacterSeed,
    ChapterExtraction,
    ExtractedCharacter,
    ExtractedEvent,
    ManuscriptCreate,
)
from app.vectors import embed_one

# Stable ID so re-running seed is a no-op (MERGE on uid)
SEED_MID = "ashen_crown_seed_v1"

_CHAPTER_1_TEXT = (
    "The streets of Duskwall ran grey under an ashen sky. Three figures moved "
    "with purpose through the mourning city: Mara, whose hands knew every lock "
    "and shadow of the docks; Brann, the captain whose armour bore the crown's "
    "seal; and Sera, the archivist who kept the city's oldest secrets. The king "
    "was three days dead, and something ancient was stirring beneath the cobblestones."
)

_CHAPTER_2_TEXT = (
    "The Citadel gate fell at midnight. Brann had held the eastern arch alone for "
    "two hours, turning back the masked assailants one by one. When Mara and Sera "
    "arrived at dawn, they found the gate sealed and Brann's body laid at its foot, "
    "his sword still in his hand. He had bought the city one more sunrise. "
    "Mara did not speak. Sera wrote everything down."
)


def seed() -> None:
    if graph.manuscript_exists(SEED_MID):
        print(f"Already seeded — skipping.  Manuscript ID: {SEED_MID}")
        return

    body = ManuscriptCreate(
        title="The Ashen Crown",
        premise=(
            "A city-state called Duskwall reels after its king's sudden death. "
            "Three unlikely figures — a smuggler, a guard captain, and an archivist — "
            "hold the key to its survival."
        ),
        characters=[
            CharacterSeed(
                name="Mara",
                traits="pragmatic smuggler, reads people quickly",
                location="Duskwall docks",
            ),
            CharacterSeed(
                name="Brann",
                traits="royal guard captain, loyal and direct",
                location="Citadel",
            ),
            CharacterSeed(
                name="Sera",
                traits="meticulous archivist, keeper of old secrets",
                location="Old Library",
            ),
        ],
        rules=["The dead do not return; Duskwall has no resurrection magic."],
    )
    graph.create_manuscript(body, manuscript_id=SEED_MID)

    print("Computing embeddings for seed chapters...")

    # Chapter 1 — the trio meet
    ch1 = ChapterExtraction(
        characters=[
            ExtractedCharacter(name="Mara", status="alive", note="introduced"),
            ExtractedCharacter(name="Brann", status="alive", note="introduced"),
            ExtractedCharacter(name="Sera", status="alive", note="introduced"),
        ],
        events=[
            ExtractedEvent(
                summary="Mara, Brann, and Sera discover the king is dead and join forces",
                involves=["Mara", "Brann", "Sera"],
            )
        ],
    )
    graph.commit_chapter(
        SEED_MID, 1, _CHAPTER_1_TEXT, ch1, embedding=embed_one(_CHAPTER_1_TEXT)
    )

    # Chapter 2 — Brann dies
    ch2 = ChapterExtraction(
        characters=[
            ExtractedCharacter(name="Mara", status="alive", note="arrives at dawn"),
            ExtractedCharacter(
                name="Brann", status="dead", note="fell defending the Citadel gate"
            ),
            ExtractedCharacter(name="Sera", status="alive", note="documents events"),
        ],
        events=[
            ExtractedEvent(
                summary="Brann falls defending the Citadel gate",
                involves=["Brann", "Mara", "Sera"],
            )
        ],
    )
    graph.commit_chapter(
        SEED_MID, 2, _CHAPTER_2_TEXT, ch2, embedding=embed_one(_CHAPTER_2_TEXT)
    )

    print("Seeded 'The Ashen Crown' successfully.")
    print(f"Manuscript ID : {SEED_MID}")
    print("Canon state   : Brann is dead. Mara and Sera are alive.")
    print(f"\nexport MID={SEED_MID}")


if __name__ == "__main__":
    seed()
