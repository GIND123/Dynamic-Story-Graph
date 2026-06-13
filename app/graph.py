import logging
import uuid

from neo4j import Driver, GraphDatabase

from app.config import EMBED_DIM, settings
from app.models import ChapterExtraction, ManuscriptCreate

logger = logging.getLogger(__name__)

_driver: Driver | None = None


def init_driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


def ensure_constraints() -> None:
    """Create uniqueness constraints for every node label.

    Raises on connection failure so the lifespan retry loop can catch it.
    Labels and constraint names are hardcoded constants — f-strings are safe
    here as an explicit exception to the no-f-string-in-Cypher rule (Cypher
    does not support parameterized label or constraint names).
    """
    drv = init_driver()
    labels = ["Manuscript", "Character", "Location", "Event", "Chapter", "Rule"]
    with drv.session() as session:
        for label in labels:
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_uid IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.uid IS UNIQUE"
            )
        # Vector index for hybrid retrieval (EMBED_DIM imported from config — single source)
        session.run(
            "CREATE VECTOR INDEX passage_index IF NOT EXISTS "
            "FOR (ch:Chapter) ON (ch.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine'}}}}"
        )
        # Block until the index is online before serving traffic (CLAUDE.md §13.4)
        session.run("CALL db.awaitIndexes(300)")
    logger.info("Neo4j constraints and vector index ensured")


# ---------------------------------------------------------------------------
# Manuscript CRUD
# ---------------------------------------------------------------------------


def create_manuscript(body: ManuscriptCreate, manuscript_id: str | None = None) -> str:
    """Write Manuscript, Characters, Locations, and Rules via MERGE. Returns manuscript_id.

    Accepts an optional manuscript_id so that seed.py can use a stable,
    predictable ID for idempotent seeding.
    """
    if manuscript_id is None:
        manuscript_id = uuid.uuid4().hex
    mid = manuscript_id
    drv = init_driver()

    def _tx(tx) -> None:
        tx.run(
            "MERGE (m:Manuscript {uid: $uid}) "
            "SET m.manuscript_id = $mid, m.title = $title, m.premise = $premise",
            uid=f"{mid}:Manuscript:{body.title}",
            mid=mid,
            title=body.title,
            premise=body.premise,
        )
        for char in body.characters:
            loc_uid = f"{mid}:Location:{char.location}"
            tx.run(
                "MERGE (l:Location {uid: $uid}) "
                "SET l.manuscript_id = $mid, l.name = $name",
                uid=loc_uid,
                mid=mid,
                name=char.location,
            )
            char_uid = f"{mid}:Character:{char.name}"
            tx.run(
                "MERGE (c:Character {uid: $uid}) "
                "SET c.manuscript_id = $mid, c.name = $name, "
                "c.status = $status, c.traits = $traits",
                uid=char_uid,
                mid=mid,
                name=char.name,
                status=char.status,
                traits=char.traits,
            )
            tx.run(
                "MATCH (c:Character {uid: $cuid}), (l:Location {uid: $luid}) "
                "MERGE (c)-[:LOCATED_AT]->(l)",
                cuid=char_uid,
                luid=loc_uid,
            )
        for i, rule_text in enumerate(body.rules):
            tx.run(
                "MERGE (r:Rule {uid: $uid}) SET r.manuscript_id = $mid, r.text = $text",
                uid=f"{mid}:Rule:{i}",
                mid=mid,
                text=rule_text,
            )

    with drv.session() as session:
        session.execute_write(_tx)

    return mid


def manuscript_exists(manuscript_id: str) -> bool:
    drv = init_driver()
    result = drv.execute_query(
        "MATCH (m:Manuscript {manuscript_id: $mid}) RETURN count(m) AS cnt",
        mid=manuscript_id,
    )
    return result.records[0]["cnt"] > 0


# ---------------------------------------------------------------------------
# Canon read
# ---------------------------------------------------------------------------


def get_canon(manuscript_id: str) -> dict:
    """Return structured canon facts: characters, rules, and recent events.

    Includes ALL characters regardless of status so the gate knows who is dead.
    Query shape from CLAUDE.md §7.2.
    """
    drv = init_driver()

    # Characters with their most-recent 3 event summaries (per-character)
    char_result = drv.execute_query(
        "MATCH (c:Character {manuscript_id: $mid}) "
        "OPTIONAL MATCH (c)-[:LOCATED_AT]->(loc:Location) "
        "OPTIONAL MATCH (e:Event {manuscript_id: $mid})-[:INVOLVES]->(c) "
        "WITH c, loc, e ORDER BY e.sequence_index DESC "
        "RETURN c.name AS name, c.status AS status, c.traits AS traits, "
        "loc.name AS location, collect(e.summary)[..3] AS recent_events",
        mid=manuscript_id,
    )
    characters = [
        {
            "name": r["name"],
            "status": r["status"],
            "traits": r["traits"],
            "location": r["location"],
            "recent_events": [s for s in (r["recent_events"] or []) if s],
        }
        for r in char_result.records
    ]

    rule_result = drv.execute_query(
        "MATCH (r:Rule {manuscript_id: $mid}) RETURN r.text AS text",
        mid=manuscript_id,
    )
    rules = [r["text"] for r in rule_result.records]

    event_result = drv.execute_query(
        "MATCH (e:Event {manuscript_id: $mid}) "
        "RETURN e.summary AS summary, e.sequence_index AS seq "
        "ORDER BY e.sequence_index ASC LIMIT 10",
        mid=manuscript_id,
    )
    events = [
        {"summary": r["summary"], "sequence_index": r["seq"]}
        for r in event_result.records
    ]

    return {"characters": characters, "rules": rules, "events": events}


# ---------------------------------------------------------------------------
# Tier-1 deterministic gate
# ---------------------------------------------------------------------------


def tier1_check(extraction: ChapterExtraction, canon: dict) -> list[str]:
    """Deterministic canon violation check against pre-fetched canon facts.

    Pure function — no database queries needed; the canon dict already carries
    all character statuses. Returns violation strings (empty list = PASS).

    Checks:
    1. Dead character named in any event.involves list
    2. Dead-in-canon character extracted as alive (resurrection)

    No flashback support in v1: any dead-character involvement is a violation.
    """
    violations: list[str] = []
    canon_status: dict[str, str] = {
        c["name"]: c["status"] for c in canon.get("characters", [])
    }

    for event in extraction.events:
        for name in event.involves:
            if canon_status.get(name) == "dead":
                violations.append(f"Dead character '{name}' is involved in an event")

    for char in extraction.characters:
        if canon_status.get(char.name) == "dead" and char.status == "alive":
            violations.append(
                f"Resurrection: '{char.name}' is dead in canon but extracted as alive"
            )

    return violations


# ---------------------------------------------------------------------------
# Chapter lifecycle
# ---------------------------------------------------------------------------


def next_chapter_number(manuscript_id: str) -> int:
    """Return max existing chapter number + 1, or 1 if none exist."""
    drv = init_driver()
    result = drv.execute_query(
        "OPTIONAL MATCH (ch:Chapter {manuscript_id: $mid}) "
        "RETURN coalesce(max(ch.number), 0) + 1 AS next_n",
        mid=manuscript_id,
    )
    return result.records[0]["next_n"]


def is_chapter_committed(manuscript_id: str, chapter_number: int) -> bool:
    drv = init_driver()
    result = drv.execute_query(
        "MATCH (ch:Chapter {manuscript_id: $mid, number: $n, status: 'COMMITTED'}) "
        "RETURN count(ch) AS cnt",
        mid=manuscript_id,
        n=chapter_number,
    )
    return result.records[0]["cnt"] > 0


def get_chapter(manuscript_id: str, number: int) -> dict | None:
    drv = init_driver()
    result = drv.execute_query(
        "MATCH (ch:Chapter {manuscript_id: $mid, number: $n}) "
        "RETURN ch.number AS number, ch.status AS status, ch.text AS text",
        mid=manuscript_id,
        n=number,
    )
    if not result.records:
        return None
    r = result.records[0]
    return {"number": r["number"], "status": r["status"], "text": r["text"]}


def commit_chapter(
    manuscript_id: str,
    chapter_number: int,
    text: str,
    extraction: ChapterExtraction,
    embedding: list[float] | None,
) -> None:
    """Persist prose, graph updates, and (Phase 4+) the passage embedding in ONE transaction.

    All writes are inside a single execute_write so canon is never half-updated.
    Reads (max sequence_index, previous event uid) happen first inside the same
    transaction before any writes, so they see the committed graph state.
    """
    drv = init_driver()
    mid = manuscript_id
    n = chapter_number

    def _tx(tx) -> None:
        # Names we will MERGE as full Character nodes from the extraction below.
        # Used to detect names that appear only in event.involves.
        declared_char_names = {c.name for c in extraction.characters}

        # ---- READS FIRST (before any writes in this tx) ----
        max_seq_rec = tx.run(
            "OPTIONAL MATCH (e:Event {manuscript_id: $mid}) "
            "RETURN coalesce(max(e.sequence_index), -1) AS max_seq",
            mid=mid,
        ).single()
        max_seq: int = max_seq_rec["max_seq"]

        prev_event_uid: str | None = None
        if max_seq >= 0:
            prev_rec = tx.run(
                "MATCH (e:Event {manuscript_id: $mid, sequence_index: $seq}) "
                "RETURN e.uid AS uid",
                mid=mid,
                seq=max_seq,
            ).single()
            if prev_rec:
                prev_event_uid = prev_rec["uid"]

        # ---- WRITES ----
        chapter_uid = f"{mid}:Chapter:{n}"

        # 1. MERGE Chapter
        tx.run(
            "MERGE (ch:Chapter {uid: $uid}) "
            "SET ch.manuscript_id = $mid, ch.number = $n, "
            "ch.text = $text, ch.status = 'COMMITTED'",
            uid=chapter_uid,
            mid=mid,
            n=n,
            text=text,
        )

        # 2. MERGE Characters (update status from extraction)
        for char in extraction.characters:
            char_uid = f"{mid}:Character:{char.name}"
            tx.run(
                "MERGE (c:Character {uid: $uid}) "
                "SET c.manuscript_id = $mid, c.name = $name, c.status = $status",
                uid=char_uid,
                mid=mid,
                name=char.name,
                status=char.status,
            )
            tx.run(
                "MATCH (c:Character {uid: $cuid}), (ch:Chapter {uid: $chuid}) "
                "MERGE (c)-[:APPEARS_IN]->(ch)",
                cuid=char_uid,
                chuid=chapter_uid,
            )

        # 3. CREATE Events with PRECEDES chain
        seq = max_seq + 1
        local_prev_uid = prev_event_uid
        for event in extraction.events:
            event_uid = f"{mid}:Event:{seq}"
            tx.run(
                "MERGE (e:Event {uid: $uid}) "
                "SET e.manuscript_id = $mid, e.summary = $summary, "
                "e.sequence_index = $seq",
                uid=event_uid,
                mid=mid,
                summary=event.summary,
                seq=seq,
            )
            if local_prev_uid is not None:
                tx.run(
                    "MATCH (prev:Event {uid: $puid}), (curr:Event {uid: $cuid}) "
                    "MERGE (prev)-[:PRECEDES]->(curr)",
                    puid=local_prev_uid,
                    cuid=event_uid,
                )
            for char_name in event.involves:
                if char_name not in declared_char_names:
                    # Hallucinated or briefly-mentioned NPC: create a node so the
                    # INVOLVES edge is preserved. Status defaults to "alive".
                    # ON CREATE SET ensures an existing character's status (e.g.
                    # 'dead') is never overwritten.
                    logger.warning(
                        "Character %r appears in event.involves but not in "
                        "extraction.characters; auto-creating with status='alive'",
                        char_name,
                    )
                    tx.run(
                        "MERGE (c:Character {uid: $uid}) "
                        "ON CREATE SET c.manuscript_id = $mid, c.name = $name, "
                        "c.status = 'alive'",
                        uid=f"{mid}:Character:{char_name}",
                        mid=mid,
                        name=char_name,
                    )
                    declared_char_names.add(char_name)
                tx.run(
                    "MATCH (e:Event {uid: $euid}), (c:Character {uid: $cuid}) "
                    "MERGE (e)-[:INVOLVES]->(c)",
                    euid=event_uid,
                    cuid=f"{mid}:Character:{char_name}",
                )
            tx.run(
                "MATCH (e:Event {uid: $euid}), (ch:Chapter {uid: $chuid}) "
                "MERGE (e)-[:OCCURS_IN]->(ch)",
                euid=event_uid,
                chuid=chapter_uid,
            )
            local_prev_uid = event_uid
            seq += 1

        # 4. Embedding (Phase 4 adds this; skipped when None)
        if embedding is not None:
            tx.run(
                "MATCH (ch:Chapter {uid: $uid}) SET ch.embedding = $emb",
                uid=chapter_uid,
                emb=embedding,
            )

    with drv.session() as session:
        session.execute_write(_tx)


def mark_needs_review(
    manuscript_id: str,
    chapter_number: int,
    last_draft: str,
    last_feedback: list[str],
) -> None:
    """Mark a chapter NEEDS_REVIEW after exhausting all generation retries."""
    drv = init_driver()
    chapter_uid = f"{manuscript_id}:Chapter:{chapter_number}"
    drv.execute_query(
        "MERGE (ch:Chapter {uid: $uid}) "
        "SET ch.manuscript_id = $mid, ch.number = $n, "
        "ch.text = $text, ch.status = 'NEEDS_REVIEW', ch.last_feedback = $feedback",
        uid=chapter_uid,
        mid=manuscript_id,
        n=chapter_number,
        text=last_draft,
        feedback="\n".join(last_feedback),
    )


# ---------------------------------------------------------------------------
# Debug / state
# ---------------------------------------------------------------------------


def get_state(manuscript_id: str) -> dict:
    """Debug snapshot: characters with status/location, last 5 events, rules."""
    drv = init_driver()

    char_result = drv.execute_query(
        "MATCH (c:Character {manuscript_id: $mid}) "
        "OPTIONAL MATCH (c)-[:LOCATED_AT]->(l:Location) "
        "RETURN c.name AS name, c.status AS status, "
        "c.traits AS traits, l.name AS location",
        mid=manuscript_id,
    )
    characters = [
        {
            "name": r["name"],
            "status": r["status"],
            "traits": r["traits"],
            "location": r["location"],
        }
        for r in char_result.records
    ]

    event_result = drv.execute_query(
        "MATCH (e:Event {manuscript_id: $mid}) "
        "RETURN e.summary AS summary, e.sequence_index AS seq "
        "ORDER BY e.sequence_index DESC LIMIT 5",
        mid=manuscript_id,
    )
    events = [
        {"summary": r["summary"], "sequence_index": r["seq"]}
        for r in event_result.records
    ]

    rule_result = drv.execute_query(
        "MATCH (r:Rule {manuscript_id: $mid}) RETURN r.text AS text",
        mid=manuscript_id,
    )
    rules = [r["text"] for r in rule_result.records]

    return {"characters": characters, "events": events, "rules": rules}
