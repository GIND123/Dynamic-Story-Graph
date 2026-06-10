import logging
import uuid

from neo4j import Driver, GraphDatabase

from app.config import settings
from app.models import ManuscriptCreate

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
    logger.info("Neo4j constraints ensured")


def create_manuscript(body: ManuscriptCreate) -> str:
    """Write Manuscript, Characters, Locations, and Rules via MERGE. Returns manuscript_id."""
    manuscript_id = uuid.uuid4().hex
    drv = init_driver()

    def _tx(tx, mid: str) -> None:
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
        session.execute_write(_tx, manuscript_id)

    return manuscript_id


def next_chapter_number(manuscript_id: str) -> int:
    """Return max existing chapter number + 1, or 1 if none exist."""
    drv = init_driver()
    result = drv.execute_query(
        "OPTIONAL MATCH (ch:Chapter {manuscript_id: $mid}) "
        "RETURN coalesce(max(ch.number), 0) + 1 AS next_n",
        mid=manuscript_id,
    )
    return result.records[0]["next_n"]


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
