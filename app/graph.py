import logging
import uuid

from neo4j import Driver, GraphDatabase

from app.config import EMBED_DIM, settings
from app.models import (
    ChapterExtraction,
    ExtractedRelation,
    ManuscriptCreate,
    RelationType,
)

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
    labels = [
        "Manuscript",
        "Character",
        "Location",
        "Event",
        "Chapter",
        "Rule",
        # Additive node labels for the relationship schema model (Part A).
        "Organization",
        "Object",
        "Attribute",
        "Alias",
        "Fact",
    ]
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

    relations = _get_active_relations(drv, manuscript_id)

    # 'relations' is an additive key (existing keys unchanged) consumed by the
    # pure tier1_check. Only ACTIVE edges (seq_invalidated IS NULL) are surfaced.
    return {
        "characters": characters,
        "rules": rules,
        "events": events,
        "relations": relations,
    }


def _get_active_relations(drv, manuscript_id: str) -> dict:
    """Read the currently-active typed relations the Tier-1 checks need.

    "Active" = the edge's seq_invalidated IS NULL. Returns a dict of lists keyed
    by relation kind; shape is consumed by tier1_check (pure). One query per kind
    keeps each MATCH simple and the result easy to assert in tests.
    """
    mid = manuscript_id

    located_at = drv.execute_query(
        "MATCH (c:Character {manuscript_id: $mid})-[r:LOCATED_AT]->(loc:Location) "
        "WHERE r.seq_invalidated IS NULL "
        "RETURN c.name AS character, loc.name AS location",
        mid=mid,
    ).records
    relates_to = drv.execute_query(
        "MATCH (a:Character {manuscript_id: $mid})-[r:RELATES_TO]->(b:Character) "
        "WHERE r.seq_invalidated IS NULL "
        "RETURN a.name AS source, b.name AS target, r.stance AS stance",
        mid=mid,
    ).records
    family_of = drv.execute_query(
        "MATCH (a:Character {manuscript_id: $mid})-[r:FAMILY_OF]->(b:Character) "
        "WHERE r.seq_invalidated IS NULL "
        "RETURN a.name AS source, b.name AS target, r.subtype AS subtype",
        mid=mid,
    ).records
    possesses = drv.execute_query(
        "MATCH (c:Character {manuscript_id: $mid})-[r:POSSESSES]->(o:Object) "
        "WHERE r.seq_invalidated IS NULL "
        "RETURN c.name AS character, o.name AS object",
        mid=mid,
    ).records
    identified_as = drv.execute_query(
        "MATCH (c:Character {manuscript_id: $mid})-[r:IDENTIFIED_AS]->(al:Alias) "
        "WHERE r.seq_invalidated IS NULL "
        "RETURN c.name AS character, al.name AS alias",
        mid=mid,
    ).records
    has_trait = drv.execute_query(
        "MATCH (c:Character {manuscript_id: $mid})-[r:HAS_TRAIT]->(t:Attribute) "
        "WHERE r.seq_invalidated IS NULL "
        "RETURN c.name AS character, t.name AS trait, r.category AS category, "
        "r.trait_key AS trait_key, r.trait_value AS trait_value",
        mid=mid,
    ).records

    return {
        "located_at": [
            {"character": r["character"], "location": r["location"]} for r in located_at
        ],
        "relates_to": [
            {"source": r["source"], "target": r["target"], "stance": r["stance"]}
            for r in relates_to
        ],
        "family_of": [
            {"source": r["source"], "target": r["target"], "subtype": r["subtype"]}
            for r in family_of
        ],
        "possesses": [
            {"character": r["character"], "object": r["object"]} for r in possesses
        ],
        "identified_as": [
            {"character": r["character"], "alias": r["alias"]} for r in identified_as
        ],
        "has_trait": [
            {
                "character": r["character"],
                "trait": r["trait"],
                "category": r["category"],
                "trait_key": r["trait_key"],
                "trait_value": r["trait_value"],
            }
            for r in has_trait
        ],
    }


# ---------------------------------------------------------------------------
# Tier-1 deterministic gate
# ---------------------------------------------------------------------------


def tier1_check(extraction: ChapterExtraction, canon: dict) -> list[str]:
    """Deterministic canon violation check against pre-fetched canon facts.

    Pure function — NO database access. Reads only the incoming extraction and
    the canon dict (built by get_canon). Returns violation strings (empty = PASS).
    A returned string routes the chapter to the existing regenerate→NEEDS_REVIEW
    path; nothing here hard-rejects.

    Checks:
    1. Dead character named in any event.involves list
    2. Dead-in-canon character extracted as alive (resurrection)
    3. Contradictory RELATES_TO stance (ally vs enemy for the same pair)
    4. Two-places-at-once (one character at >1 location within this chapter)
    5. Kinship impossibility (self-loop, parent/child inversion, spouse-of-dead)
    6. Physical-trait contradiction (same character, same trait key, new value)
    7. Identity collision (alias already bound to a different character)

    No flashback support in v1: any dead-character involvement is a violation.
    """
    violations: list[str] = []
    canon_status: dict[str, str] = {
        c["name"]: c["status"] for c in canon.get("characters", [])
    }
    rels = canon.get("relations", {})

    # 1 & 2 — existing dead/resurrection checks (unchanged)
    for event in extraction.events:
        for name in event.involves:
            if canon_status.get(name) == "dead":
                violations.append(f"Dead character '{name}' is involved in an event")

    for char in extraction.characters:
        if canon_status.get(char.name) == "dead" and char.status == "alive":
            violations.append(
                f"Resurrection: '{char.name}' is dead in canon but extracted as alive"
            )

    # 3 — contradictory stance: asserted ally/enemy that canon has as the opposite
    canon_stance: dict[frozenset, set[str]] = {}
    for rel in rels.get("relates_to", []):
        stance = (rel.get("stance") or "").lower()
        if stance in ("ally", "enemy"):
            pair = frozenset({rel["source"], rel["target"]})
            canon_stance.setdefault(pair, set()).add(stance)
    opposite = {"ally": "enemy", "enemy": "ally"}
    for r in extraction.relations:
        if r.type == RelationType.RELATES_TO:
            stance = (r.stance or "").lower()
            if stance in opposite:
                pair = frozenset({r.source, r.target})
                if opposite[stance] in canon_stance.get(pair, set()):
                    violations.append(
                        f"Stance contradiction: '{r.source}' and '{r.target}' "
                        f"asserted as {stance} but canon has them as {opposite[stance]}"
                    )

    # 4 — two-places-at-once: same character asserted at >1 location THIS chapter.
    # Cross-chapter movement is intentionally NOT flagged (the pipeline does not
    # yet invalidate prior LOCATED_AT edges); this keeps the rule conservative.
    asserted_loc: dict[str, set[str]] = {}
    for r in extraction.relations:
        if r.type == RelationType.LOCATED_AT:
            asserted_loc.setdefault(r.source, set()).add(r.target)
    for character, locs in asserted_loc.items():
        if len(locs) > 1:
            violations.append(
                f"Two places at once: '{character}' is asserted at multiple "
                f"locations in one chapter: {sorted(locs)}"
            )

    # 5 — kinship impossibility
    canon_family: set[tuple[str, str, str]] = {
        (f["source"], f["target"], (f.get("subtype") or "").lower())
        for f in rels.get("family_of", [])
    }
    for r in extraction.relations:
        if r.type != RelationType.FAMILY_OF:
            continue
        sub = (r.subtype or "").lower()
        if r.source == r.target:
            violations.append(f"Kinship self-loop: '{r.source}' FAMILY_OF itself")
            continue
        if sub in ("parent", "child"):
            inverse = "child" if sub == "parent" else "parent"
            if (r.target, r.source, sub) in canon_family:
                violations.append(
                    f"Kinship contradiction: '{r.source}' {sub} of '{r.target}' "
                    f"but canon has '{r.target}' {sub} of '{r.source}'"
                )
            if (r.source, r.target, inverse) in canon_family:
                violations.append(
                    f"Kinship contradiction: '{r.source}' asserted {sub} of "
                    f"'{r.target}' but canon has them as {inverse}"
                )
        if sub == "spouse" and (
            canon_status.get(r.target) == "dead" or canon_status.get(r.source) == "dead"
        ):
            violations.append(
                f"Marriage to dead character: spouse link between '{r.source}' "
                f"and '{r.target}' where one is dead in canon (flag for review)"
            )

    # 6 — physical-trait contradiction (structured key/value; conservatism is
    # structural, not string-parsed). Build canon (character, trait_key) ->
    # trait_value only for PHYSICAL traits that carry BOTH a key and a value.
    # Physical scope matters: physical attributes (eye color) don't change,
    # while personality/skill can develop, so same-key/diff-value is only a
    # contradiction for physical traits.
    canon_phys: dict[tuple[str, str], str] = {}
    for t in rels.get("has_trait", []):
        if (t.get("category") or "").lower() == "physical":
            key = (t.get("trait_key") or "").strip().lower()
            value = (t.get("trait_value") or "").strip().lower()
            if key and value:
                canon_phys[(t["character"], key)] = value
    for r in extraction.relations:
        if r.type != RelationType.HAS_TRAIT or (r.category or "").lower() != "physical":
            continue
        key = (r.trait_key or "").strip().lower()
        value = (r.trait_value or "").strip().lower()
        if not key or not value:
            # No stable key/value to compare on. Do NOT decide deterministically;
            # leave it for the Tier-2 judge (no Tier-1 violation emitted here).
            continue
        existing = canon_phys.get((r.source, key))
        if existing is not None and existing != value:
            # Exact key match, exact value inequality — no substring/fuzzy logic.
            violations.append(
                f"Trait contradiction: '{r.source}' {key} asserted as "
                f"'{value}' but canon has '{existing}'"
            )

    # 7 — identity collision: alias canon already binds to a different character
    canon_alias_owner: dict[str, str] = {
        i["alias"]: i["character"] for i in rels.get("identified_as", [])
    }
    for r in extraction.relations:
        if r.type == RelationType.IDENTIFIED_AS:
            owner = canon_alias_owner.get(r.target)
            if owner is not None and owner != r.source:
                violations.append(
                    f"Identity collision: alias '{r.target}' asserted for "
                    f"'{r.source}' but canon binds it to '{owner}'"
                )

    return violations


# ---------------------------------------------------------------------------
# Relationship schema model (Part A) — typed edge MERGE helper
# ---------------------------------------------------------------------------

# Node labels that may be an edge endpoint. Used to validate any label that is
# interpolated into a Cypher pattern (Cypher cannot parameterize labels), so an
# arbitrary string can never reach the query body.
_NODE_LABELS: frozenset[str] = frozenset(
    {
        "Manuscript",
        "Character",
        "Location",
        "Event",
        "Chapter",
        "Rule",
        "Organization",
        "Object",
        "Attribute",
        "Alias",
        "Fact",
    }
)

# Canonical (source_label, target_label) per relation type. Polymorphic targets
# (CONTROLS, KNOWS_ABOUT) default to one label; callers pass overrides for the
# other valid endpoints. See CLAUDE.md §7.2 and the task's edge endpoint table.
_RELATION_ENDPOINTS: dict[RelationType, tuple[str, str]] = {
    RelationType.LOCATED_AT: ("Character", "Location"),
    RelationType.PART_OF: ("Location", "Location"),
    RelationType.POSSESSES: ("Character", "Object"),
    RelationType.CONTROLS: ("Character", "Object"),
    RelationType.FAMILY_OF: ("Character", "Character"),
    RelationType.RELATES_TO: ("Character", "Character"),
    RelationType.MEMBER_OF: ("Character", "Organization"),
    RelationType.IDENTIFIED_AS: ("Character", "Alias"),
    RelationType.HAS_TRAIT: ("Character", "Attribute"),
    RelationType.OWES: ("Character", "Character"),
    RelationType.KNOWS_ABOUT: ("Character", "Fact"),
    RelationType.INVOLVES: ("Event", "Character"),
    RelationType.CAUSED: ("Character", "Event"),
    # Thin direct dialogue edge: a single-hop speaker->addressee view alongside
    # the EVENT+INVOLVES dialogue model (additive, neither replaces the other).
    RelationType.SPEAKS_TO: ("Character", "Character"),
}

# Discriminator fields carried from ExtractedRelation onto the edge.
_RELATION_DISCRIMINATORS = (
    "subtype",
    "stance",
    "romantic",
    "category",
    "kind",
    "role",
    "quote_type",
    "trait_key",
    "trait_value",
    "confidence",
)


def merge_relation(
    tx,
    manuscript_id: str,
    relation: ExtractedRelation,
    *,
    source_ref: str,
    seq_introduced: int,
    source: str = "eval",
    seq_invalidated: int | None = None,
    source_label: str | None = None,
    target_label: str | None = None,
) -> None:
    """MERGE a typed edge (and its endpoint nodes) from an ExtractedRelation.

    Runs inside a caller-provided transaction (`tx.run`), matching the style of
    the existing INVOLVES handling in commit_chapter. Endpoint nodes are created
    by uid if absent (Attribute/Alias/Fact/Object/Organization included).

    The edge type and node labels are interpolated into the query because Cypher
    cannot parameterize them — but they are validated against the closed
    RelationType enum and the _NODE_LABELS allowlist first, so an arbitrary
    string can never reach the query body. Every data value is a $parameter.

    Every edge carries the universal property bundle:
      source ('gen'|'eval'|'seed'|'dataset'), source_ref, seq_introduced,
      seq_invalidated (null = currently active), plus any present discriminators.
    ON CREATE SET is used so re-projection never overwrites an existing edge.
    """
    rel_type = RelationType(relation.type)  # raises on anything outside the 13
    default_src, default_tgt = _RELATION_ENDPOINTS[rel_type]
    s_label = source_label or default_src
    t_label = target_label or default_tgt
    if s_label not in _NODE_LABELS or t_label not in _NODE_LABELS:
        raise ValueError(f"Unknown node label: {s_label!r} or {t_label!r}")

    mid = manuscript_id
    source_uid = f"{mid}:{s_label}:{relation.source}"
    # HAS_TRAIT: the Attribute node is the trait DIMENSION, keyed by trait_key
    # (e.g. one 'eye_color' node), so per-value strings don't fragment it into
    # many nodes. The specific value rides on the edge as trait_value. Other
    # relations (and legacy traits without a trait_key) key the node by target.
    if rel_type == RelationType.HAS_TRAIT and relation.trait_key:
        target_name = relation.trait_key
    else:
        target_name = relation.target
    target_uid = f"{mid}:{t_label}:{target_name}"

    # Universal property bundle + present discriminators; Neo4j cannot store
    # null, so None values are dropped (absent == null on read-back). A null
    # seq_invalidated therefore reads back as null = "currently active".
    bundle: dict = {
        "source": source,
        "source_ref": source_ref,
        "seq_introduced": seq_introduced,
        "seq_invalidated": seq_invalidated,
    }
    for field in _RELATION_DISCRIMINATORS:
        value = getattr(relation, field)
        if value is not None:
            bundle[field] = value
    bundle = {k: v for k, v in bundle.items() if v is not None}

    # rel_type.value and the labels are validated constants (closed sets above).
    query = (
        f"MERGE (s:{s_label} {{uid: $suid}}) "
        "ON CREATE SET s.manuscript_id = $mid, s.name = $sname "
        f"MERGE (t:{t_label} {{uid: $tuid}}) "
        "ON CREATE SET t.manuscript_id = $mid, t.name = $tname "
        f"MERGE (s)-[r:{rel_type.value}]->(t) "
        "ON CREATE SET r += $bundle"
    )
    tx.run(
        query,
        suid=source_uid,
        tuid=target_uid,
        mid=mid,
        sname=relation.source,
        tname=target_name,
        bundle=bundle,
    )


# ---------------------------------------------------------------------------
# Eval-data guard (Part B) — keep eval manuscripts out of generation
# ---------------------------------------------------------------------------


def is_eval_manuscript(manuscript_id: str) -> bool:
    """True if the manuscript was ingested for evaluation (source='eval').

    The generation pipeline must never write chapters into eval data, so the
    chapter-request entrypoint checks this before enqueuing a job.
    """
    drv = init_driver()
    result = drv.execute_query(
        "MATCH (m:Manuscript {manuscript_id: $mid}) "
        "RETURN coalesce(m.source, 'gen') = 'eval' AS is_eval",
        mid=manuscript_id,
    )
    if not result.records:
        return False
    return bool(result.records[0]["is_eval"])


# ---------------------------------------------------------------------------
# Chapter lifecycle
# ---------------------------------------------------------------------------


def next_chapter_number(manuscript_id: str) -> int:
    """Return max COMMITTED chapter number + 1, or 1 if none committed.

    Filters to status='COMMITTED' so a NEEDS_REVIEW chapter is retried on the
    next POST (its number is returned again) instead of being permanently
    skipped.
    """
    drv = init_driver()
    result = drv.execute_query(
        "OPTIONAL MATCH (ch:Chapter {manuscript_id: $mid, status: 'COMMITTED'}) "
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

        # 1. MERGE Chapter (REMOVE clears any stale NEEDS_REVIEW feedback on retry)
        tx.run(
            "MERGE (ch:Chapter {uid: $uid}) "
            "SET ch.manuscript_id = $mid, ch.number = $n, "
            "ch.text = $text, ch.status = 'COMMITTED' "
            "REMOVE ch.last_feedback",
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

        # 3b. Typed relations from the extraction, written with source='gen'
        # in this SAME transaction. merge_relation defaults each endpoint label
        # per relation type (HAS_TRAIT->Attribute, IDENTIFIED_AS->Alias,
        # MEMBER_OF->Organization, ...) and uses ON CREATE SET, so re-commit is
        # idempotent and never clobbers an existing edge or a character's status.
        for relation in extraction.relations:
            merge_relation(
                tx,
                mid,
                relation,
                source_ref=chapter_uid,
                seq_introduced=n,
                source="gen",
            )

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
