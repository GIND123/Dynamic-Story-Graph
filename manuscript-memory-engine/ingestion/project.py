"""Project a DatasetIR into the existing Neo4j graph + vector index.

Eval-only: the Manuscript node is tagged source='eval' and namespaced
(pdnc:{folder} / litbank:{gid}) so the generation pipeline can never select it
(see graph.is_eval_manuscript). All writes go through the existing MERGE-on-uid
patterns and the Part-A `graph.merge_relation` edge helper. Passage embeddings
are written onto Chapter nodes so they are indexed by the EXISTING passage_index
(no parallel index, same EMBED_DIM).
"""

from __future__ import annotations

import logging

from app import graph, vectors
from app.models import ExtractedRelation, RelationType
from ingestion.base import DatasetIR, DatasetLoader

logger = logging.getLogger(__name__)

# Entity IR type -> graph node label.
_TYPE_TO_LABEL = {
    "CHARACTER": "Character",
    "LOCATION": "Location",
    "ORGANIZATION": "Organization",
    "OBJECT": "Object",
}


def project_to_graph(loader: DatasetLoader, manuscript_id: str | None = None) -> dict:
    """Load `loader`'s IR and write it into the graph as eval data.

    Returns a small summary dict (counts). Idempotent: every node/edge is
    MERGEd on uid, so re-projection creates no duplicates.
    """
    ir = loader.load()
    mid = manuscript_id or loader.namespace()

    # name -> node label, so event INVOLVES edges point at the right node type.
    label_by_name = {e.name: _TYPE_TO_LABEL[e.type] for e in ir.entities}

    drv = graph.init_driver()
    with drv.session() as session:
        session.execute_write(_write_structure, mid, ir, label_by_name)

    n_passages = _write_passages(drv, mid, ir)

    summary = {
        "manuscript_id": mid,
        "entities": len(ir.entities),
        "events": len(ir.events),
        "quotations": len(ir.quotations),
        "passages": n_passages,
    }
    logger.info("Projected eval dataset: %s", summary)
    return summary


def _write_structure(tx, mid: str, ir: DatasetIR, label_by_name: dict) -> None:
    """All structural writes (nodes + typed edges) in one transaction."""
    # 1. Manuscript node, tagged source='eval' (the generation guard reads this).
    tx.run(
        "MERGE (m:Manuscript {uid: $uid}) "
        "SET m.manuscript_id = $mid, m.title = $title, m.source = 'eval'",
        uid=f"{mid}:Manuscript:{mid}",
        mid=mid,
        title=mid,
    )

    # 2. Entity nodes.
    for entity in ir.entities:
        label = _TYPE_TO_LABEL[entity.type]
        tx.run(
            f"MERGE (n:{label} {{uid: $uid}}) "
            "SET n.manuscript_id = $mid, n.name = $name, n.source = 'eval'",
            uid=f"{mid}:{label}:{entity.name}",
            mid=mid,
            name=entity.name,
        )
        # 3. IDENTIFIED_AS edges (characters only, per schema CHARACTER->ALIAS).
        if entity.type == "CHARACTER":
            for alias in entity.aliases:
                graph.merge_relation(
                    tx,
                    mid,
                    ExtractedRelation(
                        type=RelationType.IDENTIFIED_AS,
                        source=entity.name,
                        target=alias,
                        kind="alias",
                    ),
                    source_ref=f"alias:{entity.name}",
                    seq_introduced=0,
                    source="eval",
                )

    # 4. Narrative events -> Event node + INVOLVES edges to co-occurring entities.
    for event in ir.events:
        event_uid = f"{mid}:Event:{event.order}"
        tx.run(
            "MERGE (e:Event {uid: $uid}) "
            "SET e.manuscript_id = $mid, e.summary = $summary, "
            "e.sequence_index = $seq, e.source = 'eval'",
            uid=event_uid,
            mid=mid,
            summary=event.summary,
            seq=event.order,
        )
        for name in event.participant_names:
            graph.merge_relation(
                tx,
                mid,
                ExtractedRelation(
                    type=RelationType.INVOLVES,
                    source=str(event.order),
                    target=name,
                    role="patient",
                ),
                source_ref=f"event:{event.order}",
                seq_introduced=event.order,
                source="eval",
                target_label=label_by_name.get(name, "Character"),
            )

    # 5. Quotations -> dialogue-act Event nodes (quote_type kept as a property),
    #    speaker INVOLVES (agent), addressees INVOLVES (patient).
    base_seq = len(ir.events)
    for i, quote in enumerate(ir.quotations):
        seq = base_seq + i
        event_uid = f"{mid}:Event:quote:{quote.ref}"
        tx.run(
            "MERGE (e:Event {uid: $uid}) "
            "SET e.manuscript_id = $mid, e.summary = $summary, "
            "e.sequence_index = $seq, e.quote_type = $qt, "
            "e.is_dialogue = true, e.source = 'eval'",
            uid=event_uid,
            mid=mid,
            summary=quote.text[:240],
            seq=seq,
            qt=quote.quote_type,
        )
        graph.merge_relation(
            tx,
            mid,
            ExtractedRelation(
                type=RelationType.INVOLVES,
                source=f"quote:{quote.ref}",
                target=quote.speaker_name,
                role="agent",
            ),
            source_ref=quote.ref,
            seq_introduced=seq,
            source="eval",
            target_label="Character",
        )
        for addressee in quote.addressee_names:
            graph.merge_relation(
                tx,
                mid,
                ExtractedRelation(
                    type=RelationType.INVOLVES,
                    source=f"quote:{quote.ref}",
                    target=addressee,
                    role="patient",
                ),
                source_ref=quote.ref,
                seq_introduced=seq,
                source="eval",
                target_label="Character",
            )


def _write_passages(drv, mid: str, ir: DatasetIR) -> int:
    """Embed each segment and write it as a Chapter node (indexed by passage_index).

    Embeddings are computed outside the transaction (CPU-bound); the Chapter
    writes then go in one execute_write. status='COMMITTED' so the existing
    similar_passages query (which filters COMMITTED) can retrieve eval passages
    for the eval manuscript_id only.
    """
    if not ir.segments:
        return 0

    embedded = [
        (seg.index, seg.text, vectors.embed_one(seg.text)) for seg in ir.segments
    ]

    def _tx(tx) -> None:
        for index, text, embedding in embedded:
            tx.run(
                "MERGE (ch:Chapter {uid: $uid}) "
                "SET ch.manuscript_id = $mid, ch.number = $n, ch.text = $text, "
                "ch.status = 'COMMITTED', ch.source = 'eval', ch.embedding = $emb",
                uid=f"{mid}:Chapter:{index}",
                mid=mid,
                n=index,
                text=text,
                emb=embedding,
            )

    with drv.session() as session:
        session.execute_write(_tx)
    return len(embedded)
