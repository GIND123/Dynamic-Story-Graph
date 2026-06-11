"""Vector embedding and similarity search.

Uses fastembed (ONNX, CPU) with BAAI/bge-small-en-v1.5 (dim=384).
The model is downloaded on first use; call warmup() at worker startup to
avoid the cold-start penalty on the first task (CLAUDE.md §13.18).

Vector similarity queries go through the Neo4j passage_index. The index
cannot pre-filter by manuscript_id, so we over-fetch k*4 candidates and
filter after YIELD (see DECISIONS.md).
"""

import logging

from fastembed import TextEmbedding

from app.config import EMBED_DIM

logger = logging.getLogger(__name__)

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def warmup() -> None:
    """Download model weights and force ONNX compilation with a trivial inference.

    Must be called at worker process init, not inside a task, so the first
    real chapter generation is not penalised by a 10-30s model load.
    """
    model = _get_model()
    list(model.embed(["warmup"]))
    logger.info("fastembed warmed up (model=BAAI/bge-small-en-v1.5, dim=%d)", EMBED_DIM)


def embed_one(text: str) -> list[float]:
    """Embed a single text string. Returns a list of EMBED_DIM floats."""
    model = _get_model()
    vectors = list(model.embed([text]))
    result = vectors[0].tolist()
    if len(result) != EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: got {len(result)}, expected {EMBED_DIM}"
        )
    return result


def similar_passages(manuscript_id: str, query: str, k: int) -> list[dict]:
    """Return the k most similar COMMITTED chapters from this manuscript.

    Over-fetches k*4 to work around the vector index's inability to pre-filter
    by manuscript_id. Post-filters in the WHERE clause after YIELD.
    """
    from app.graph import init_driver  # deferred to avoid loading Neo4j driver in API

    embedding = embed_one(query)
    k_overfetch = k * 4

    drv = init_driver()
    result = drv.execute_query(
        "CALL db.index.vector.queryNodes('passage_index', $k_overfetch, $embedding) "
        "YIELD node, score "
        "WHERE node.manuscript_id = $manuscript_id AND node.status = 'COMMITTED' "
        "RETURN node.number AS chapter, node.text AS text, score "
        "ORDER BY score DESC LIMIT $k",
        k_overfetch=k_overfetch,
        embedding=embedding,
        manuscript_id=manuscript_id,
        k=k,
    )
    passages = [
        {"chapter": r["chapter"], "text": r["text"], "score": r["score"]}
        for r in result.records
    ]
    if passages:
        logger.info(
            "Retrieved %d similar passages for manuscript %s (top score: %.3f)",
            len(passages),
            manuscript_id,
            passages[0]["score"],
        )
    return passages
