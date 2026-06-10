"""Vector embedding and similarity search.

Phase 2 stub — both functions return empty/None.
Phase 4 wires in fastembed (BAAI/bge-small-en-v1.5, dim=384) and the
Neo4j vector index query with k_overfetch=k*4 post-filter.
"""

from app.config import EMBED_DIM  # imported to assert single-source dimension


def similar_passages(manuscript_id: str, query: str, k: int) -> list[dict]:
    """Return the top-k similar past chapter passages for this manuscript.

    Phase 2 stub: always returns an empty list so the pipeline proceeds without
    vector retrieval. Phase 4 replaces this with a fastembed query + Neo4j
    vector index call.
    """
    return []


def embed_one(text: str) -> list[float] | None:
    """Embed a single text string and return a vector of length EMBED_DIM (384).

    Phase 2 stub: returns None so commit_chapter skips writing the embedding.
    Phase 4 replaces this with fastembed inference (BAAI/bge-small-en-v1.5).
    """
    _ = EMBED_DIM  # keep the import live; dimension enforced in Phase 4
    return None
