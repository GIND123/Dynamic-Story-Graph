"""Tests proving that EMBED_DIM is the single source of truth for vector dimension.

The acceptance criterion: a test proves the dimension constant is used in
both the index DDL (graph.ensure_constraints) and the embedder (vectors.embed_one).

embed_one downloads BAAI/bge-small-en-v1.5 on first run (~23 MB, cached after).
"""

import inspect


class TestEmbedDimSingleSourceOfTruth:
    def test_embedder_output_matches_embed_dim(self) -> None:
        """embed_one returns exactly EMBED_DIM floats — both sides use the same constant."""
        from app.config import EMBED_DIM
        from app.vectors import embed_one

        vec = embed_one("Duskwall stands on the edge of an ancient dark")
        assert isinstance(vec, list)
        assert len(vec) == EMBED_DIM, f"Expected {EMBED_DIM}, got {len(vec)}"
        assert all(isinstance(v, float) for v in vec)

    def test_graph_ddl_references_embed_dim(self) -> None:
        """ensure_constraints references the EMBED_DIM constant, not a hardcoded literal.

        inspect.getsource returns the raw source text; the f-string expression
        {EMBED_DIM} appears there (not the evaluated value 384), proving the
        DDL uses the single source of truth rather than a magic number.
        """
        import app.graph as g

        source = inspect.getsource(g.ensure_constraints)
        assert "EMBED_DIM" in source, (
            "ensure_constraints must reference EMBED_DIM, not a hardcoded dimension"
        )

    def test_embed_one_is_deterministic(self) -> None:
        """Same input produces the same vector (ONNX model is deterministic)."""
        from app.vectors import embed_one

        text = "Mara moved through the docks at night"
        v1 = embed_one(text)
        v2 = embed_one(text)
        assert v1 == v2

    def test_different_texts_produce_different_vectors(self) -> None:
        """Semantically different texts should not produce identical vectors."""
        from app.vectors import embed_one

        v1 = embed_one("Brann is a loyal guard captain")
        v2 = embed_one("The Ashen Crown lies forgotten in the vaults")
        assert v1 != v2


class TestBudgetTruncation:
    """Re-asserts the retrieval budget logic with Phase-4 context.

    The core budget tests live in test_retrieval.py; this class verifies
    that retrieved passages respect the budget when assembled into context.
    """

    def test_retrieved_passages_respect_token_budget(self) -> None:
        from app.retrieval import assemble_context

        long_passage = ("A" * 100 + ". ") * 100  # ~12 000 chars
        passages = [{"text": long_passage, "chapter": 1, "score": 0.9}]
        canon = {
            "characters": [
                {
                    "name": "Mara",
                    "status": "alive",
                    "traits": "smuggler",
                    "location": "docks",
                    "recent_events": [],
                }
            ],
            "rules": [],
            "events": [],
        }

        result = assemble_context(canon, passages)
        passage_part = result.split("=== SIMILAR PASSAGES ===")[1]
        # TOKEN_BUDGET_PASSAGES default = 2000 → ~8000 chars; allow header overhead
        assert len(passage_part) < 9500
