"""Offline tests for the GPT4-Books name-cloze metric (no network, no LLM).

Verifies: gold is read from column 3, exact-match accuracy with symmetric
honorific stripping, the wrong-but-valid-character = entity-drift bucket, and
that a raising predictor is counted as an abstain (never crashes).
"""

from pathlib import Path

from evals.metrics import gpt4_books_cloze as gc


def _write_book(tmp_path: Path, book_id: str, rows: list[tuple[str, str, str]]) -> Path:
    """rows = (chatgpt_pred, gold, passage). Writes the 4-col TSV layout."""
    root = tmp_path
    d = root / "model_output" / "chatgpt_results"
    d.mkdir(parents=True)
    lines = [f"<name>{p}</name>\t{p}\t{g}\t{passage}" for (p, g, passage) in rows]
    (d / f"{book_id}.txt").write_text("\n".join(lines), encoding="utf-8")
    return root


def test_gold_is_column_three_and_buckets(tmp_path):
    root = _write_book(
        tmp_path,
        "book",
        [
            ("Jane", "Elizabeth", "and [MASK] would not quit her"),  # gold=Elizabeth
            ("x", "Bob", "[MASK] drew his sword"),
            ("x", "Carol", "[MASK] lit the lamp"),
            ("x", "Mr. Darcy", "[MASK] bowed stiffly"),
        ],
    )
    # Alias-normalized predictions: correct / drift / abstain / honorific-match.
    answers = {
        "and [MASK] would not quit her": "Elizabeth",  # correct
        "[MASK] drew his sword": "Elizabeth",  # wrong but valid -> drift
        "[MASK] lit the lamp": "",  # abstain
        "[MASK] bowed stiffly": "Darcy",  # matches "Mr. Darcy"
    }

    def predict(passage, book_id):
        # find which passage this is
        for p, a in answers.items():
            if p in passage:
                return a or None
        return None

    rep = gc.run_book_cloze("book", predict=predict, root=root, max_rows=100, seed=0)
    assert rep.n == 4
    assert rep.correct == 2  # Elizabeth + (Darcy == Mr. Darcy)
    assert rep.entity_drift == 1  # predicted Elizabeth for Bob
    assert rep.abstain == 1  # Carol
    assert rep.miss == 0
    assert rep.accuracy == 0.5
    assert rep.entity_drift_rate == 0.25


def test_predictor_exception_is_abstain(tmp_path):
    root = _write_book(tmp_path, "book", [("x", "Bob", "[MASK] ran")])

    def boom(passage, book_id):
        raise RuntimeError("model dropped connection")

    rep = gc.run_book_cloze("book", predict=boom, root=root, max_rows=100, seed=0)
    assert rep.abstain == 1
    assert rep.accuracy == 0.0


def test_chatgpt_reference_column(tmp_path):
    # col2 (ChatGPT pred) == gold on row 1 only -> reference accuracy 0.5
    root = _write_book(
        tmp_path,
        "book",
        [("Bob", "Bob", "[MASK] a"), ("Jane", "Alice", "[MASK] b")],
    )
    rep = gc.run_book_cloze(
        "book", predict=lambda p, b: None, root=root, max_rows=100, seed=0
    )
    assert rep.chatgpt_reference_accuracy == 0.5
    assert rep.chatgpt_correct == 1


def test_deterministic_sampling(tmp_path):
    rows = [("x", f"N{i}", f"[MASK] {i}") for i in range(20)]
    root = _write_book(tmp_path, "book", rows)
    a = gc.run_book_cloze(
        "book", predict=lambda p, b: None, root=root, max_rows=5, seed=7
    )
    b = gc.run_book_cloze(
        "book", predict=lambda p, b: None, root=root, max_rows=5, seed=7
    )
    assert a.sample_indices == b.sample_indices
    assert len(a.sample_indices) == 5
