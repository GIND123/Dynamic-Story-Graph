import importlib.util
import json
from pathlib import Path

import pytest

from gnsm.training import evolvtrip_adapter as adapter

HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_parse_triples_strips_the_target_character_suffix() -> None:
    record = {
        "triples": {
            "Target Character": [
                "(King Lear, BelievesAboutCordelia, she will outshine her sisters)",
                "(King Lear, FeelsTowardsCordelia, rising anger)",
                "(King Lear, IntendsTo, disown Cordelia)",
            ]
        }
    }
    parsed = adapter.parse_triples(record)
    assert [t.base_relation for t in parsed] == ["BelievesAbout", "FeelsTowards", "IntendsTo"]
    assert [t.dimension for t in parsed] == ["belief", "emotion", "intention"]
    assert parsed[0].obj == "she will outshine her sisters"


def test_parse_triples_unparsable_strings_are_skipped() -> None:
    record = {"triples": {"k": ["not a triple at all"]}}
    assert adapter.parse_triples(record) == []


def test_parse_triples_unknown_relation_prefix_falls_back_to_other() -> None:
    record = {"triples": {"k": ["(A, SomethingElse, B)"]}}
    parsed = adapter.parse_triples(record)
    assert parsed[0].base_relation == "other"
    assert parsed[0].dimension == "other"


def _write_fixture(tmp_path: Path) -> Path:
    records = [
        {
            "book_name": "Test Book",
            "character": "Alice",
            "plot_index": 1,
            "plot_summary": "Alice arrives.",
            "scenario": "Alice enters the room.",
            "triples": {"k": ["(Alice, BelievesAbout, the door is locked)"]},
        },
        {
            "book_name": "Test Book",
            "character": "Alice",
            "plot_index": 2,
            "plot_summary": "Alice leaves.",
            "scenario": "Alice walks out.",
            "triples": {
                "k": [
                    "(Alice, BelievesAbout, the door is locked)",
                    "(Alice, FeelsTowards, relief)",
                ]
            },
        },
        {
            # different character: must not pair with Alice's records
            "book_name": "Test Book",
            "character": "Bob",
            "plot_index": 1,
            "plot_summary": "Bob waits.",
            "scenario": "Bob stands still.",
            "triples": {},
        },
    ]
    path = tmp_path / "all_books_current.json"
    path.write_text(json.dumps(records))
    return path


def test_load_examples_pairs_consecutive_plot_points_per_character(tmp_path: Path) -> None:
    examples = adapter.load_examples(_write_fixture(tmp_path))
    assert len(examples) == 1  # only Alice has 2 plot points; Bob has 1 (no pair)
    example = examples[0]
    assert (example.book, example.character) == ("Test Book", "Alice")
    assert (example.step_from, example.step_to) == (1, 2)


def test_load_examples_delta_label_picks_the_dimension_that_gained_triples(tmp_path: Path) -> None:
    example = adapter.load_examples(_write_fixture(tmp_path))[0]
    # belief triple count unchanged (1 -> 1), emotion triple count gained (0 -> 1)
    assert adapter.DELTA_VOCAB[example.delta_label] == "emotion"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_collate_batch_shapes(tmp_path: Path) -> None:
    examples = adapter.load_examples(_write_fixture(tmp_path))
    config = adapter.BatchConfig(nodes=4, edges_per_graph=3, input_dim=8, hidden_dim=16)
    batch = adapter.collate_batch(examples, config)
    assert tuple(batch["node_features"].shape) == (1, 4, 8)
    assert tuple(batch["next_node_features"].shape) == (1, 4, 8)
    assert tuple(batch["action_features"].shape) == (1, 16)
    assert tuple(batch["edge_pairs"].shape) == (1, 3, 2)
    assert tuple(batch["edge_labels"].shape) == (3,)
    assert tuple(batch["attribute_labels"].shape) == (3,)
    assert tuple(batch["emotion_labels"].shape) == (1,)
    assert tuple(batch["delta_labels"].shape) == (1,)
