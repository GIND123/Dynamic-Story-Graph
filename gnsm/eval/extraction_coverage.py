"""Measure what the deterministic reference extractor actually finds in a
corpus -- a precondition check for any metric built on top of it.

Why this exists: `gnsm.verifier.consistency.ConsistencyVerifier` is a real,
deterministic checker, so it is tempting to run it over generated text and
report a "consistency score". But all three of its checks are conditional on
structure the extractor must first find:

  _dead_character_checks -> needs `status` attributes
  _location_checks       -> needs LOCATED_AT edges
  _delta_checks          -> needs a predicted GraphDelta

If the extractor finds none of those in a given corpus, the verifier reports
zero violations no matter what the model generates -- a vacuous perfect score
that looks like a result. Run this first and check the coverage before
believing any verifier-derived number on a new corpus.

`gnsm/extraction/reference.py` says so itself: "deliberately conservative
rules... not research-quality replacements for Maverick/BookNLP/PDNC."
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CoverageReport:
    n_texts: int
    mean_entities: float
    mean_edges: float
    mean_attributes: float
    mean_quotes: float
    texts_with_any_edge: int
    texts_with_any_attribute: int

    @property
    def supports_consistency_checks(self) -> bool:
        """True only if the verifier's preconditions appear at all.

        Without edges or attributes the verifier cannot fire, so a
        zero-violation score would carry no information.
        """

        return self.texts_with_any_edge > 0 or self.texts_with_any_attribute > 0


def measure_coverage(texts: list[str], scene_id: str = "coverage-probe") -> CoverageReport:
    from gnsm.extraction.pipeline import ExtractionPipeline

    pipeline = ExtractionPipeline.reference()
    entities = edges = attributes = quotes = 0
    with_edge = with_attribute = 0
    for text in texts:
        graph = pipeline.extract(text, scene_id)
        entities += len(graph.entities)
        edges += len(graph.edges)
        attributes += len(graph.attributes)
        quotes += len(graph.quotes)
        with_edge += 1 if graph.edges else 0
        with_attribute += 1 if graph.attributes else 0

    n = max(1, len(texts))
    return CoverageReport(
        n_texts=len(texts),
        mean_entities=round(entities / n, 3),
        mean_edges=round(edges / n, 3),
        mean_attributes=round(attributes / n, 3),
        mean_quotes=round(quotes / n, 3),
        texts_with_any_edge=with_edge,
        texts_with_any_attribute=with_attribute,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gnsm extraction-coverage",
        description="Does the reference extractor find enough structure for verifier metrics?",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/evolvtrip_data/all_books_current.json")
    )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)

    from gnsm.training.evolvtrip_adapter import load_examples

    examples = load_examples(args.data)[: args.limit]
    report = measure_coverage([example.action_text for example in examples])
    payload = asdict(report) | {"supports_consistency_checks": report.supports_consistency_checks}
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
