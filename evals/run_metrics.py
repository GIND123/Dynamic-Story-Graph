"""Run evaluation metrics over an ingested eval manuscript.

Usage (from repo root):
    python evals/run_metrics.py pdnc:PrideAndPrejudice
    python evals/run_metrics.py litbank:1023_bleak_house

Which metrics run:
  - consistency      : always (synthetic, no graph/dataset needed).
  - quote_attribution: for pdnc:* manuscripts (needs live Neo4j + the PDNC
                       dataset files for character_info.csv).
  - name_cloze       : for any manuscript with passages (needs live Neo4j).

Graph-dependent metrics are guarded by Neo4j connectivity and dataset presence,
mirroring evals/run_eval.py: if either is missing they are skipped (exit 0), the
synthetic consistency metric still runs.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings  # noqa: E402
from evals.metrics import consistency  # noqa: E402


def _print_consistency() -> None:
    report = consistency.evaluate_consistency()
    print("\n=== Consistency / Tier-1 gate (synthetic) ===")
    print(
        f"  precision={report['precision']:.2f} recall={report['recall']:.2f} "
        f"f1={report['f1']:.2f}  "
        f"(TP={report['tp']} FP={report['fp']} FN={report['fn']} TN={report['tn']}, "
        f"n={report['n_cases']})"
    )
    print("  per class:")
    for cls, c in sorted(report["per_class"].items()):
        print(
            f"    {cls:9s} P={c['precision']:.2f} R={c['recall']:.2f} "
            f"F1={c['f1']:.2f}  (tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']})"
        )


def _get_live_driver():
    """Return a connected Neo4j driver, or None if unreachable."""
    try:
        from app import graph

        driver = graph.init_driver()
        driver.verify_connectivity()
        return driver
    except Exception as exc:  # noqa: BLE001 (report and skip, like run_eval)
        print(f"  Neo4j not reachable ({exc}); graph metrics skipped.")
        return None


def _print_quote_attribution(manuscript_id: str, driver) -> None:
    from evals.metrics import quote_attribution

    folder = manuscript_id.split(":", 1)[1]
    novel_dir = Path(settings.pdnc_root) / "data" / folder
    if not novel_dir.is_dir():
        print(f"  PDNC novel dir not found ({novel_dir}); quote_attribution skipped.")
        return

    result = quote_attribution.run_quote_attribution(
        manuscript_id, str(novel_dir), driver=driver
    )
    print("\n=== Quote attribution (gold = PDNC speakers) ===")
    for scope in ("filtered", "all"):
        rep = result[scope]
        print(f"  [{scope}] filter_rule={rep.filter_rule}")
        print(
            f"    overall={rep.overall:.3f}  explicit={rep.explicit:.3f}  "
            f"non_explicit={rep.non_explicit:.3f} "
            f"(implicit={rep.implicit:.3f} anaphoric={rep.anaphoric:.3f})"
        )
        print(f"    n_quotes={rep.n_quotes} n_characters={rep.n_characters}")


def _print_name_cloze(manuscript_id: str, driver) -> None:
    from evals.metrics import name_cloze

    report = name_cloze.run_name_cloze(manuscript_id, driver=driver)
    print("\n=== Name cloze ===")
    print(
        f"  accuracy={report.accuracy:.3f}  n_passages={report.n_passages}  "
        f"n_skipped={report.n_skipped}  seed={report.seed}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_metrics", description=__doc__)
    parser.add_argument("manuscript_id", help="e.g. pdnc:PrideAndPrejudice")
    args = parser.parse_args(argv)
    mid = args.manuscript_id

    print(f"Metrics for manuscript_id={mid}")
    _print_consistency()  # always runs

    driver = _get_live_driver()
    if driver is not None:
        if mid.startswith("pdnc:"):
            _print_quote_attribution(mid, driver)
        _print_name_cloze(mid, driver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
