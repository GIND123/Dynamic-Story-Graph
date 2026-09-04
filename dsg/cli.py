"""Command line for the Dynamic Story Graph study.

    python -m dsg doctor                       # environment + corpus check
    python -m dsg extract --backend qwen3b     # local MLX extraction
    python -m dsg study  --proposals DIR       # replay policies + score
    python -m dsg report --results DIR         # tables and figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _doctor(args: argparse.Namespace) -> int:
    import importlib

    print(f"python {sys.version.split()[0]}")
    for module in ("numpy", "matplotlib", "mlx_lm", "modal"):
        try:
            m = importlib.import_module(module)
            print(f"  {module:12s} {getattr(m, '__version__', 'ok')}")
        except Exception:
            print(f"  {module:12s} MISSING")
    try:
        from dsg.data.pdnc import load_corpus

        corpus = load_corpus()
        chars = sum(len(n.text) for n in corpus)
        quotes = sum(len(n.quotes) for n in corpus)
        print(f"  PDNC: {len(corpus)} novels, {chars:,} chars, {quotes:,} gold quotes")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"  PDNC: unavailable ({exc})")
    return 0


def _extract(args: argparse.Namespace) -> int:
    from dsg.experiments.run_extraction import run

    out = run(
        Path(args.out), backend_name=args.backend, books=args.books or None,
        max_windows=args.max_windows or None, window_chars=args.window_chars,
        corpus=args.corpus,
    )
    print(f"proposals -> {out}")
    return 0


def _study(args: argparse.Namespace) -> int:
    from dsg.experiments.run_study import run_study, summarize

    payload = run_study(
        Path(args.proposals), Path(args.out),
        policies=args.policies.split(",") if args.policies else None,
        with_curves=not args.no_curves,
        limit=args.books or None,
        corpus=args.corpus,
    )
    print(summarize(payload))
    print(f"\nresults -> {Path(args.out) / 'results.json'}")
    return 0


def _push(args: argparse.Namespace) -> int:
    """Mirror local results to the Hub so a lost laptop costs nothing."""
    from dsg.hub import ARTIFACT_REPO, push_folder

    repo = args.repo or ARTIFACT_REPO
    pushed = []
    for raw in args.paths.split(","):
        path = Path(raw.strip())
        if not path.exists():
            print(f"skip {path} (missing)")
            continue
        url = push_folder(
            path, repo, repo_type="model", path_in_repo=path.name,
            message=f"{args.note}: {path.name}",
        )
        pushed.append(f"{path} -> {url}/tree/main/{path.name}")
    for line in pushed:
        print(line)
    return 0 if pushed else 1


def _gen_report(args: argparse.Namespace) -> int:
    from dsg.generate.report import build_report

    path = build_report(Path(args.generation), Path(args.out))
    print(f"generation report -> {path}")
    return 0


def _compare(args: argparse.Namespace) -> int:
    from dsg.experiments.compare_runs import build

    path = build(
        Path(args.results_root), Path(args.out),
        names=args.runs.split(",") if args.runs else None,
    )
    print(f"comparison -> {path}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from dsg.experiments.report import build_report

    path = build_report(Path(args.results), Path(args.out))
    print(f"report -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dsg", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="environment and corpus check")
    p.set_defaults(func=_doctor)

    p = sub.add_parser("extract", help="run local extraction")
    p.add_argument("--backend", default="qwen3b")
    p.add_argument("--out", default="artifacts/proposals/local")
    p.add_argument("--books", type=int, default=0)
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--window-chars", type=int, default=0)
    p.add_argument("--corpus", default="pdnc")
    p.set_defaults(func=_extract)

    p = sub.add_parser("study", help="replay policies and score")
    p.add_argument("--proposals", required=True)
    p.add_argument("--out", default="artifacts/results/latest")
    p.add_argument("--policies", default="")
    p.add_argument("--books", type=int, default=0)
    p.add_argument("--no-curves", action="store_true")
    p.add_argument("--corpus", default="pdnc")
    p.set_defaults(func=_study)

    p = sub.add_parser("push", help="back up results and figures to the Hub")
    p.add_argument("--paths", default="artifacts/report,artifacts/results")
    p.add_argument("--repo", default="")
    p.add_argument("--note", default="artifact backup")
    p.set_defaults(func=_push)

    p = sub.add_parser("gen-report", help="score a generation run")
    p.add_argument("--generation", required=True)
    p.add_argument("--out", default="artifacts/report/generation")
    p.set_defaults(func=_gen_report)

    p = sub.add_parser("compare", help="cross-corpus and cross-scale comparison")
    p.add_argument("--results-root", default="artifacts/results")
    p.add_argument("--out", default="artifacts/report/comparison")
    p.add_argument("--runs", default="")
    p.set_defaults(func=_compare)

    p = sub.add_parser("report", help="tables and figures")
    p.add_argument("--results", required=True)
    p.add_argument("--out", default="artifacts/report")
    p.set_defaults(func=_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
