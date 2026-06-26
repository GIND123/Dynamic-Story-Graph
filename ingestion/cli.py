"""CLI entrypoint for eval-dataset ingestion.

Usage:
    python -m ingestion.cli pdnc <NovelFolderName> [--manuscript-id ID]
    python -m ingestion.cli litbank <doc_id>       [--manuscript-id ID]

Dataset roots come from config/env (PDNC_ROOT / LITBANK_ROOT), never hardcoded.
For PDNC the identifier is a novel folder under {pdnc_root}/data/ (or a direct
path); for LitBank it is a document id resolved under {litbank_root}.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.config import settings
from ingestion.litbank import LitBankLoader
from ingestion.pdnc import PDNCLoader
from ingestion.project import project_to_graph

logger = logging.getLogger(__name__)


def _resolve_pdnc_dir(identifier: str) -> Path:
    """Resolve a PDNC novel folder from the configured root or a direct path."""
    candidate = Path(settings.pdnc_root) / "data" / identifier
    if candidate.is_dir():
        return candidate
    direct = Path(identifier)
    if direct.is_dir():
        return direct
    raise FileNotFoundError(
        f"PDNC novel folder not found: tried {candidate} and {direct}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingestion.cli", description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)

    p_pdnc = sub.add_parser("pdnc", help="Ingest one PDNC novel folder")
    p_pdnc.add_argument("identifier", help="Novel folder name (under {pdnc_root}/data)")
    p_pdnc.add_argument("--manuscript-id", default=None)

    p_lit = sub.add_parser("litbank", help="Ingest one LitBank document")
    p_lit.add_argument("identifier", help="LitBank document id")
    p_lit.add_argument("--manuscript-id", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    if args.dataset == "pdnc":
        loader = PDNCLoader(_resolve_pdnc_dir(args.identifier))
    else:
        loader = LitBankLoader(settings.litbank_root, args.identifier)

    summary = project_to_graph(loader, args.manuscript_id)
    logger.info(
        "Ingested %s -> manuscript_id=%s "
        "(entities=%d events=%d quotations=%d passages=%d)",
        args.dataset,
        summary["manuscript_id"],
        summary["entities"],
        summary["events"],
        summary["quotations"],
        summary["passages"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
