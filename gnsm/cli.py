"""Command-line entry points for smoke tests and research stages."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from gnsm.pipeline import GNSMSystem
from gnsm.schemas import PlotAction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnsm", description="Grounded Narrative State Model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the no-download end-to-end reference system")
    demo.add_argument(
        "--scene",
        default='Mara is alive. Mara waited in the observatory. Mara said, "The signal is back."',
    )
    demo.add_argument("--intent", default="Mara studies the returned signal without leaving.")

    extract = subparsers.add_parser("extract", help="extract and print a scene graph")
    extract.add_argument("path", type=Path)
    extract.add_argument("--scene-id", default="scene-0")

    subparsers.add_parser("tree", help="print the intended project directory map")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return _demo(args.scene, args.intent)
    if args.command == "extract":
        return _extract(args.path, args.scene_id)
    if args.command == "tree":
        print(PROJECT_TREE)
        return 0
    return 2


def _demo(scene: str, intent: str) -> int:
    system = GNSMSystem.reference()
    state = system.initialize(scene)
    participants = tuple(
        entity_id for entity_id, entity in state.graph.entities.items() if entity.name == "Mara"
    )
    result = system.generate_next(
        previous_scene=scene,
        state=state,
        action=PlotAction(intent=intent, participants=participants),
        next_scene_id="scene-1",
        rolling_summary="Mara detected a signal in the observatory.",
    )
    payload = {
        "accepted": result.verification.accepted,
        "attempts": result.attempts,
        "draft": result.text,
        "entities": sorted(result.graph.entities),
        "state_dimension": result.state.dimension,
        "diagnostics": result.state.diagnostics,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.verification.accepted else 1


def _extract(path: Path, scene_id: str) -> int:
    system = GNSMSystem.reference()
    graph = system.extractor.extract(path.read_text(encoding="utf-8"), scene_id)
    print(json.dumps(graph.to_dict(), indent=2, default=lambda value: value.value))
    return 0


PROJECT_TREE = """gnsm/
|-- extraction/          C1-C3 entity/coref, quote, relation/state
|-- state/               C4-C5 encoder, transition, losses, probes
|-- generation/          C6 frozen generator and adapters
|-- verifier/            C7 consistency checks and controller
|-- eval/                consistency, context sweep, human study
|-- data/                dataset manifests and local data roots
|-- configs/             defaults, model profiles, ablations
|-- training/            staged training entry points
|-- schemas.py           cross-plane graph/state contracts
|-- pipeline.py          end-to-end facade
`-- cli.py               demo and extraction commands"""
