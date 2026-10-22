"""Start the labelling app for one annotator.

    python -m dsg.annotate --me annotator_01 --initials CB

Pulls main on start, pushes your own two files every five minutes while there
are new saves, and once more when you stop it (Ctrl-C). ``--no-sync`` works
offline; use the "Sync now" button or restart with sync on later.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dsg.annotate import sheets
from dsg.annotate.sync import GitSync


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dsg.annotate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--me", required=True, help="your file name, e.g. annotator_01")
    p.add_argument("--initials", required=True, help="written to the labeller column")
    p.add_argument("--port", type=int, default=8502)
    p.add_argument("--interval", type=float, default=300, help="seconds between pushes")
    p.add_argument("--no-sync", action="store_true", help="never pull or push")
    p.add_argument("--branch", default="main")
    p.add_argument("--root", default=".", help="repository root")
    args = p.parse_args(argv)

    if not re.fullmatch(r"annotator_\d\d", args.me):
        p.error("--me must look like annotator_01")
    root = Path(args.root).resolve()
    paths = [sheets.sheet_path(root, s, args.me) for s in sheets.SHEETS]
    missing = [str(x) for x in paths if not x.exists()]
    if missing:
        p.error("not found: " + ", ".join(missing) + " (run from the repository root)")

    syncer = GitSync(root, paths, args.me, branch=args.branch, enabled=not args.no_sync)
    syncer.pull()
    if syncer.status["error"]:
        print(f"[sync] {syncer.status['error']}", file=sys.stderr)

    from dsg.annotate.app import create_app

    app = create_app(root, args.me, args.initials, syncer)
    syncer.start(args.interval, app.config["summary"])
    print(f"Labelling as {args.me} ({args.initials.upper()}): http://127.0.0.1:{args.port}")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    finally:
        if syncer.enabled and syncer.dirty:
            status = syncer.sync(app.config["summary"]())
            print(f"[sync] {status['error'] or 'pushed at ' + status['last_push']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
