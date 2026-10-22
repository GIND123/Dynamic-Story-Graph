"""The Flask app: one page, a small JSON API, bound to 127.0.0.1 only."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from dsg.annotate import sheets
from dsg.annotate.sync import GitSync
from dsg.eval.labels import SCHEMAS, check


def create_app(root: Path, me: str, initials: str, syncer: GitSync) -> Flask:
    root = Path(root)
    app = Flask(__name__)

    def summary() -> str:
        parts = []
        for s in sheets.SHEETS:
            p = sheets.progress(root, s, me)
            parts.append(f"{s} {p['done']}/{p['total']}")
        return ", ".join(parts)

    app.config["summary"] = summary

    def fail(message: str, code: int = 400):
        return jsonify({"error": message}), code

    @app.get("/")
    def index():
        return render_template("index.html", me=me, initials=initials)

    @app.get("/api/state")
    def state():
        return jsonify({"me": me, "initials": initials, "sync": syncer.status,
                        "sheets": {s: sheets.progress(root, s, me) for s in sheets.SHEETS}})

    @app.get("/api/row")
    def row():
        sheet = request.args.get("sheet", "")
        try:
            i = int(request.args.get("i", "0"))
            _, rows = sheets.load(root, sheet, me)
        except (ValueError, sheets.LabelError) as exc:
            return fail(str(exc))
        if not 0 <= i < len(rows):
            return fail(f"row {i + 1} is outside 1–{len(rows)}")
        r = rows[i]
        return jsonify({"i": i, "total": len(rows), "row": r,
                        "asked": sheets.asked(sheet, r), "done": sheets.is_done(sheet, r),
                        "options": {c: list(SCHEMAS[sheet][c]) for c in sheets.QUESTIONS[sheet]},
                        "progress": sheets.progress(root, sheet, me)})

    @app.post("/api/save")
    def save():
        body = request.get_json(silent=True) or {}
        sheet = body.get("sheet", "")
        try:
            with syncer.lock:
                saved = sheets.save(root, sheet, me, body.get("item_id", ""),
                                    body.get("answers") or {}, initials)
            syncer.mark_dirty()
        except sheets.LabelError as exc:
            return fail(str(exc))
        return jsonify({"row": saved, "done": sheets.is_done(sheet, saved),
                        "progress": sheets.progress(root, sheet, me), "sync": syncer.status})

    @app.post("/api/check")
    def run_check():
        sheet = (request.get_json(silent=True) or {}).get("sheet", "")
        if sheet not in sheets.SHEETS:
            return fail(f"unknown sheet {sheet!r}")
        problems, _ = check(root / "audit" / sheet, partial=True)
        # check() reads every annotator's file; report only this annotator's rows.
        mine = [p for p in problems if p.startswith(me + ":")]
        return jsonify({"problems": mine})

    @app.post("/api/sync")
    def sync_now():
        return jsonify({"sync": syncer.sync(summary())})

    return app
