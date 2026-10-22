"""The labelling app: saves the guides allow, files check accepts, git sync of own files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dsg.annotate import sheets
from dsg.annotate.sync import GitSync
from dsg.eval import labels

DET_COLS = ["item_id", "item_type", "subject", "kind", "true_value", "matched_word", "context",
            "real_violation", "real_restatement", "labeller", "notes", "double_annotate"]
CALC_COLS = ["item_id", "subject", "old_object", "new_object", "label", "correct_value",
             "correct_value_text", "labeller", "notes", "double_annotate"]


def _detector_row(i: int) -> dict:
    kind = ("flagged_violation", "flagged_restatement", "unflagged_mention")[i % 3]
    return {"item_id": f"D{i:03d}", "item_type": kind, "subject": "Miriam", "kind": "eyes",
            "true_value": "green", "matched_word": "blue",
            "context": f"[[Miriam]]'s <<blue>> eyes, row {i}",
            "real_violation": "n/a" if kind == "flagged_restatement" else "",
            "real_restatement": "n/a" if kind == "flagged_violation" else "",
            "labeller": "", "notes": "", "double_annotate": "yes" if i < 3 else "no"}


def _calculus_row(i: int) -> dict:
    return {"item_id": f"C{i:03d}", "subject": "Harriet", "old_object": "Highbury",
            "new_object": "London", "label": "", "correct_value": "", "correct_value_text": "",
            "labeller": "", "notes": "", "double_annotate": "yes" if i < 2 else "no"}


@pytest.fixture()
def root(tmp_path):
    for sheet, cols, make, n in (("detector", DET_COLS, _detector_row, 9),
                                 ("calculus", CALC_COLS, _calculus_row, 6)):
        d = tmp_path / "audit" / sheet
        labels._write(d / "sheet_blind.csv", cols, [make(i) for i in range(n)])
        labels.split(d, 2)
    return tmp_path


def _answers(sheet: str, row: dict) -> dict:
    if sheet == "calculus":
        return {"label": "SUPERSEDE", "correct_value": "new", "notes": ""}
    return {c: "no" for c in sheets.asked(sheet, row)} | {"notes": ""}


def test_a_row_asks_only_what_the_guide_asks(root):
    _, rows = sheets.load(root, "detector", "annotator_01")
    by_type = {r["item_type"]: sheets.asked("detector", r) for r in rows}
    assert by_type["flagged_violation"] == ["real_violation"]
    assert by_type["flagged_restatement"] == ["real_restatement"]
    assert by_type["unflagged_mention"] == ["real_violation", "real_restatement"]


def test_a_save_keeps_na_cells_and_writes_initials(root):
    _, rows = sheets.load(root, "detector", "annotator_01")
    r = next(x for x in rows if x["item_type"] == "flagged_violation")
    saved = sheets.save(root, "detector", "annotator_01", r["item_id"],
                        {"real_violation": "yes", "real_restatement": "yes", "notes": ""}, "cb")
    assert saved["real_violation"] == "yes"
    assert saved["real_restatement"] == "n/a"
    assert saved["labeller"] == "CB"


@pytest.mark.parametrize("answers, initials", [
    ({"real_violation": "unsure", "real_restatement": "no", "notes": ""}, "CB"),
    ({"real_violation": "maybe", "real_restatement": "no", "notes": ""}, "CB"),
    ({"real_violation": "yes", "notes": ""}, "CB"),
    ({"real_violation": "yes", "real_restatement": "no", "notes": ""}, " "),
])
def test_a_save_the_guide_forbids_is_refused(root, answers, initials):
    _, rows = sheets.load(root, "detector", "annotator_01")
    r = next(x for x in rows if x["item_type"] == "unflagged_mention")
    before = sheets.sheet_path(root, "detector", "annotator_01").read_bytes()
    with pytest.raises(sheets.LabelError):
        sheets.save(root, "detector", "annotator_01", r["item_id"], answers, initials)
    assert sheets.sheet_path(root, "detector", "annotator_01").read_bytes() == before


def test_only_the_saved_row_changes_on_disk(root):
    path = sheets.sheet_path(root, "calculus", "annotator_01")
    before = path.read_bytes().split(b"\r\n")
    _, rows = sheets.load(root, "calculus", "annotator_01")
    sheets.save(root, "calculus", "annotator_01", rows[1]["item_id"],
                {"label": "UNSURE", "correct_value": "unsure", "notes": "pronoun unclear"}, "CB")
    after = path.read_bytes().split(b"\r\n")
    assert [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b] == [2]


def test_files_labelled_in_the_app_pass_check(root):
    for sheet in sheets.SHEETS:
        for me in ("annotator_01", "annotator_02"):
            _, rows = sheets.load(root, sheet, me)
            for r in rows:
                sheets.save(root, sheet, me, r["item_id"], _answers(sheet, r), "CB")
            assert sheets.progress(root, sheet, me)["next_open"] is None
        problems, _ = labels.check(root / "audit" / sheet)
        assert problems == []


# --- git sync -------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture()
def clones(tmp_path, monkeypatch):
    """A bare remote and two checkouts of it, with an isolated git config."""
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Test\n\temail = test@example.com\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(remote), str(seed))
    _git(seed, "symbolic-ref", "HEAD", "refs/heads/main")
    for me in ("annotator_01", "annotator_02"):
        f = seed / "audit" / "detector" / "annotators" / f"{me}.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("item_id,label\r\nD1,\r\n")
    (seed / "README.md").write_text("x\n")
    _git(seed, "add", ".")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "push", "origin", "main")
    out = []
    for name in ("a", "b"):
        _git(tmp_path, "clone", str(remote), str(tmp_path / name))
        out.append(tmp_path / name)
    return remote, out


def _syncer(clone: Path, me: str) -> GitSync:
    return GitSync(clone, [clone / "audit" / "detector" / "annotators" / f"{me}.csv"], me)


def _label(clone: Path, me: str, value: str) -> None:
    (clone / "audit" / "detector" / "annotators" / f"{me}.csv").write_text(
        f"item_id,label\r\nD1,{value}\r\n")


def test_two_annotators_push_their_own_files(clones):
    remote, (a, b) = clones
    _label(a, "annotator_01", "yes")
    assert _syncer(a, "annotator_01").sync("detector 1/1")["error"] is None
    _label(b, "annotator_02", "no")
    (b / "README.md").write_text("an unrelated local edit\n")
    status = _syncer(b, "annotator_02").sync("detector 1/1")
    assert status["error"] is None and status["last_push"]
    log = _git(remote, "log", "--format=%s", "main").splitlines()
    assert log[:2] == ["Labels: annotator_02 detector 1/1 [skip ci]",
                       "Labels: annotator_01 detector 1/1 [skip ci]"]
    files = _git(remote, "show", "--name-only", "--format=", "main").split()
    assert files == ["audit/detector/annotators/annotator_02.csv"]
    assert (b / "README.md").read_text() == "an unrelated local edit\n"


def test_sync_refuses_to_push_other_unpushed_work(clones):
    remote, (a, _) = clones
    (a / "README.md").write_text("someone else's commit\n")
    _git(a, "commit", "-am", "unrelated")
    _label(a, "annotator_01", "yes")
    status = _syncer(a, "annotator_01").sync()
    assert "README.md" in status["error"]
    assert _git(remote, "log", "--format=%s", "main").splitlines() == ["seed"]


def test_sync_refuses_off_main(clones):
    _, (a, _) = clones
    _git(a, "checkout", "-b", "elsewhere")
    _label(a, "annotator_01", "yes")
    assert "not 'main'" in _syncer(a, "annotator_01").sync()["error"]


# --- Flask routes ---------------------------------------------------------------

def test_the_api_saves_and_reports_only_my_problems(root):
    pytest.importorskip("flask")
    from dsg.annotate.app import create_app

    syncer = GitSync(root, [], "annotator_01", enabled=False)
    client = create_app(root, "annotator_01", "CB", syncer).test_client()
    state = client.get("/api/state").get_json()
    assert state["sheets"]["calculus"]["done"] == 0
    row = client.get("/api/row?sheet=calculus&i=0").get_json()
    assert row["asked"] == ["label", "correct_value"]
    bad = client.post("/api/save", json={"sheet": "calculus", "item_id": row["row"]["item_id"],
                                         "answers": {"label": "UNSURE", "correct_value": "new"}})
    assert bad.status_code == 400 and "note" in bad.get_json()["error"]
    ok = client.post("/api/save", json={"sheet": "calculus", "item_id": row["row"]["item_id"],
                                        "answers": {"label": "REVISE", "correct_value": "old"}})
    assert ok.get_json()["progress"]["done"] == 1 and syncer.dirty
    _, rows = sheets.load(root, "detector", "annotator_02")
    r = rows[0]
    sheets.save(root, "detector", "annotator_02", r["item_id"], _answers("detector", r), "XX")
    path = sheets.sheet_path(root, "detector", "annotator_02")
    path.write_text(path.read_text().replace(r["item_id"], "BROKEN", 1))
    problems = client.post("/api/check", json={"sheet": "detector"}).get_json()["problems"]
    assert problems == []
