"""Readability proxies, the prose-vs-classic comparison, and its report section."""

from __future__ import annotations

import json

import pytest

from dsg.generate.canon import build_premises
from dsg.generate.conditions import PROSE_LADDER, template_of
from dsg.generate.score import (
    DIAGNOSTICS,
    GUARDS,
    PROSE_METRICS,
    classic_twin,
    compare_templates,
    dialogue_share,
    phrase_repeats,
    score_runs,
    word_count,
)


def test_dialogue_share_counts_words_inside_quotes():
    assert dialogue_share('He said, "Come here now." She left.') == pytest.approx(3 / 7)
    assert dialogue_share("“Come here,” he said.") == pytest.approx(2 / 4)
    assert dialogue_share("No one spoke at all.") == 0.0
    assert dialogue_share("") == 0.0


def test_an_unmatched_quote_never_swallows_the_next_paragraph():
    text = '"Wait," he said. "Don\'t\n\nThe rain came down on the empty quay.'
    assert dialogue_share(text) == pytest.approx(1 / word_count(text))


def test_repeated_phrases_are_counted_across_the_story():
    seen: set = set()
    assert phrase_repeats("Hesper set the iron seal on the table.", seen)[0] == 0
    repeated, counted = phrase_repeats("Hesper set the iron seal on the table.", seen)
    assert repeated == counted > 0, "the same sentence again is all repeats"
    fresh = phrase_repeats("Jarrow walked down to the harbour wall alone.", seen)
    assert fresh[0] == 0


def test_function_word_phrases_are_not_counted():
    seen: set = set()
    phrase_repeats("and then he was", seen)
    assert phrase_repeats("and then he was", seen) == (0, 0)


def test_within_a_chapter_repetition_counts_too():
    seen: set = set()
    repeated, _ = phrase_repeats("the grey sea moved. the grey sea moved.", seen)
    assert repeated > 0


# ---------------------------------------------------------------- run scoring

_PROSE_TEXT = (
    '{name} turned the seal over. "Not tonight," she said.\n\n'
    '"Then when?" {other} asked, already at the door.'
)
_CLASSIC_TEXT = (
    "{name} was a person with many qualities. {name} felt sad and thoughtful. "
    "{name} was a person with many qualities and {other} knew it."
)


def _payload(n_stories=4, prose_breaks_canon=False, prose_loses_capture=False):
    premises = build_premises(n=n_stories)
    records = []
    for premise in premises:
        name, other = premise.characters[0], premise.characters[1]
        for condition in PROSE_LADDER:
            prose = template_of(condition) == "prose"
            for chapter in (1, 2, 3):
                text = (_PROSE_TEXT if prose else _CLASSIC_TEXT).format(name=name, other=other)
                broken = prose and prose_breaks_canon and chapter >= 2
                captured = [] if (prose and prose_loses_capture) else [
                    f.fact_id for f in premise.canon
                ]
                records.append({
                    "story_id": premise.story_id, "condition": condition,
                    "chapter": chapter, "chars": len(text), "text": text,
                    "violations": [premise.canon[chapter].fact_id] if broken else [],
                    "restated": [] if prose else [premise.canon[0].fact_id],
                    "evidence": {}, "template": "prose" if prose else "classic",
                    "word_budget": 450 if prose else 500,
                    "truncated": prose and chapter == 3,
                    "continuity": ("cue" if chapter > 1 else "none") if prose else "none",
                    "canon_in_state": captured if "dsg" in condition else [],
                })
    return {
        "meta": {"model": "stub", "stories": n_stories, "chapters": 3,
                 "conditions": list(PROSE_LADDER), "templates": ["classic", "prose"],
                 "example_sha256": "", "beat_rotation": 0},
        "premises": [p.to_json() for p in premises],
        "records": records,
    }


def test_run_scores_carry_every_proxy():
    scores = {s.condition: s.as_dict() for s in score_runs(_payload(n_stories=1))}
    prose, classic = scores["base:last-chapter~prose"], scores["base:last-chapter"]
    sample = _PROSE_TEXT.format(name="Hesper", other="Jarrow")
    assert prose["dialogue_share"] == pytest.approx(4 / word_count(sample))
    assert classic["dialogue_share"] == 0.0
    assert classic["repeated_phrase_rate"] > prose["repeated_phrase_rate"]
    assert prose["truncated_rate"] == pytest.approx(1 / 3)
    assert prose["restated_after_ch1"] == 0.0 and classic["restated_after_ch1"] == 1.0
    assert prose["budget_ratio"] == pytest.approx(
        word_count(_PROSE_TEXT.format(name="Hesper", other="Jarrow")) / 450, rel=0.2
    )
    # Beats 1-3 of the stock outline are large, medium, medium.
    assert prose["words_per_beat_large"] and prose["words_per_beat_medium"]
    assert prose["words_per_beat_small"] is None


def test_old_payloads_without_text_or_budget_still_score():
    premise = build_premises(1)[0]
    payload = {
        "meta": {"chapters": 1}, "premises": [premise.to_json()],
        "records": [{"story_id": premise.story_id, "condition": "base:none",
                     "chapter": 1, "chars": 10, "violations": [], "restated": [],
                     "evidence": {}}],
    }
    row = score_runs(payload)[0].as_dict()
    assert row["budget_ratio"] is None and row["restated_after_ch1"] is None
    assert row["dialogue_share"] == 0.0 and row["words_per_beat"] == 0.0


def test_twins_are_found_through_the_selection_suffix():
    assert classic_twin("base:dsg-hybrid~prose") == "base:dsg-hybrid"
    assert classic_twin("base:last-chapter~prose@bo4-graph") == "base:last-chapter@bo4-graph"


def test_compare_templates_pairs_only_twins_and_reports_guards():
    payload = _payload()
    payload["records"] = [
        r for r in payload["records"] if r["condition"] != "base:dsg-hybrid"
    ]   # leave dsg-hybrid~prose without its twin
    out = compare_templates(score_runs(payload))
    assert list(out) == ["base:last-chapter~prose_vs_base:last-chapter"]
    entry = out["base:last-chapter~prose_vs_base:last-chapter"]
    assert set(entry["metrics"]) == set(PROSE_METRICS) | set(GUARDS) | set(DIAGNOSTICS)
    assert all(v["holds"] for v in entry["guards"].values()), "nothing broke"
    assert entry["canon_capture"] is None, "last-chapter carries no state"


def test_a_prose_arm_that_breaks_canon_fails_its_guard():
    out = compare_templates(score_runs(_payload(prose_breaks_canon=True)))
    for entry in out.values():
        assert not entry["guards"]["violation_rate"]["holds"]
        assert entry["guards"]["on_premise"]["holds"]


def test_a_canon_capture_gap_is_flagged_on_state_arms_only():
    out = compare_templates(score_runs(_payload(prose_loses_capture=True)))
    hybrid = out["base:dsg-hybrid~prose_vs_base:dsg-hybrid"]["canon_capture"]
    assert hybrid["flagged"] and hybrid["classic"] - hybrid["prose"] == pytest.approx(1.0)
    assert out["base:last-chapter~prose_vs_base:last-chapter"]["canon_capture"] is None
    calm = compare_templates(score_runs(_payload()))
    assert not calm["base:dsg-hybrid~prose_vs_base:dsg-hybrid"]["canon_capture"]["flagged"]


# --------------------------------------------------------------------- report

def test_the_report_adds_a_prose_section_only_for_prose_runs(tmp_path):
    from dsg.generate.report import build_report

    source = tmp_path / "generation.json"
    source.write_text(json.dumps(_payload(prose_breaks_canon=True, prose_loses_capture=True)))
    text = build_report(source, tmp_path / "with").read_text()
    section = text.split("## Prose template vs classic", 1)[1]
    assert "bundled treatment" in section
    assert "**not shown to hold**" in section and "**holds**" in section
    assert "FLAG: arms differ by more than 0.10" in section
    assert "restated_after_ch1 (diagnostic)" in section
    assert "n/a (this memory carries no state)" in section
    assert "Continuity device on the prose arm (chapters): cue 8, none 4." in section
    saved = json.loads((tmp_path / "with" / "prose.json").read_text())
    assert len(saved) == 2

    classic = _payload()
    classic["records"] = [r for r in classic["records"] if r["template"] == "classic"]
    classic["meta"]["conditions"] = ["base:last-chapter", "base:dsg-hybrid"]
    source.write_text(json.dumps(classic))
    text = build_report(source, tmp_path / "without").read_text()
    assert "Prose template" not in text
    assert not (tmp_path / "without" / "prose.json").exists()
