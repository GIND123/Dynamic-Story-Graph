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


def _with_lengths(prose_words, classic_words):
    """Prose budgets 300 (small beat) and 600 (large beat); classic asks 500."""
    premise = build_premises(1)[0]
    premise.beats = ["An alliance shifts.", "A last reckoning between the two who began it."]
    records = []
    for chapter, budget in ((1, 300), (2, 600)):
        for template, n, asked in (("prose", prose_words[chapter - 1], budget),
                                   ("classic", classic_words, 500)):
            records.append({"story_id": premise.story_id, "chapter": chapter,
                            "template": template, "word_budget": asked,
                            "text": " ".join(["word"] * n)})
    return {"premises": [premise.to_json()], "records": records}


def test_a_budget_counts_as_followed_only_when_length_tracks_it():
    from dsg.generate.report import budget_finding

    followed = budget_finding(_with_lengths((310, 590), 500))
    assert followed["followed"] and followed["budget_ratio"]["prose"] == pytest.approx(
        (310 / 300 + 590 / 600) / 2
    )
    flat = budget_finding(_with_lengths((640, 650), 650))   # writes the same length anyway
    assert not flat["followed"]
    assert flat["prose_words_by_size"] == {"small": 640, "large": 650}


def test_the_report_says_when_the_budget_was_ignored(tmp_path):
    from dsg.generate.report import build_report

    payload = _payload()
    for rec in payload["records"]:          # everyone writes ~6x the budget
        rec["text"] = " ".join([rec["text"]] * 10)
    payload["meta"]["chapter_sampling"] = {"max_tokens": 1400, "repetition_penalty": 1.1}
    source = tmp_path / "generation.json"
    source.write_text(json.dumps(payload))
    section = build_report(source, tmp_path / "r").read_text().split("## Prose template")[1]
    assert "**Absolute budgets were ignored.**" in section and "`budget_ratio`" in section
    # The stock beats 1-3 have no small beat, so relative length cannot be read.
    assert "Relative length by beat size: not measurable" in section
    assert "Chapter sampling (both arms): max_tokens 1400, repetition_penalty 1.1." in section


# ------------------------------------------------------------ example leakage

_EXAMPLE = (
    "Nell cut the heel off the loaf. Tomas watched the hedge while the rain came. "
    "She wiped the knife on her apron and he took the bread to the door."
)


def _leak_payload(prose_texts, classic_texts, example=_EXAMPLE):
    premises = build_premises(n=len(prose_texts))
    records = []
    for premise, prose, classic in zip(premises, prose_texts, classic_texts, strict=True):
        for condition, text in (("base:last-chapter~prose", prose),
                                ("base:last-chapter", classic)):
            records.append({"story_id": premise.story_id, "condition": condition,
                            "chapter": 1, "text": text})
    return {"meta": {"example": example}, "premises": [p.to_json() for p in premises],
            "records": records}


def test_distinctive_words_drop_stopwords_and_what_the_premise_supplies():
    from dsg.generate.score import distinctive_words

    premise = build_premises(1)[0].to_json()
    premise["setting"] += " with a door"
    found = distinctive_words(_EXAMPLE, premise)
    assert {"loaf", "hedge", "apron", "knife"} <= found
    assert not {"the", "and", "she", "her"} & found, "stopwords are not distinctive"
    assert "door" not in found, "the premise supplied it"


def test_leakage_uses_classic_as_the_baseline():
    from dsg.generate.score import compare_leakage

    payload = _leak_payload(
        ["Hesper wiped the knife on her apron beside the hedge."] * 2,
        ["Hesper walked to the quay in the rain."] * 2,
    )
    entry = compare_leakage(payload)["base:last-chapter~prose"]
    assert entry["as_specified"]["mean_a"] > entry["as_specified"]["mean_b"]
    assert {"knife", "apron", "hedge"} <= set(entry["prose_only_words"])


def test_the_sharp_measure_ignores_words_the_model_uses_anyway():
    """"rain" is in the example, but classic writes it in every story regardless."""
    from dsg.generate.score import compare_leakage

    payload = _leak_payload(
        ["The rain came and Hesper waited.", "The rain came and Jarrow waited."],
        ["The rain came over the quay.", "The rain came over the moor."],
    )
    entry = compare_leakage(payload)["base:last-chapter~prose"]
    assert entry["sharp"]["mean_delta"] == pytest.approx(0.0)
    assert "rain" not in entry["prose_only_words"]


def test_no_example_means_no_leakage_section(tmp_path):
    from dsg.generate.report import _leakage_paragraph
    from dsg.generate.score import compare_leakage

    payload = _leak_payload(["x"], ["y"], example="")
    assert compare_leakage(payload) == {}
    assert "not applicable (this run used no example passage)" in _leakage_paragraph(payload)
    payload["meta"]["example_sha256"] = "ab" * 32          # had one, text not recorded
    assert "not measured" in _leakage_paragraph(payload)
    shown = _leakage_paragraph(_leak_payload(["knife apron"], ["quay"]))
    assert "| `base:last-chapter~prose` | sharp |" in shown
    assert "Example words only `base:last-chapter~prose` used: apron, knife." in shown


def test_absolute_and_relative_budget_following_are_reported_separately():
    from dsg.generate.report import _budget_paragraph, budget_finding

    # Prose writes ~1.5x every budget but keeps small beats short; classic is flat.
    payload = _with_lengths((450, 900), 750)
    found = budget_finding(payload)
    assert not found["absolute_followed"] and found["relative_followed"]
    text = _budget_paragraph(payload)
    assert text.startswith("**Absolute budgets were ignored.**")
    assert "Relative length followed beat size in the prose arm" in text
    assert "450 / 900 (small / large)" in text and "750 / 750 (small / large)" in text

    flat = _budget_paragraph(_with_lengths((640, 650), 650))
    assert "Relative length did not follow beat size" in flat
    on_target = _budget_paragraph(_with_lengths((310, 590), 500))
    assert on_target.startswith("**Absolute budgets were roughly followed.**")


def test_the_name_check_flags_example_names_only_prose_wrote():
    from dsg.generate.score import example_name_check

    example = "Nell cut the loaf. Tomas watched. She said nothing, and He left."
    payload = _leak_payload(
        ["Nell walked to the quay. She waited.", "Tomas and Nell argued. He left."],
        ["Hesper walked to the quay. She waited.", "Jarrow argued. He left."],
        example=example,
    )
    flagged = example_name_check(payload)["base:last-chapter~prose"]
    assert set(flagged) == {"Nell", "Tomas"}, "She and He also appear in classic"
    assert len(flagged["Nell"]) == 2 and len(flagged["Tomas"]) == 1
    assert example_name_check(_leak_payload(["x"], ["y"], example="")) == {}


def test_the_leakage_section_shows_the_name_check():
    from dsg.generate.report import _leakage_paragraph

    leaky = _leak_payload(["Nell cut the loaf by the hedge."], ["Hesper walked."],
                          example="Nell cut the loaf by the hedge.")
    assert "**Name check**" in _leakage_paragraph(leaky)
    assert "FLAG -- **Nell** in 1 chapter (" in _leakage_paragraph(leaky)
    clean = _leak_payload(["She cut the loaf."], ["She walked."], example="She cut the loaf.")
    assert "- `base:last-chapter~prose`: none" in _leakage_paragraph(clean)


# -------------------------------------------------------------- copy + bundle

def test_longest_shared_run_counts_consecutive_words():
    from dsg.generate.score import longest_shared_run

    assert longest_shared_run("The rain came down on the quay.", "rain came down on the") == 5
    assert longest_shared_run("One two three.", "four five six.") == 0
    assert longest_shared_run("She Cut THE heel", "she cut the heel off") == 4, "case-folded"


def test_copy_check_flags_eight_word_runs_against_example_and_previous():
    from dsg.generate.score import copy_check

    example = "She cut the heel off the loaf and pushed the rest across the table."
    premise = build_premises(1)[0]
    one = "Hesper walked to the quay and waited for the boats to come in at last."
    recital = "She cut the heel off the loaf and pushed the rest across. Hesper left."
    reuse = "Hesper walked to the quay and waited for the boats. Then rain."
    records = [
        {"story_id": premise.story_id, "condition": c, "chapter": ch, "text": t}
        for c, ch, t in (
            ("base:last-chapter~prose", 1, one), ("base:last-chapter~prose", 2, recital),
            ("base:last-chapter", 1, one), ("base:last-chapter", 2, reuse),
        )
    ]
    found = copy_check({"meta": {"example": example}, "records": records})
    prose, classic = found["base:last-chapter~prose"], found["base:last-chapter"]
    assert prose["example"]["flagged"] == [[premise.story_id, 2, 12]]
    assert prose["previous"]["flagged"] == []
    assert classic["previous"]["flagged"] == [[premise.story_id, 2, 10]]
    assert copy_check({"meta": {}, "records": records})["base:last-chapter"]["example"] is None


def test_the_bundle_note_lists_the_example_only_when_the_run_had_one():
    from dsg.generate.report import bundle_note

    with_example = bundle_note({"example_sha256": "ab" * 32})
    without = bundle_note({"example_sha256": ""})
    assert "an example passage in the system turn" in with_example
    assert "example passage in the system turn" not in without
    assert "This run used no example passage." in without
    for note in (with_example, without):
        assert note.startswith("**Prose is a bundled treatment.**")
        assert "a system voice" in note and "a hint of the next beat" in note


def test_the_report_shows_the_copy_check(tmp_path):
    from dsg.generate.report import build_report

    payload = _payload()
    payload["meta"]["example_sha256"] = ""
    source = tmp_path / "generation.json"
    source.write_text(json.dumps(payload))
    section = build_report(source, tmp_path / "r").read_text().split("## Prose template")[1]
    assert "This run used no example passage." in section
    assert "Example leakage: not applicable" in section
    assert "**Verbatim copying**" in section and "| vs example: longest |" in section
    # The stand-in chapters repeat word for word, so the previous-chapter check fires.
    assert "Flagged: `base:" in section
