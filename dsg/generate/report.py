"""Figures and tables for the generation experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsg.experiments.report import CONTEXT, INK, INK_SOFT, _save, _style
from dsg.generate.score import (
    DIAGNOSTICS,
    PROSE_METRICS,
    compare,
    compare_leakage,
    compare_templates,
    curves,
    score_runs,
    summarize,
)

ORDER = (
    "base:none",
    "base:last-chapter",
    "base:rolling-summary",
    "base:full-context",
    "base:append-only-state",
    "base:dsg-state",
    "tuned:full-context",
    "tuned:dsg-state",
    "tuned:dsg-repair",
    "base:dsg-hybrid",
    "base:dsg-hybrid-repair",
    "base:beat-retrieval",
    "base:beat-hybrid",
    "base:beat-hybrid-repair",
    "base:last-chapter@bo4-random",
    "base:last-chapter@bo4-graph",
)

FOCAL = "#2a78d6"    # the system: state under the full calculus
FOIL = "#eb6834"     # its ablation: state that cannot be revised
STRONG = "#1baf7a"   # the strong practical baseline: paste the transcript

COLOR_FOR = {
    "base:none": CONTEXT,
    "base:last-chapter": CONTEXT,
    "base:rolling-summary": CONTEXT,
    "base:full-context": STRONG,
    "base:append-only-state": FOIL,
    "base:dsg-state": FOCAL,
    "tuned:full-context": STRONG,
    "tuned:dsg-state": FOCAL,
    "tuned:dsg-repair": FOCAL,
    "base:dsg-hybrid": FOIL,
    "base:dsg-hybrid-repair": FOIL,
    "base:beat-retrieval": FOCAL,
    "base:beat-hybrid": FOCAL,
    "base:beat-hybrid-repair": FOCAL,
    "base:last-chapter@bo4-random": FOIL,
    "base:last-chapter@bo4-graph": FOCAL,
}
LABEL = {
    "base:none": "no memory",
    "base:last-chapter": "last chapter",
    "base:rolling-summary": "rolling summary",
    "base:full-context": "full context",
    "base:append-only-state": "state, no revision",
    "base:dsg-state": "DSG state",
    "tuned:full-context": "tuned + full context",
    "tuned:dsg-state": "tuned + DSG state",
    "tuned:dsg-repair": "tuned + DSG + guard",
    "base:dsg-hybrid": "digest + recent",
    "base:dsg-hybrid-repair": "digest + recent + guard",
    "base:beat-retrieval": "beat retrieval",
    "base:beat-hybrid": "beat retrieval + recent",
    "base:beat-hybrid-repair": "beat retrieval + recent + guard",
    "base:last-chapter@bo4-random": "4 samples, chosen blind",
    "base:last-chapter@bo4-graph": "4 samples, ranked by graph",
}
# Drawn with a marker and full weight; everything else is context grey.
EMPHASISED = (
    "base:full-context", "base:append-only-state", "base:dsg-state",
    "tuned:full-context", "tuned:dsg-state", "tuned:dsg-repair",
    "base:dsg-hybrid", "base:dsg-hybrid-repair",
    "base:beat-retrieval", "base:beat-hybrid", "base:beat-hybrid-repair",
    "base:last-chapter@bo4-random", "base:last-chapter@bo4-graph",
)
# Dashed where the backbone is the fine-tuned one, so variant reads off the line.
TUNED = tuple(c for c in ORDER if c.startswith("tuned:"))


def figure_violation_curve(payload: dict, out: Path) -> Path:
    """The headline: how much of the planted canon each condition has broken."""
    scores = score_runs(payload)
    series = curves(scores)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for condition in ORDER:
        ys = series.get(condition)
        if not ys:
            continue
        xs = list(range(1, len(ys) + 1))
        focal = condition in EMPHASISED
        ax.plot(
            xs, ys, color=COLOR_FOR[condition],
            linewidth=2.2 if focal else 1.2, alpha=1.0 if focal else 0.55,
            marker="o" if focal else None, markersize=4,
            linestyle=(0, (5, 2)) if condition in TUNED else "-",
            zorder=3 if focal else 2,
        )
        ax.text(xs[-1] + 0.12, ys[-1], LABEL[condition], fontsize=8, va="center",
                color=COLOR_FOR[condition] if focal else INK_SOFT)
    ax.set_xlim(1, len(next(iter(series.values()))) + 6.5)
    ax.set_ylim(bottom=0)
    _style(
        ax,
        xlabel="chapter",
        ylabel="share of planted canon contradicted",
        title="Canon broken as the story gets longer (lower is better)",
    )
    fig.tight_layout()
    return _save(fig, out)


def figure_final_bars(payload: dict, out: Path) -> Path:
    scores = score_runs(payload)
    panels = [
        ("violation_rate", "Canon contradicted by the end", "lower is better"),
        ("restatement_rate", "Canon actively restated", "higher is better"),
        ("prompt_tokens_last", "Context tokens at the final chapter",
         "lower is cheaper"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    present = [c for c in ORDER if any(s.condition == c for s in scores)]
    for ax, (metric, title, note) in zip(axes, panels, strict=False):
        vals = []
        for condition in present:
            rows = [s.as_dict()[metric] for s in scores if s.condition == condition]
            vals.append(sum(rows) / len(rows) if rows else 0.0)
        ax.barh(range(len(present)), vals,
                color=[COLOR_FOR[c] for c in present], height=0.62, zorder=3)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels([LABEL[c] for c in present], fontsize=8)
        ax.invert_yaxis()
        span = max(vals) if max(vals) else 1.0
        for i, v in enumerate(vals):
            ax.text(v + span * 0.02, i,
                    f"{v:,.0f}" if metric in ("total_chars", "prompt_tokens_last")
                    else f"{v:.3f}",
                    va="center", fontsize=8, color=INK)
        ax.set_xlim(0, span * 1.25)
        _style(ax, xlabel=note, title=title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out)


def _comparison_table(comparisons: dict) -> str:
    lines = ["| comparison | metric | mean Δ | 95% CI | excludes 0 | wins | n |",
             "|---|---|---|---|---|---|---|"]
    for pair, entry in comparisons.items():
        n = int(entry.get("n_stories", 0))
        for metric in ("violation_rate", "restatement_rate",
                       "mean_first_violation", "total_chars",
                       "prompt_tokens_last"):
            stat = entry.get(metric)
            if not stat:
                continue
            lines.append(
                f"| {pair.replace('_vs_', ' vs ')} | {metric} | "
                f"{stat['mean_delta']:+.4f} | "
                f"[{stat['ci_low']:+.4f}, {stat['ci_high']:+.4f}] | "
                f"{'**yes**' if stat['excludes_zero'] else 'no'} | "
                f"{int(stat['wins_a'])}–{int(stat['wins_b'])} | {n} |"
            )
    return "\n".join(lines)


def _examples(payload: dict, limit: int = 12) -> str:
    rows = [
        r for r in payload["records"]
        if r.get("evidence")
        and r["condition"] in ("base:full-context", "base:rolling-summary",
                               "base:none", "base:append-only-state")
    ]
    rows.sort(key=lambda r: -r["chapter"])
    if not rows:
        return "_none recorded_"
    lines = ["| condition | ch | fact | contradicting text |", "|---|---|---|---|"]
    for r in rows[:limit]:
        for fact_id, why in list(r["evidence"].items())[:1]:
            lines.append(
                f"| {r['condition']} | {r['chapter']} | `{fact_id}` | "
                f"…{why[:130].replace('|', '/')}… |"
            )
    return "\n".join(lines)


BUNDLE_NOTE = (
    "**Prose is a bundled treatment.** Its arm differs from classic in all of: "
    "a system voice (replacing the backbone's default assistant prompt), the "
    "example passage, canon framed as background in plain wording, the previous "
    "chapter's end (quoted, or a cue when the memory already carries it), a word "
    "budget sized to the beat (chapter 1 never below medium), a hint of the next "
    "beat, and instructions against summarised feelings and listed traits. A "
    "difference between the arms is the bundle's, not any one part's."
)


def _fmt(value: float, metric: str) -> str:
    return f"{value:,.0f}" if metric.startswith("words_per_beat") else f"{value:.3f}"


# A budget counts as followed only when writing lands within this band of what
# was asked AND small beats come out clearly shorter than large ones.
BUDGET_BAND = (0.8, 1.2)
SIZE_SPREAD = 0.8   # small-beat words must be <= 80% of large-beat words


def budget_finding(payload: dict) -> dict:
    """What the model did with the word budgets, per template, from the records."""
    from dsg.generate.canon import beat_size
    from dsg.generate.score import word_count

    beats = {p["story_id"]: p["beats"] for p in payload["premises"]}
    ratios: dict[str, list[float]] = {}
    sized: dict[str, list[int]] = {}
    for rec in payload["records"]:
        if not rec.get("word_budget") or not rec.get("text"):
            continue
        n = word_count(rec["text"])
        template = rec.get("template", "classic")
        ratios.setdefault(template, []).append(n / rec["word_budget"])
        if template == "prose":
            story_beats = beats[rec["story_id"]]
            size = beat_size(story_beats[(rec["chapter"] - 1) % len(story_beats)])
            sized.setdefault(size, []).append(n)
    mean = {t: sum(v) / len(v) for t, v in ratios.items()}
    words = {k: sum(v) / len(v) for k, v in sized.items()}
    prose = mean.get("prose")
    in_band = prose is not None and BUDGET_BAND[0] <= prose <= BUDGET_BAND[1]
    spread = (
        words["small"] <= SIZE_SPREAD * words["large"]
        if "small" in words and "large" in words else None
    )
    return {"budget_ratio": mean, "prose_words_by_size": words,
            "followed": bool(in_band and spread)}


def _budget_paragraph(payload: dict) -> str:
    found = budget_finding(payload)
    ratio = found["budget_ratio"]
    sizes = ", ".join(f"{k} {v:,.0f}" for k, v in sorted(found["prose_words_by_size"].items()))
    cited = (
        f"`budget_ratio` (words written / words asked): prose "
        f"{ratio.get('prose', float('nan')):.2f}, classic {ratio.get('classic', float('nan')):.2f}"
        f"; prose words per chapter by beat size: {sizes or 'n/a'}."
    )
    if found["followed"]:
        return f"**Word budgets were roughly followed.** {cited}"
    return (
        "**The model did not follow the word budget.** " + cited + " Per-beat budgets "
        "were requested, not achieved, so this run does not test them: length "
        "differences below are what the model wrote, not what it was asked for."
    )


def _leakage_paragraph(payload: dict) -> str:
    leaks = compare_leakage(payload)
    if not leaks:
        return "Example leakage: not measured (no example text recorded for this run)."
    lines = [
        "**Example leakage** -- share of the example's distinctive content words "
        "that appear in a story. Classic never sees the example, so it is the "
        "baseline. *As specified*: stopwords and words the premise supplies are "
        "excluded. *Sharp*: also excludes words the classic arm writes in other "
        "stories, which this model uses with no example at all and which otherwise "
        "swamp the few that betray it.",
        "",
        "| prose arm | measure | prose | classic baseline | Δ | 95% CI | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for prose, entry in leaks.items():
        for name in ("as_specified", "sharp"):
            stat = entry[name]
            lines.append(
                f"| `{prose}` | {name.replace('_', ' ')} | {stat['mean_a']:.3f} | "
                f"{stat['mean_b']:.3f} | {stat['mean_delta']:+.3f} | "
                f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] | {int(stat['n'])} |"
            )
    lines.append("")
    for prose, entry in leaks.items():
        found = ", ".join(entry.get("prose_only_words", [])) or "none"
        lines.append(f"Example words only `{prose}` used: {found}.")
    return "\n".join(lines)


def _prose_section(payload: dict, comparisons: dict) -> str:
    meta = payload["meta"]
    example = meta.get("example_sha256") or ""
    lines = [
        "## Prose template vs classic",
        "",
        BUNDLE_NOTE,
        "",
        f"Example passage: {'sha256 `' + example[:12] + '…`' if example else 'none'}. "
        f"Beat rotation: {meta.get('beat_rotation', 0)}. Every comparison is "
        "paired within story; with few stories the intervals are wide, and the "
        "blinded human read is the outcome these numbers explain.",
        "",
        "Chapter sampling (both arms): " + ", ".join(
            f"{k} {v}" for k, v in meta.get("chapter_sampling", {"max_tokens": 900}).items()
        ) + ".",
        "",
        _budget_paragraph(payload),
        "",
        _leakage_paragraph(payload),
    ]
    continuity: dict[str, dict[str, int]] = {}
    for rec in payload["records"]:
        if rec.get("template") == "prose":
            bucket = continuity.setdefault(rec["condition"], {})
            device = rec.get("continuity", "none")
            bucket[device] = bucket.get(device, 0) + 1
    for entry in comparisons.values():
        m = entry["metrics"]
        lines += ["", f"### `{entry['prose']}` vs `{entry['classic']}` "
                      f"({entry['n_stories']} stories)", "",
                  "**Guards** (a prose gain counts only if both hold):", ""]
        for metric, verdict in entry["guards"].items():
            stat = m[metric]
            side = "upper" if verdict["margin"] > 0 else "lower"
            lines.append(
                f"- `{metric}`: Δ {stat['mean_delta']:+.3f} "
                f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] -- "
                f"{'**holds**' if verdict['holds'] else '**not shown to hold**'} "
                f"({side} bound vs margin {verdict['margin']:+.2f})"
            )
        capture = entry["canon_capture"]
        if capture is None:
            lines.append("- `canon_capture`: n/a (this memory carries no state)")
        else:
            flag = " -- **FLAG: arms differ by more than 0.10**" if capture["flagged"] else ""
            lines.append(
                f"- `canon_capture`: prose {capture['prose']:.3f}, "
                f"classic {capture['classic']:.3f}{flag}"
            )
        lines += ["", "| metric | prose | classic | Δ | 95% CI | prose–classic wins | n |",
                  "|---|---|---|---|---|---|---|"]
        for metric in PROSE_METRICS + DIAGNOSTICS:
            stat = m[metric]
            if not stat["n"]:
                lines.append(f"| {metric} | – | – | – | – | – | 0 |")
                continue
            label = f"{metric} (diagnostic)" if metric in DIAGNOSTICS else metric
            lines.append(
                f"| {label} | {_fmt(stat['mean_a'], metric)} | "
                f"{_fmt(stat['mean_b'], metric)} | {stat['mean_delta']:+.3f} | "
                f"[{stat['ci_low']:+.3f}, {stat['ci_high']:+.3f}] | "
                f"{int(stat['wins_a'])}–{int(stat['wins_b'])} | {int(stat['n'])} |"
            )
        used = continuity.get(entry["prose"], {})
        if used:
            counts = ", ".join(f"{k} {v}" for k, v in sorted(used.items()))
            lines += ["", f"Continuity device on the prose arm (chapters): {counts}."]
        lines.append("")
    lines += [
        "Diagnostics carry no verdict: restating less is the intended effect, "
        "and it also leaves fewer mentions for a contradiction to be found near, "
        "so read `violation_rate` with them, never alone.",
        "",
    ]
    return "\n".join(lines)


def build_report(generation_json: Path, out_dir: Path) -> Path:
    payload = json.loads(Path(generation_json).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = out_dir / "figures"
    figure_violation_curve(payload, figs / "gen1-canon-violation")
    figure_final_bars(payload, figs / "gen2-final")

    scores = score_runs(payload)
    meta = payload["meta"]
    body = [
        "# Generation: consistency at length",
        "",
        f"{meta['stories']} stories × {len(meta['conditions'])} memory conditions × "
        f"{meta['chapters']} chapters, {meta['model']}. "
        f"Canon is stated in chapter 1 only; from chapter 2 each condition keeps "
        f"whatever its memory carries. Violations are deterministic string tests "
        f"against facts we planted, so no model or human judgement enters the "
        f"measurement.",
        "",
        "## Means over stories",
        "",
        summarize(scores, ORDER),
        "",
        "## Paired comparisons (bootstrap over stories)",
        "",
        _comparison_table(compare(scores)),
        "",
        "## Continuity errors on real generated text",
        "",
        _examples(payload),
        "",
        "## Figures",
        "",
        "![canon](figures/gen1-canon-violation.png)",
        "![final](figures/gen2-final.png)",
        "",
    ]
    templates = compare_templates(scores)
    if templates:
        (out_dir / "prose.json").write_text(json.dumps(templates, indent=2))
        body += [_prose_section(payload, templates), ""]
    path = out_dir / "generation.md"
    path.write_text("\n".join(body))
    return path
