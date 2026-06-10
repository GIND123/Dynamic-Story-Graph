"""Context assembly for the generation pipeline.

Merges structured canon facts and similar prose passages into a single
token-budgeted context string for the LLM.

Token heuristic: tokens ≈ len(text) // 4. This avoids a network call to the
token-counting API; see DECISIONS.md for the trade-off.
"""

from app.config import settings


def format_canon(canon: dict) -> str:
    """Serialize the canon dict as a human-readable string for the judge prompt.

    Format uses '- Name | status | location | traits' lines so that
    FakeLLM (and humans) can parse character status unambiguously.
    """
    lines: list[str] = ["CHARACTERS:"]
    for c in canon.get("characters", []):
        loc = c.get("location") or "unknown"
        lines.append(f"- {c['name']} | {c['status']} | {loc} | {c['traits']}")
        recent = [r for r in c.get("recent_events", []) if r]
        if recent:
            lines.append(f"  Recent: {'; '.join(recent)}")

    rules = canon.get("rules", [])
    if rules:
        lines.append("\nRULES:")
        for r in rules:
            lines.append(f"- {r}")

    events = canon.get("events", [])
    if events:
        lines.append("\nRECENT EVENTS (chronological):")
        for i, e in enumerate(events, 1):
            lines.append(f"{i}. {e['summary']}")

    return "\n".join(lines)


def _truncate_at_sentence(text: str, max_tokens: int) -> str:
    """Return text that fits within max_tokens (heuristic), cutting at the last
    sentence-ending punctuation. Falls back to a hard cut if none is found."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    chunk = text[:max_chars]
    for i in range(len(chunk) - 1, -1, -1):
        if chunk[i] in ".!?":
            return chunk[: i + 1]

    return chunk


def assemble_context(canon: dict, passages: list[dict]) -> str:
    """Build the full context string for draft_chapter.

    Facts section comes first (always included, load-bearing).
    Passages fill the remaining TOKEN_BUDGET_PASSAGES budget, truncated at
    sentence boundaries if they do not fit entirely.
    """
    facts_str = _truncate_at_sentence(format_canon(canon), settings.token_budget_facts)
    sections = [f"=== CANON FACTS ===\n{facts_str}"]

    if passages:
        passage_budget = settings.token_budget_passages
        used_tokens = 0
        passage_chunks: list[str] = []

        for p in passages:
            text = p.get("text", "")
            text_tokens = len(text) // 4

            if used_tokens + text_tokens <= passage_budget:
                passage_chunks.append(f"[Chapter {p.get('chapter', '?')}] {text}")
                used_tokens += text_tokens
            else:
                remaining = passage_budget - used_tokens
                if remaining > 0:
                    truncated = _truncate_at_sentence(text, remaining)
                    if truncated:
                        passage_chunks.append(
                            f"[Chapter {p.get('chapter', '?')}] {truncated}"
                        )
                break

        if passage_chunks:
            sections.append("=== SIMILAR PASSAGES ===\n" + "\n\n".join(passage_chunks))

    return "\n\n".join(sections)
