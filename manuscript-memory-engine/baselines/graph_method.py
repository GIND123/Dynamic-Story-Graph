"""PREDICTOR 3 — Graph (structural).

Reads ONLY structural facts from a sequence_index neighborhood, with the target
quote's OWN speaker edge HELD OUT (never read). From the held-out neighborhood it
derives who is in the scene + the recent speaker order and predicts via a
documented heuristic (dialogue alternation + scene presence + previous_speaker).
It uses far fewer tokens than the budget — that efficiency is the result.

`neighborhood_agents(..., exclude_seq=center)` is the holdout: the integrity of
the whole experiment depends on the target's gold agent edge being absent from
the predictor's input, which is asserted in the tests.
"""

from __future__ import annotations

import time
from collections import Counter

from baselines.base import BasePredictor, _Driver
from baselines.config import count_tokens


def neighborhood_agents(
    driver: _Driver,
    manuscript_id: str,
    center: int,
    window: int,
    *,
    exclude_seq: int,
) -> list[tuple[int, str]]:
    """Agents of dialogue quotes in [center-window, center+window], in order,
    EXCLUDING the row at `exclude_seq` (the HOLDOUT — the target's own edge).

    Manuscript-scoped. Returns [(sequence_index, agent_name)].
    """
    result = driver.execute_query(
        "MATCH (e:Event {manuscript_id: $mid, is_dialogue: true})"
        "-[:INVOLVES {role: 'agent'}]->(spk:Character) "
        "WHERE e.sequence_index >= $lo AND e.sequence_index <= $hi "
        "AND e.sequence_index <> $exclude "
        "RETURN e.sequence_index AS seq, spk.name AS agent ORDER BY e.sequence_index",
        mid=manuscript_id,
        lo=center - window,
        hi=center + window,
        exclude=exclude_seq,
    )
    return [(r["seq"], r["agent"]) for r in result.records]


def character_degrees(driver: _Driver, manuscript_id: str) -> list[tuple[str, int]]:
    """Characters ranked by INVOLVES degree (structural activity), scoped."""
    result = driver.execute_query(
        "MATCH (c:Character {manuscript_id: $mid})<-[r:INVOLVES]-(:Event) "
        "RETURN c.name AS name, count(r) AS degree ORDER BY degree DESC",
        mid=manuscript_id,
    )
    return [(r["name"], r["degree"]) for r in result.records]


def _predict_from_scene(
    neighborhood: list[tuple[int, str]],
    candidates: list[str],
    previous_speaker: str | None,
) -> str | None:
    """Heuristic: among scene speakers, alternate away from previous_speaker."""
    allowed = set(candidates)
    scene = [agent for _seq, agent in neighborhood if agent in allowed]
    if not scene:
        return None
    if previous_speaker in allowed:
        others = [a for a in scene if a != previous_speaker]
        if others:  # two-party dialogue alternates
            return Counter(others).most_common(1)[0][0]
    return Counter(scene).most_common(1)[0][0]


class GraphPredictor(BasePredictor):
    method_name = "graph"

    # -- quote attribution -------------------------------------------------
    def predict_quote(self, quote_text: str, context: dict) -> str | None:
        mid = context.get("manuscript_id") or self.manuscript_id
        candidates = list(context.get("candidates", []))
        center = context.get("sequence_index")
        previous_speaker = context.get("previous_speaker")
        if center is None:
            self._record(0, 0, 0.0)
            return None

        t0 = time.perf_counter()
        neighborhood = neighborhood_agents(
            self.driver, mid, center, self.config.graph_window, exclude_seq=center
        )
        latency = time.perf_counter() - t0

        facts = "Scene speakers (seq: name):\n" + "\n".join(
            f"{seq}: {agent}" for seq, agent in neighborhood
        )
        facts += f"\nPrevious speaker: {previous_speaker}\nCandidates: {candidates}"
        prediction = _predict_from_scene(neighborhood, candidates, previous_speaker)
        # Structural read: tokens are the few hundred of facts, not the budget.
        self._record(count_tokens(facts), count_tokens(prediction or ""), latency)
        return prediction

    # -- cloze -------------------------------------------------------------
    def predict_cloze(self, masked_passage: str, manuscript_id: str) -> str | None:
        from evals.metrics.name_cloze import _contains, is_proper_name

        t0 = time.perf_counter()
        degrees = character_degrees(self.driver, manuscript_id)
        latency = time.perf_counter() - t0

        # Predict the highest-degree proper-name character NOT already visible in
        # the masked passage (the masked answer is hidden; a visible name is wrong).
        prediction = None
        for name, _deg in degrees:
            if is_proper_name(name) and not _contains(masked_passage, name):
                prediction = name
                break
        facts = "Character degrees:\n" + "\n".join(f"{n}: {d}" for n, d in degrees[:25])
        self._record(count_tokens(facts), count_tokens(prediction or ""), latency)
        return prediction
