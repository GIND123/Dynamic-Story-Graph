"""The revision-aware narrative state store.

One class, ``NarrativeState``, holds the reader's model and exposes exactly two
mutating entry points -- ``observe_entity`` and ``apply_assertion``. Both route
every change through the revision calculus, so no caller can write to the store
without the operation being classified, logged and invariant-checked.

Policies differ only in which operations they may use. A policy that may not
revise does not silently drop a conflicting fact: it asserts it anyway, leaving
two live contradictory beliefs in one slot. That is exactly the breakage the
study measures, so it is reproduced faithfully rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsg import invariants
from dsg.lexicon import (
    has_revelation_marker,
    is_multi_valued,
    is_mutable,
    is_underspecified,
    normalize_predicate,
    refines,
)
from dsg.matching import (
    MERGE_THRESHOLD,
    MatchKind,
    classify_surface,
    match_kind,
    surface_score,
)
from dsg.schemas import (
    Assertion,
    Certainty,
    CommitLevel,
    EntityNode,
    Op,
    Span,
    Status,
    UpdateRecord,
    Violation,
)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Which parts of the calculus a system is allowed to use."""

    name: str
    carry_state: bool = True
    deferred_commitment: bool = True
    allow_merge: bool = True
    # Folding two *committed* nodes is a rollback, and is a separate
    # capability from fact-level revision: a system can fix identity while
    # still being unable to retract a fact, and the two buy different things.
    allow_committed_merge: bool = True
    allow_supersede: bool = True
    allow_revise: bool = True
    # ELABORATE rewrites an assertion's object in place. It is monotone in
    # information but it is still a write, so a store that only ever appends
    # cannot perform it: it must add a second assertion and hold both. That
    # inability is precisely the modelled failure, so it is not granted for
    # free to the baselines.
    allow_elaborate: bool = True
    commit_min_mentions: int = 2
    merge_threshold: float = MERGE_THRESHOLD
    # A weaker match still binds when it is the *only* candidate: a reader
    # who meets 'Bennet' knowing exactly one Bennet does take them to be the
    # same person. Deferral is about holding ambiguity open, not refusing to
    # decide when there is nothing to decide between.
    weak_threshold: float = 0.3
    # How close the runner-up must be for the match to count as contested.
    ambiguity_margin: float = 0.15


POLICIES: dict[str, PolicyConfig] = {
    # No persistent state at all: each window is read in isolation.
    "window-only": PolicyConfig(
        "window-only", carry_state=False, deferred_commitment=False,
        allow_merge=False, allow_committed_merge=False,
        allow_supersede=False, allow_revise=False, allow_elaborate=False,
    ),
    # The standard incremental pipeline: accumulate, never unwrite.
    "append-only": PolicyConfig(
        "append-only", deferred_commitment=False, allow_merge=False,
        allow_committed_merge=False, allow_supersede=False, allow_revise=False,
        allow_elaborate=False,
    ),
    # + identity merging (as a rollback, since commitment is eager), but
    # still no fact-level revision: isolates what fixing identity alone buys.
    "dsg-merge": PolicyConfig(
        "dsg-merge", deferred_commitment=False, allow_merge=True,
        allow_committed_merge=True, allow_supersede=False, allow_revise=False,
        allow_elaborate=False,
    ),
    # Full calculus but *eager* commitment: every node commits on first sight,
    # so identity fixes must be non-monotone rollbacks.
    "dsg-eager": PolicyConfig("dsg-eager", deferred_commitment=False),
    # Full system: deferred commitment turns most identity fixes into monotone
    # elaboration of deliberately underspecified structure.
    "dsg-full": PolicyConfig("dsg-full"),
    # Same calculus, fed non-causally (full lookahead). Upper reference.
    "retrospective": PolicyConfig("retrospective"),
}


@dataclass(slots=True)
class CandidateAssertion:
    """A proposal from the extractor, before the calculus rules on it."""

    subject: str                 # entity id
    predicate: str               # raw, un-normalized
    object: str                  # entity id or literal
    polarity: bool = True
    certainty: Certainty = Certainty.NARRATED
    evidence: str = ""
    span: Span | None = None
    confidence: float = 0.5


class NarrativeState:
    def __init__(self, policy: PolicyConfig) -> None:
        self.policy = policy
        self.entities: dict[str, EntityNode] = {}
        self.assertions: dict[str, Assertion] = {}
        self.log: list[UpdateRecord] = []
        self.violations: list[Violation] = []
        self.index = 0
        self.prefix_end = 0
        self._counter = 0
        # Nodes the current step is allowed to match against. A policy that
        # carries no state keeps its earlier nodes in the store -- so its
        # final graph is an honest union of per-window extractions -- but
        # cannot resolve against them.
        self._matchable: set[str] = set()
        # Per-step history for the discourse-position curves.
        self.trace: list[dict[str, int]] = []

    # ------------------------------------------------------------------ ids
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter:05d}"

    def _record(self, op: Op, target: str, detail: str = "", payload: str | None = None) -> Op:
        self.log.append(UpdateRecord(self.index, op, target, detail, payload))
        return op

    # -------------------------------------------------------------- entities
    def live_entities(self) -> list[EntityNode]:
        return [n for n in self.entities.values() if n.live]

    def deref(self, entity_id: str) -> str:
        """Follow a merge chain to the surviving node."""
        seen: set[str] = set()
        cur = entity_id
        while cur in self.entities and self.entities[cur].merged_into is not None:
            if cur in seen:
                break
            seen.add(cur)
            cur = self.entities[cur].merged_into  # type: ignore[assignment]
        return cur

    def _candidates(self, surface: str) -> list[tuple[float, EntityNode, str]]:
        scored: list[tuple[float, EntityNode, str]] = []
        for node in self.live_entities():
            if node.id not in self._matchable:
                continue
            best, kind = 0.0, MatchKind.NONE
            for known in node.surfaces:
                k = match_kind(known, surface)
                s = surface_score(known, surface)
                if s > best:
                    best, kind = s, k
            if best > 0:
                scored.append((best, node, kind))
        scored.sort(key=lambda t: (-t[0], t[1].first_seen))
        return scored

    def resolve(self, surface: str) -> str | None:
        """Best-effort lookup used at query time; never mutates."""
        if not surface:
            return None
        cands = self._candidates(surface)
        if not cands:
            return None
        best, node, _ = cands[0]
        if best >= self.policy.merge_threshold:
            return node.id
        # Ambiguous surname: only answer if exactly one node carries it.
        strong = [c for c in cands if c[0] >= 0.3]
        if len(strong) == 1:
            return strong[0][1].id
        return None

    def observe_entity(
        self,
        surface: str,
        span: Span | None = None,
        hint: str | None = None,
    ) -> str:
        """Bind a surface form to a node, creating or elaborating as required."""
        surface = (surface or "").strip()
        if not surface:
            return ""
        kind = classify_surface(surface)
        proper = kind == "name"
        cands = self._candidates(surface)

        target: EntityNode | None = None
        if cands:
            best, node, match = cands[0]
            contested = (
                len(cands) > 1 and (best - cands[1][0]) < self.policy.ambiguity_margin
            )
            if best >= self.policy.merge_threshold and not contested:
                target = node
            elif best >= self.policy.weak_threshold and not contested:
                # A weak but uncontested match binds under every policy, so the
                # ablation isolates commitment semantics rather than smuggling in
                # a difference in resolution decisions.
                target = node
            elif not self.policy.deferred_commitment and best > 0:
                # Eager commitment resolves a contested reference greedily by
                # score. Deferral instead opens a provisional node below and
                # waits for evidence that says which person this is.
                target = node
                self._record(
                    Op.ASSERT, node.id,
                    f"eager attach of contested '{surface}' on {match} "
                    f"(score {best:.2f}, runner-up {cands[1][0]:.2f})",
                )

        if target is None and hint:
            hinted = self.resolve(hint)
            if hinted is not None:
                target = self.entities[self.deref(hinted)]

        if target is not None:
            was_provisional = target.commit is CommitLevel.PROVISIONAL
            had_name = bool(target.proper_names)
            target.observe(surface, self.index, span, proper)
            if was_provisional and proper and not had_name:
                self._record(
                    Op.ELABORATE, target.id,
                    f"underspecified node named '{surface}'",
                )
            self._maybe_promote(target)
            return target.id

        node = EntityNode(
            id=self._next_id("e"),
            canonical=surface,
            commit=(
                CommitLevel.COMMITTED
                if not self.policy.deferred_commitment
                else (CommitLevel.PROVISIONAL if kind != "name" else CommitLevel.PROVISIONAL)
            ),
            first_seen=self.index,
            last_seen=self.index,
        )
        node.observe(surface, self.index, span, proper)
        self.entities[node.id] = node
        self._matchable.add(node.id)
        self._maybe_promote(node)
        self._record(Op.ASSERT, node.id, f"new {node.commit} entity '{surface}' ({kind})")
        return node.id

    def _maybe_promote(self, node: EntityNode) -> None:
        """A provisional node commits once it is both named and re-attested."""
        if node.commit is CommitLevel.COMMITTED:
            return
        if node.proper_names and node.mention_count() >= self.policy.commit_min_mentions:
            node.commit = CommitLevel.COMMITTED
            self._record(Op.ELABORATE, node.id, "provisional -> committed")

    def names_are_compatible(self, a: EntityNode, b: EntityNode) -> bool:
        """Could these two nodes' proper names denote the same person?

        Compatible means one name is a titled or partial form of the other --
        'Mr. Darcy' and 'Darcy', 'Elizabeth' and 'Elizabeth Bennet'. Two
        *distinct* established names denoting one person does happen in fiction,
        but it is a revelation the text makes explicitly, not something a
        passing mention can assert.
        """
        if not (a.proper_names and b.proper_names):
            return True
        return any(
            match_kind(x, y) in (MatchKind.EXACT, MatchKind.TITLED, MatchKind.NAME_SUBSET)
            for x in a.proper_names
            for y in b.proper_names
        )

    def merge_entities(
        self, keep_id: str, drop_id: str, reason: str = "", revealed: bool = False
    ) -> Op:
        """Fold ``drop`` into ``keep``. Monotone iff ``drop`` was provisional.

        A merge is irreversible in practice: once two characters are fused the
        combined node matches both name sets and attracts further merges, so a
        single bad link cascades. The guard below is therefore a hard structural
        constraint rather than a heuristic -- an extractor may propose that an
        unnamed reference is a named character, but it may not assert that two
        separately named characters are one person. That claim requires an
        explicit identity revelation (``revealed``).
        """
        keep_id, drop_id = self.deref(keep_id), self.deref(drop_id)
        if keep_id == drop_id or keep_id not in self.entities or drop_id not in self.entities:
            return self._record(Op.NOOP, keep_id, "merge no-op")
        if not self.policy.allow_merge:
            return self._record(Op.NOOP, keep_id, f"merge suppressed by policy: {reason}")

        keep, drop = self.entities[keep_id], self.entities[drop_id]
        if not revealed and not self.names_are_compatible(keep, drop):
            return self._record(
                Op.NOOP, keep_id,
                f"refused merge of separately named characters "
                f"({keep.canonical} / {drop.canonical}): {reason}",
            )
        # Keep the node that is already committed, or the earlier one.
        if keep.commit is CommitLevel.PROVISIONAL and drop.commit is CommitLevel.COMMITTED:
            keep, drop = drop, keep
            keep_id, drop_id = drop_id, keep_id

        monotone = drop.commit is CommitLevel.PROVISIONAL
        op = Op.MERGE_PROVISIONAL if monotone else Op.MERGE_COMMITTED
        if op is Op.MERGE_COMMITTED and not self.policy.allow_committed_merge:
            return self._record(
                Op.NOOP, keep_id, f"committed merge not permitted: {reason}"
            )

        for s, c in drop.surfaces.items():
            keep.surfaces[s] = keep.surfaces.get(s, 0) + c
        keep.proper_names |= drop.proper_names
        keep.provenance.extend(drop.provenance[: max(0, 64 - len(keep.provenance))])
        keep.first_seen = min(keep.first_seen, drop.first_seen)
        keep.last_seen = max(keep.last_seen, drop.last_seen)
        if not keep.proper_names and drop.proper_names:
            keep.canonical = sorted(drop.proper_names)[0]
        drop.status = Status.MERGED_AWAY
        drop.merged_into = keep.id

        affected = self._repoint(drop.id, keep.id)
        self._maybe_promote(keep)
        self._record(op, keep.id, f"absorbed {drop.id}: {reason}", payload=drop.id)
        self._reconcile_slots(affected)
        return op

    def _repoint(self, old: str, new: str) -> set[tuple[str, str]]:
        affected: set[tuple[str, str]] = set()
        for a in self.assertions.values():
            if a.subject == old:
                a.subject = new
                affected.add((new, a.predicate))
            if a.object == old:
                a.object = new
                affected.add((a.subject, a.predicate))
        return affected

    # ------------------------------------------------------------ assertions
    def live_on_slot(self, subject: str, predicate: str) -> list[Assertion]:
        return [
            a for a in self.assertions.values()
            if a.live and a.subject == subject and a.predicate == predicate
        ]

    def apply_assertion(self, cand: CandidateAssertion) -> Op:
        predicate = normalize_predicate(cand.predicate)
        subject = self.deref(cand.subject)
        obj = self.deref(cand.object) if cand.object in self.entities else cand.object
        if not subject:
            return self._record(Op.NOOP, "", "assertion without subject")

        existing = self.live_on_slot(subject, predicate)
        same_object = [
            a for a in existing
            if a.object.strip().lower() == str(obj).strip().lower()
        ]

        # Reinforcement: already believed, so only the evidence strengthens.
        for a in same_object:
            if a.polarity == cand.polarity:
                a.confidence = min(1.0, a.confidence + 0.15)
                if a.commit is CommitLevel.PROVISIONAL and a.confidence >= 0.8:
                    a.commit = CommitLevel.COMMITTED
                return self._record(Op.NOOP, a.id, "reinforced")

        conflicts = [
            a for a in existing
            if not is_multi_valued(predicate)
            or a.object.strip().lower() == str(obj).strip().lower()
        ]
        conflicts = [a for a in conflicts if not (a.object == obj and a.polarity == cand.polarity)]

        if not conflicts:
            self._assert_new(subject, predicate, str(obj), cand)
            return Op.ASSERT

        op = self._classify(conflicts[0], str(obj), cand, predicate)

        if op is Op.ELABORATE and self.policy.allow_elaborate:
            target = conflicts[0]
            target.object = str(obj)
            target.confidence = max(target.confidence, cand.confidence)
            target.certainty = cand.certainty
            if target.provenance is None and cand.span is not None:
                target.provenance = cand.span
            return self._record(
                Op.ELABORATE, target.id,
                f"{subject}.{predicate} specified to '{obj}'",
            )

        if op is Op.SUPERSEDE and self.policy.allow_supersede:
            return self._close_and_link(
                conflicts, str(obj), cand, subject, predicate, revise=False
            )

        if op is Op.REVISE and self.policy.allow_revise:
            return self._close_and_link(conflicts, str(obj), cand, subject, predicate, revise=True)

        # The policy cannot express the required operation. Assert anyway and
        # let the contradiction stand -- this is the measured failure mode.
        self._assert_new(subject, predicate, str(obj), cand, record=False)
        return self._record(
            Op.ASSERT, f"{subject}.{predicate}",
            f"unrevised conflict: '{conflicts[0].object}' vs '{obj}' "
            f"(needed {op.value}, policy {self.policy.name})",
        )

    def _classify(
        self, existing: Assertion, obj: str, cand: CandidateAssertion, predicate: str
    ) -> Op:
        """The core judgement: elaboration, world change, or reader correction."""
        if is_underspecified(existing.object) or refines(existing.object, obj):
            return Op.ELABORATE
        if existing.commit is CommitLevel.PROVISIONAL and existing.confidence < 0.5:
            return Op.ELABORATE
        if has_revelation_marker(cand.evidence):
            return Op.REVISE
        if existing.certainty is Certainty.REPORTED and cand.certainty is Certainty.NARRATED:
            return Op.REVISE
        if not is_mutable(predicate):
            return Op.REVISE
        return Op.SUPERSEDE

    def _assert_new(
        self,
        subject: str,
        predicate: str,
        obj: str,
        cand: CandidateAssertion,
        record: bool = True,
    ) -> Assertion:
        a = Assertion(
            id=self._next_id("a"),
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=cand.polarity,
            certainty=cand.certainty,
            commit=CommitLevel.COMMITTED if cand.confidence >= 0.8 else CommitLevel.PROVISIONAL,
            confidence=cand.confidence,
            d_open=self.index,
            valid_from=self.index,
            provenance=cand.span,
        )
        self.assertions[a.id] = a
        if record:
            self._record(Op.ASSERT, a.id, f"{subject}.{predicate} = '{obj}'")
        return a

    def _close_and_link(
        self,
        conflicts: list[Assertion],
        obj: str,
        cand: CandidateAssertion,
        subject: str,
        predicate: str,
        revise: bool,
    ) -> Op:
        self._assert_new(subject, predicate, obj, cand, record=False)
        new_id = f"a{self._counter:05d}"
        new = self.assertions[new_id]
        for old in conflicts:
            old.d_close = self.index
            if revise:
                old.status = Status.RETRACTED
                old.retracted_by = new.id
                self._cascade_retract(old.id)
            else:
                old.status = Status.SUPERSEDED
                old.superseded_by = new.id
                old.valid_to = self.index
                new.supersedes = old.id
        op = Op.REVISE if revise else Op.SUPERSEDE
        return self._record(
            op, new.id,
            f"{subject}.{predicate}: '{conflicts[0].object}' -> '{obj}'",
            payload=conflicts[0].id,
        )

    def _cascade_retract(self, root: str) -> None:
        """A retracted belief takes what was inferred from it with it."""
        frontier = {root}
        while frontier:
            nxt: set[str] = set()
            for a in self.assertions.values():
                if a.live and set(a.derived_from) & frontier:
                    a.status = Status.RETRACTED
                    a.d_close = self.index
                    self._record(Op.REVISE, a.id, "cascade retract")
                    nxt.add(a.id)
            frontier = nxt

    def _reconcile_slots(self, slots: set[tuple[str, str]]) -> None:
        """After a merge, previously separate slots may now collide."""
        for subject, predicate in slots:
            live = self.live_on_slot(subject, predicate)
            if len(live) < 2 or is_multi_valued(predicate):
                continue
            live.sort(key=lambda a: a.d_open)
            keeper = live[-1]
            for old in live[:-1]:
                if old.object.strip().lower() == keeper.object.strip().lower():
                    old.status = Status.SUPERSEDED
                    old.d_close = self.index
                    old.superseded_by = keeper.id
                    self._record(Op.NOOP, old.id, "duplicate folded after merge")
                    continue
                if not self.policy.allow_supersede:
                    continue
                old.status = Status.SUPERSEDED
                old.d_close = self.index
                old.valid_to = self.index
                old.superseded_by = keeper.id
                keeper.supersedes = old.id
                self._record(Op.SUPERSEDE, keeper.id, "post-merge slot reconciliation")

    # ------------------------------------------------------------- bookkeeping
    def step(self, index: int, prefix_end: int) -> None:
        self.index = index
        self.prefix_end = prefix_end
        if not self.policy.carry_state:
            # Keep what was extracted, forget that it is there: no node from
            # an earlier window can be matched, so every re-mention starts a
            # fresh node. That is what 'no persistent state' actually costs.
            self._matchable.clear()

    def close_step(self) -> list[Violation]:
        found = invariants.check(
            self.entities, self.assertions.values(), self.index, self.prefix_end
        )
        self.violations.extend(found)
        self.trace.append(
            {
                "index": self.index,
                "entities_live": len(self.live_entities()),
                "entities_committed": sum(
                    1 for n in self.live_entities()
                    if n.commit is CommitLevel.COMMITTED
                ),
                "assertions_live": sum(1 for a in self.assertions.values() if a.live),
                "violations_new": len(found),
                "violations_total": len(self.violations),
                "ops": len(self.log),
            }
        )
        return found

    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {op.value: 0 for op in Op}
        for rec in self.log:
            counts[rec.op.value] += 1
        counts["monotone"] = sum(1 for r in self.log if r.monotone)
        counts["rollback"] = sum(1 for r in self.log if r.rollback)
        return counts

    def digest(self, max_entities: int = 24) -> str:
        """Bounded, entity-focused state summary -- what the model may see."""
        nodes = sorted(
            self.live_entities(), key=lambda n: (-n.mention_count(), n.first_seen)
        )[:max_entities]
        lines = []
        for n in nodes:
            aliases = sorted(n.surfaces, key=lambda s: -n.surfaces[s])[:4]
            lines.append(f"- {n.canonical} (aka {', '.join(aliases)})")
        return "\n".join(lines)
