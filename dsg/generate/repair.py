"""Using the graph as a guard on generation.

After a chapter is written it is read back, and every fact it proposes is put to
the state as a dry run. A proposal that would force a `REVISE` on an
**immutable** predicate is, by the calculus's own definition, not the world
changing but the text contradicting what it already established -- eye colour,
material, kinship and birthplace do not change within a story. That is the
signal to reject the chapter and ask again, naming what was broken.

The guard reuses ``classify_only``, the same classifier the update path uses, so
it cannot drift away from the calculus it is guarding.
"""

from __future__ import annotations

from dataclasses import dataclass

from dsg.lexicon import is_mutable, normalize_predicate
from dsg.proposals import WindowProposal
from dsg.schemas import Op
from dsg.store import CandidateAssertion, NarrativeState


@dataclass(frozen=True, slots=True)
class Conflict:
    subject: str
    predicate: str
    established: str
    proposed: str

    def describe(self) -> str:
        return (
            f"{self.subject}: already established as "
            f"{self.predicate.replace('_', ' ')} \"{self.established}\", "
            f"but this chapter says \"{self.proposed}\""
        )


def detect_conflicts(
    state: NarrativeState, proposal: WindowProposal, max_report: int = 6
) -> list[Conflict]:
    """Immutable-predicate contradictions the chapter would introduce."""
    found: list[Conflict] = []
    seen: set[tuple[str, str]] = set()
    for fact in proposal.facts:
        predicate = normalize_predicate(fact.predicate)
        if is_mutable(predicate):
            continue
        subject_id = state.resolve(fact.subject)
        if subject_id is None:
            continue
        op, prior = state.classify_only(
            CandidateAssertion(
                subject=subject_id, predicate=fact.predicate, object=fact.object
            )
        )
        if op is not Op.REVISE or prior is None:
            continue
        key = (subject_id, predicate)
        if key in seen:
            continue
        seen.add(key)
        node = state.entities.get(subject_id)
        established = (
            state.entities[prior.object].canonical
            if prior.object in state.entities
            else prior.object
        )
        found.append(
            Conflict(
                subject=node.canonical if node else fact.subject,
                predicate=predicate,
                established=established,
                proposed=fact.object,
            )
        )
        if len(found) >= max_report:
            break
    return found


def repair_instruction(conflicts: list[Conflict]) -> str:
    """The corrective note appended when a chapter is sent back."""
    if not conflicts:
        return ""
    lines = "\n".join(f"- {c.describe()}" for c in conflicts)
    return (
        "\nYour previous attempt at this chapter contradicted the established "
        "canon:\n" + lines + "\nRewrite the chapter so that every one of those "
        "details matches what was established. Change the wording, not the "
        "canon.\n"
    )
