from gnsm.schemas import (
    AttributeState,
    Entity,
    EntityType,
    GraphDelta,
    QuoteAttribution,
    SceneGraph,
)
from gnsm.verifier import ConsistencyVerifier


def test_verifier_rejects_dead_character_activity() -> None:
    mara = Entity("character:mara", "Mara", EntityType.CHARACTER)
    current = SceneGraph(
        "s0",
        entities={mara.id: mara},
        attributes=[AttributeState(mara.id, "status", "dead")],
    )
    realized = SceneGraph(
        "s1",
        entities={mara.id: mara},
        quotes=[QuoteAttribution("I returned.", 0, 11, speaker_id=mara.id, confidence=1.0)],
    )
    report = ConsistencyVerifier().verify(current, realized, GraphDelta())
    assert not report.accepted
    assert report.issues[0].kind.value == "dead_character_acts"


def test_verifier_allows_a_reference_to_dead_character() -> None:
    mara = Entity("character:mara", "Mara", EntityType.CHARACTER)
    current = SceneGraph(
        "s0",
        entities={mara.id: mara},
        attributes=[AttributeState(mara.id, "status", "dead")],
    )
    realized = SceneGraph("s1", entities={mara.id: mara})
    report = ConsistencyVerifier().verify(current, realized, GraphDelta())
    assert report.accepted


def test_verifier_accepts_clean_empty_delta() -> None:
    graph = SceneGraph("s0")
    report = ConsistencyVerifier().verify(graph, SceneGraph("s1"), GraphDelta())
    assert report.accepted
