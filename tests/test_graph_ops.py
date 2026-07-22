from gnsm.schemas import (
    AttributeDelta,
    AttributeState,
    DeltaOperation,
    EdgeDelta,
    EdgeType,
    Entity,
    EntityType,
    GraphDelta,
    RelationEdge,
    SceneGraph,
)
from gnsm.state.graph_ops import apply_delta, diff_graphs


def test_delta_round_trip() -> None:
    mara = Entity("character:mara", "Mara", EntityType.CHARACTER)
    tower = Entity("location:tower", "Tower", EntityType.LOCATION)
    graph = SceneGraph(
        "s0",
        entities={mara.id: mara, tower.id: tower},
        attributes=[AttributeState(mara.id, "status", "alive")],
    )
    location = RelationEdge(mara.id, tower.id, EdgeType.LOCATED_AT)
    delta = GraphDelta(
        edge_changes=[EdgeDelta(DeltaOperation.ADD, location)],
        attribute_changes=[AttributeDelta(mara.id, "status", "alive", "injured")],
    )
    updated = apply_delta(graph, delta, scene_id="s1")
    observed = diff_graphs(graph, updated)
    assert observed.edge_changes == delta.edge_changes
    assert observed.attribute_changes == delta.attribute_changes
