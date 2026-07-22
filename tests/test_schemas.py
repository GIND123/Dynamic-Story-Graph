from gnsm.schemas import (
    EdgeType,
    Entity,
    EntityType,
    RelationEdge,
    SceneGraph,
)


def test_graph_reports_dangling_edge() -> None:
    mara = Entity("character:mara", "Mara", EntityType.CHARACTER)
    graph = SceneGraph(
        "scene-0",
        entities={mara.id: mara},
        edges=[RelationEdge(mara.id, "location:tower", EdgeType.LOCATED_AT)],
    )
    assert graph.validate() == ["edge target 'location:tower' is not registered"]
