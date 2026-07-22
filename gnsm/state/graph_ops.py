"""Pure graph-delta and graph-diff operations."""

from __future__ import annotations

from copy import deepcopy

from gnsm.schemas import (
    AttributeDelta,
    AttributeState,
    DeltaOperation,
    EdgeDelta,
    GraphDelta,
    SceneGraph,
)


def diff_graphs(before: SceneGraph, after: SceneGraph) -> GraphDelta:
    before_edges = before.edge_index()
    after_edges = after.edge_index()
    edge_changes = [
        EdgeDelta(DeltaOperation.REMOVE, before_edges[key])
        for key in before_edges.keys() - after_edges.keys()
    ]
    edge_changes.extend(
        EdgeDelta(DeltaOperation.ADD, after_edges[key])
        for key in after_edges.keys() - before_edges.keys()
    )

    before_attributes = before.attribute_index()
    after_attributes = after.attribute_index()
    attribute_changes: list[AttributeDelta] = []
    for key in before_attributes.keys() | after_attributes.keys():
        old = before_attributes.get(key)
        new = after_attributes.get(key)
        old_value = old.value if old else None
        new_value = new.value if new else None
        if old_value != new_value:
            attribute_changes.append(AttributeDelta(key[0], key[1], old_value, new_value))
    return GraphDelta(edge_changes=edge_changes, attribute_changes=attribute_changes)


def apply_delta(graph: SceneGraph, delta: GraphDelta, *, scene_id: str | None = None) -> SceneGraph:
    result = deepcopy(graph)
    result.scene_id = scene_id or result.scene_id
    edges = result.edge_index()
    for change in delta.edge_changes:
        if change.operation is DeltaOperation.ADD:
            edges[change.edge.key] = change.edge
        else:
            edges.pop(change.edge.key, None)
    result.edges = list(edges.values())

    attributes = result.attribute_index()
    for change in delta.attribute_changes:
        key = change.entity_id, change.key
        if change.new_value is None:
            attributes.pop(key, None)
        else:
            attributes[key] = AttributeState(
                entity_id=change.entity_id,
                key=change.key,
                value=change.new_value,
            )
    result.attributes = list(attributes.values())
    return result
