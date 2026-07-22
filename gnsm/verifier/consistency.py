"""Deterministic checks against current canon and predicted graph deltas."""

from __future__ import annotations

from gnsm.schemas import (
    DeltaOperation,
    EdgeType,
    GraphDelta,
    IssueKind,
    SceneGraph,
    VerificationIssue,
    VerificationReport,
)
from gnsm.state.graph_ops import diff_graphs


class ConsistencyVerifier:
    def verify(
        self,
        current: SceneGraph,
        realized: SceneGraph,
        predicted_delta: GraphDelta,
    ) -> VerificationReport:
        issues: list[VerificationIssue] = []
        issues.extend(self._dead_character_checks(current, realized))
        issues.extend(self._location_checks(current, realized, predicted_delta))
        issues.extend(self._delta_checks(current, realized, predicted_delta))
        return VerificationReport(
            accepted=not issues,
            issues=issues,
            precision_denominator=len(issues),
            recall_denominator=(
                len(predicted_delta.edge_changes) + len(predicted_delta.attribute_changes)
            ),
        )

    @staticmethod
    def _dead_character_checks(
        current: SceneGraph, realized: SceneGraph
    ) -> list[VerificationIssue]:
        status = {
            attribute.entity_id: attribute.value
            for attribute in current.attributes
            if attribute.key == "status"
        }
        action_edges = {EdgeType.LOCATED_AT, EdgeType.POSSESSES, EdgeType.SPEAKS_TO}
        active_entities = {edge.source for edge in realized.edges if edge.type in action_edges}
        active_entities.update(
            quote.speaker_id for quote in realized.quotes if quote.speaker_id is not None
        )
        return [
            VerificationIssue(
                IssueKind.DEAD_CHARACTER_ACTS,
                f"{entity_id} is dead in canon and cannot act or speak.",
                (entity_id,),
            )
            for entity_id in active_entities
            if status.get(entity_id) == "dead"
        ]

    @staticmethod
    def _location_checks(
        current: SceneGraph,
        realized: SceneGraph,
        predicted_delta: GraphDelta,
    ) -> list[VerificationIssue]:
        expected_moves = {
            change.edge.source
            for change in predicted_delta.edge_changes
            if change.edge.type is EdgeType.LOCATED_AT
        }
        old_locations = {
            edge.source: edge.target for edge in current.edges if edge.type is EdgeType.LOCATED_AT
        }
        realized_locations = {
            edge.source: edge.target for edge in realized.edges if edge.type is EdgeType.LOCATED_AT
        }
        issues: list[VerificationIssue] = []
        for entity_id, location_id in realized_locations.items():
            old_location = old_locations.get(entity_id)
            if old_location and old_location != location_id and entity_id not in expected_moves:
                issues.append(
                    VerificationIssue(
                        IssueKind.WRONG_LOCATION,
                        f"{entity_id} moved from {old_location} to {location_id} "
                        "without a predicted move.",
                        (entity_id, old_location, location_id),
                    )
                )
        return issues

    @staticmethod
    def _delta_checks(
        current: SceneGraph,
        realized: SceneGraph,
        predicted_delta: GraphDelta,
    ) -> list[VerificationIssue]:
        if predicted_delta.is_empty:
            return []
        actual = diff_graphs(current, realized)
        actual_edges = {(change.operation, change.edge.key) for change in actual.edge_changes}
        issues: list[VerificationIssue] = []
        for expected in predicted_delta.edge_changes:
            if (expected.operation, expected.edge.key) not in actual_edges:
                operation = "added" if expected.operation is DeltaOperation.ADD else "removed"
                issues.append(
                    VerificationIssue(
                        IssueKind.MISSING_EXPECTED_CHANGE,
                        f"Expected {expected.edge.type.value} edge to be {operation}: "
                        f"{expected.edge.source} -> {expected.edge.target}.",
                        (expected.edge.source, expected.edge.target),
                    )
                )
        actual_attributes = {
            (change.entity_id, change.key, change.old_value, change.new_value)
            for change in actual.attribute_changes
        }
        for expected in predicted_delta.attribute_changes:
            key = (expected.entity_id, expected.key, expected.old_value, expected.new_value)
            if key not in actual_attributes:
                issues.append(
                    VerificationIssue(
                        IssueKind.MISSING_EXPECTED_CHANGE,
                        f"Expected {expected.entity_id}.{expected.key} to change from "
                        f"{expected.old_value!r} to {expected.new_value!r}.",
                        (expected.entity_id,),
                    )
                )
        return issues
