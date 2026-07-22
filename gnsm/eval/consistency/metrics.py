"""Contradiction precision/recall and F1; never report rate alone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def classification_metrics(predicted: set[str], gold: set[str]) -> ClassificationMetrics:
    true_positives = len(predicted & gold)
    false_positives = len(predicted - gold)
    false_negatives = len(gold - predicted)
    precision = true_positives / (true_positives + false_positives) if predicted else 1.0
    recall = true_positives / (true_positives + false_negatives) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationMetrics(
        precision,
        recall,
        f1,
        true_positives,
        false_positives,
        false_negatives,
    )
