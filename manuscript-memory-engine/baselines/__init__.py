"""Baseline comparison: three READ-ONLY predictors raced through the Change-3
metric harnesses on an EQUAL information budget.

  - flat_long_context : reads a window of surrounding PROSE (~TOKEN_BUDGET).
  - vector_rag        : retrieves budget-matched k PROSE chunks (excludes the
                        target line) and reasons over them.
  - graph_method      : reads ONLY structural facts from a sequence_index
                        neighborhood with the target quote's own speaker edge
                        held out — far fewer tokens than the budget (the point).

Predictors never modify the graph/pipeline. The graph predictor never reads the
target quote's own gold agent edge (the holdout). Each records per-call
input/output tokens and latency so the runner can report accuracy + tokens +
cost + latency side by side.
"""
