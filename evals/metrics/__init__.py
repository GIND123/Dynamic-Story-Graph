"""Evaluation metrics over ingested eval manuscripts (READ-ONLY consumers).

Three metrics:
  - quote_attribution : PDNC speaker-attribution accuracy per quote_type.
  - name_cloze        : masked-name recovery over passage text.
  - consistency       : tier1_check precision/recall/F1 on a synthetic set.

None of these modify the graph, the generation pipeline, the schema, or the
ingestion loaders. Graph-dependent metrics query Neo4j directly (NOT get_canon,
which caps events and hides dialogue). Real runs are guarded by driver/dataset
presence, mirroring evals/run_eval.py; tests run fully offline against fixtures.
"""
