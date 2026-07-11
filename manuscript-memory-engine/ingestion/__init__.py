"""Dataset ingestion for EVALUATION ONLY.

Projects real literary datasets (PDNC full novels, LitBank gold-span excerpts)
into the existing Neo4j graph + vector index. Ingested manuscripts are tagged
source='eval' so the generation pipeline can never select them.

This package never feeds generation; it exists to evaluate the graph/retrieval
model against gold annotations.
"""
