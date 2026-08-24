"""
evaluation_adapter/

Thin compatibility layer that exposes HHGoa's existing embedding and
generation components through the interface required by
rag-local-eval-loop (see TARGET_INTERFACE.md in that repo).

Nothing here re-implements HHGoa's retrieval or generation logic. It
only adapts calling conventions:
  - embedder.py  wraps project.ingestion.embeddings.dense_model
  - generator.py wraps project.llm_fallback.groq_client + the same
    confidence-gate policy used in project/api/text.py

See README.md's "Evaluation" section for how to run this.
"""