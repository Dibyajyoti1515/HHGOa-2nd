"""
retrieval/common.py

Shared constant used across all retrieval strategies. In the notebook
this exact string was repeated inline inside every retrieve_top_k_*
function (Cell 26-DEFS, Cell 39, Cell 45). Centralized here instead of
duplicated once per file.
"""

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_prefixed_query(query_text: str) -> str:
    return f"{QUERY_PREFIX}{query_text}"