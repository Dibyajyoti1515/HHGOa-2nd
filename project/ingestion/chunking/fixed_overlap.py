"""
ingestion/chunking/fixed_overlap.py

Ported from Cell 7 of Update_version_2.ipynb -- chunk_fixed_overlap.
Fixed-size token windows with configurable overlap. Logic unchanged.
"""

import pandas as pd


def chunk_fixed_overlap(df: pd.DataFrame, chunk_tokens: int = 256, overlap: float = 0.2) -> pd.DataFrame:
    rows = []
    step = int(chunk_tokens * (1 - overlap))
    for _, r in df.iterrows():
        words = (r["text_en"] or "").split()
        if len(words) <= chunk_tokens:
            spans = [words]
        else:
            spans = [words[i:i + chunk_tokens] for i in range(0, len(words), step) if words[i:i + chunk_tokens]]
        for j, span in enumerate(spans):
            new_row = r.to_dict()
            new_row["text_en"] = " ".join(span)
            new_row["chunk_strategy"] = "fixed_overlap"
            new_row["chunk_id"] = f"{r['passage_id']}_fixed_{j}"
            rows.append(new_row)
    return pd.DataFrame(rows)
