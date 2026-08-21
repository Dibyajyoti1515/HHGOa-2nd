"""
ingestion/chunking/semantic.py

Ported from Cell 7 of Update_version_2.ipynb -- chunk_semantic.
Sentence-similarity based grouping with target/min/max token bounds
(fixed version: 128/256/320). Logic unchanged.
"""

import re as _re

import numpy as np
import pandas as pd


def chunk_semantic(df: pd.DataFrame, model, similarity_threshold: float = 0.75,
                    target_tokens: int = 256, min_tokens: int = 128, max_tokens: int = 320) -> pd.DataFrame:

    def word_count(sents):
        return sum(len(s.split()) for s in sents)

    rows = []
    for _, r in df.iterrows():
        sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", r["text_en"] or "") if s.strip()]
        if not sents:
            continue
        if len(sents) == 1:
            groups = [sents]
        else:
            embs = model.encode(sents, normalize_embeddings=True)
            groups, current = [], [sents[0]]
            for i in range(1, len(sents)):
                sim = float(np.dot(embs[i - 1], embs[i]))
                current_len = word_count(current)
                force_split = current_len >= max_tokens
                natural_split = (sim < similarity_threshold) and (current_len >= target_tokens * 0.7)
                if force_split or natural_split:
                    groups.append(current)
                    current = []
                current.append(sents[i])
            groups.append(current)

            merged = []
            i = 0
            while i < len(groups):
                g = groups[i]
                if word_count(g) < min_tokens:
                    if i + 1 < len(groups):
                        groups[i + 1] = g + groups[i + 1]
                    elif merged:
                        merged[-1] = merged[-1] + g
                    else:
                        merged.append(g)
                    i += 1
                    continue
                merged.append(g)
                i += 1
            groups = merged

        for j, g in enumerate(groups):
            if not g:
                continue
            new_row = r.to_dict()
            new_row["text_en"] = " ".join(g)
            new_row["chunk_strategy"] = "semantic"
            new_row["chunk_id"] = f"{r['passage_id']}_sem_{j}"
            new_row["chunk_word_count"] = word_count(g)
            rows.append(new_row)
    return pd.DataFrame(rows)
