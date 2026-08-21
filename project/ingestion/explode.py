"""
ingestion/explode.py

Ported from Cell 4 of Update_version_2.ipynb ("Explode passages (keeps native text)").
Logic unchanged.
"""

import pandas as pd


def explode_msmarco_xi(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = r["passages"]
        eng_passages = p["English_passages"]
        trans_passages = p["Translated_passages"]
        is_sel = p["is_selected"]
        for i, (en_p, tr_p, sel) in enumerate(zip(eng_passages, trans_passages, is_sel)):
            rows.append({
                "text_en": en_p,
                "passage_id": f"{lang_code}_{r['query_id']}_{i}",
                "query_id": int(r["query_id"]),
                "lang": lang_code,
                "query_type": r["query_type"],
                "source_dataset": "MSMARCO-XI",
                "is_selected": bool(sel),
                "text": tr_p,
                "query": r["query"],
                "answer": r["Answer"],
                "query_en": r["Eng_Query"],
                "answer_en": r["Eng_Answer"],
            })
    return pd.DataFrame(rows)
