"""
project/ingestion/explode.py

Explode MS MARCO-XI passages while preserving native text
and metadata.
"""

import pandas as pd


def explode_msmarco_xi(
    df: pd.DataFrame,
    lang_code: str,
) -> pd.DataFrame:

    rows = []

    for _, r in df.iterrows():

        passages = r["passages"]

        eng_passages = passages["English_passages"]
        trans_passages = passages["Translated_passages"]
        is_selected = passages["is_selected"]

        for i, (en_p, tr_p, selected) in enumerate(
            zip(
                eng_passages,
                trans_passages,
                is_selected,
            )
        ):

            rows.append(
                {
                    "text_en": en_p,
                    "passage_id": f"{lang_code}_{r['query_id']}_{i}",
                    "query_id": int(r["query_id"]),
                    "lang": lang_code,
                    "query_type": r["query_type"],
                    "source_dataset": "MSMARCO-XI",
                    "is_selected": bool(selected),
                    "text": tr_p,
                    "query": r["query"],
                    "answer": r["Answer"],
                    "query_en": r["Eng_Query"],
                    "answer_en": r["Eng_Answer"],
                }
            )

    return pd.DataFrame(rows)