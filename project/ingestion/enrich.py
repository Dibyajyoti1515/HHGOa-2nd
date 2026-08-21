"""
ingestion/enrich.py

Ported from Cell 6 of Update_version_2.ipynb ("Enrich: regex years +
spaCy NER"). YEAR_RE is the same regex constant defined in the
notebook's Cell 1. Logic unchanged.
"""

import re

import pandas as pd

YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    year_mentions, ppl, orgs, locs = [], [], [], []
    for doc_text in nlp.pipe(df["text_en"].fillna("").tolist(), batch_size=256):
        year_mentions.append([int(y) for y in YEAR_RE.findall(doc_text.text)])
        ppl.append([e.text for e in doc_text.ents if e.label_ == "PERSON"])
        orgs.append([e.text for e in doc_text.ents if e.label_ == "ORG"])
        locs.append([e.text for e in doc_text.ents if e.label_ in ("GPE", "LOC")])
    df["year_mentions"] = year_mentions
    df["entities_people"] = ppl
    df["entities_orgs"] = orgs
    df["entities_locations"] = locs
    return df
