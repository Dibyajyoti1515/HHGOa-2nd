"""
ingestion/chunking/tagging.py

Ported from Cell 7 of Update_version_2.ipynb -- tag_metadata_aware.
Derives has_year/has_person/has_org/has_location boolean flags from
the entity lists produced by ingestion/enrich.py. Logic unchanged.
"""

import pandas as pd


def tag_metadata_aware(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_year"] = df["year_mentions"].apply(lambda y: len(y) > 0)
    df["has_person"] = df["entities_people"].apply(lambda p: len(p) > 0)
    df["has_org"] = df["entities_orgs"].apply(lambda o: len(o) > 0)
    df["has_location"] = df["entities_locations"].apply(lambda l: len(l) > 0)
    return df
