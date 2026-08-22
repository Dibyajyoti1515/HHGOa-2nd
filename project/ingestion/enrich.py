"""
project/ingestion/enrich.py

Regex year extraction + spaCy NER enrichment.
"""

import re

import pandas as pd


YEAR_RE = re.compile(
    r"\b(1[89]\d{2}|20\d{2})\b"
)


def enrich(
    df: pd.DataFrame,
) -> pd.DataFrame:

    import spacy

    nlp = spacy.load(
        "en_core_web_sm"
    )

    year_mentions = []
    people = []
    organizations = []
    locations = []

    texts = (
        df["text_en"]
        .fillna("")
        .tolist()
    )

    for doc in nlp.pipe(
        texts,
        batch_size=256,
    ):

        year_mentions.append(
            [
                int(year)
                for year in YEAR_RE.findall(
                    doc.text
                )
            ]
        )

        people.append(
            [
                entity.text
                for entity in doc.ents
                if entity.label_ == "PERSON"
            ]
        )

        organizations.append(
            [
                entity.text
                for entity in doc.ents
                if entity.label_ == "ORG"
            ]
        )

        locations.append(
            [
                entity.text
                for entity in doc.ents
                if entity.label_ in ("GPE", "LOC")
            ]
        )

    df = df.copy()

    df["year_mentions"] = year_mentions
    df["entities_people"] = people
    df["entities_orgs"] = organizations
    df["entities_locations"] = locations

    return df