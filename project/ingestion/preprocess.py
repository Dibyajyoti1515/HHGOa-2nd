"""
project/ingestion/preprocess.py

Text cleanup before chunking.
"""

import html
import re

import pandas as pd

from project.logging_system.logger import get_logger

logger = get_logger(__name__)


def preprocess_text(text: str) -> str:

    if not text:
        return ""

    text = html.unescape(text)

    text = (
        text
        .encode("utf-8", "ignore")
        .decode("utf-8", "ignore")
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    text = re.sub(
        r"www\.\S+",
        "",
        text,
    )

    text = re.sub(
        r"\[\d+\]",
        "",
        text,
    )

    text = re.sub(
        r"\s*\.\s*'\s*",
        ". ",
        text,
    )

    text = re.sub(
        r"…",
        " ",
        text,
    )

    text = re.sub(
        r"(\d)\s*…\s*(\d)",
        r"\1\2",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def is_valid_row(
    text: str,
    min_chars: int = 15,
) -> bool:

    return bool(text) and len(text.strip()) >= min_chars


def preprocess_corpus(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["text_en"] = df["text_en"].apply(
        preprocess_text
    )

    df["text"] = df["text"].apply(
        preprocess_text
    )

    before = len(df)

    df = df[
        df["text_en"].apply(is_valid_row)
    ].reset_index(drop=True)

    logger.info(
        "Validity filter: %s -> %s rows",
        f"{before:,}",
        f"{len(df):,}",
    )

    return df