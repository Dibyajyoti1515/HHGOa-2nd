"""
scripts/run_ingestion.py

CLI entrypoint for the ingestion pipeline.

Usage:
    python -m scripts.run_ingestion --input path/to/data.parquet --lang en
"""

import argparse

from project.ingestion.pipeline import run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline.")
    parser.add_argument("--input", required=True, help="Path to source parquet file.")
    parser.add_argument("--lang", default="en", help="Language code tag for passage IDs (default: en).")
    args = parser.parse_args()

    run_ingestion(data_path=args.input, lang_code=args.lang)


if __name__ == "__main__":
    main()
