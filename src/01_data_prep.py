"""Data preparation pipeline for Resume-JD matching.

This script downloads the `facehuggerapoorv/resume-jd-match` dataset from Hugging Face,
normalizes field names, performs text cleaning, and persists model-ready tabular data.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import DatasetDict, load_dataset


LOGGER = logging.getLogger("data_prep")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_TEXT_PATTERN = re.compile(r"[^a-zA-Z0-9\s\.\,\-\+\#\/]")
PAIR_TEXT_PATTERN = re.compile(
    r"job description\s*<<(.*?)>>\s*(?:and\s*)?(?:the\s+)?resume\s*:?\s*<<(.*?)>>\s*\.?\s*(?:the\s+result\s+is|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def configure_logging() -> None:
    """Configure process-wide logging for reproducible execution traces."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def clean_text(text: str) -> str:
    """Normalize free-form resume/JD text into a compact, model-friendly format."""
    normalized = text.strip().lower()
    normalized = NON_TEXT_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def _resolve_column_name(columns: Iterable[str], candidates: list[str], target: str) -> str:
    """Resolve a column name using candidate exact matches or keyword heuristics."""
    column_set = {col.lower(): col for col in columns}

    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]

    for lowered, original in column_set.items():
        if any(keyword in lowered for keyword in candidates):
            return original

    available = ", ".join(columns)
    raise ValueError(f"Could not infer `{target}` column from dataset columns: {available}")


def _coerce_label(raw_label: object) -> int:
    """Convert raw label values to binary integer classes."""
    if isinstance(raw_label, bool):
        return int(raw_label)
    if isinstance(raw_label, (int, float)):
        return int(raw_label > 0)

    normalized = str(raw_label).strip().lower()
    if normalized in {"0", "false", "no", "no fit", "not fit", "irrelevant", "mismatch"}:
        return 0
    if normalized in {
        "1",
        "true",
        "match",
        "matched",
        "yes",
        "relevant",
        "fit",
        "potential fit",
        "good fit",
        "strong fit",
    }:
        return 1
    return int("fit" in normalized and "no fit" not in normalized)


def _parse_pair_from_prompt(text: str) -> tuple[str, str]:
    """Extract JD and resume segments from prompt-style dataset rows."""
    match = PAIR_TEXT_PATTERN.search(text)
    if match:
        jd_text = match.group(1).strip()
        resume_text = match.group(2).strip()
        return resume_text, jd_text

    raise ValueError("Unable to parse resume/JD pair from prompt text format.")


def prepare_dataframe(dataset_name: str) -> pd.DataFrame:
    """Download and convert raw dataset into a canonical three-column frame."""
    raw = load_dataset(dataset_name)
    if isinstance(raw, DatasetDict):
        if "train" in raw:
            frame = raw["train"].to_pandas()
        else:
            first_split = next(iter(raw.keys()))
            frame = raw[first_split].to_pandas()
            LOGGER.warning("No train split found. Falling back to split: %s", first_split)
    else:
        frame = raw.to_pandas()

    if frame.empty:
        raise ValueError("Loaded dataset is empty. Aborting pipeline.")

    lower_cols = {c.lower() for c in frame.columns}
    if {"resume_text", "jd_text", "label"}.issubset(lower_cols):
        resume_col = _resolve_column_name(
            columns=frame.columns,
            candidates=["resume_text", "resume", "cv", "candidate_profile"],
            target="resume",
        )
        jd_col = _resolve_column_name(
            columns=frame.columns,
            candidates=["jd_text", "job_description", "jd", "description", "job_desc"],
            target="job_description",
        )
        label_col = _resolve_column_name(
            columns=frame.columns,
            candidates=["label", "match", "is_match", "target", "matched"],
            target="label",
        )

        cleaned = pd.DataFrame(
            {
                "resume_text": frame[resume_col].astype(str).map(clean_text),
                "jd_text": frame[jd_col].astype(str).map(clean_text),
                "label": frame[label_col].map(_coerce_label).astype(int),
            }
        )
    elif {"text", "label"}.issubset(lower_cols):
        text_col = _resolve_column_name(columns=frame.columns, candidates=["text"], target="text")
        label_col = _resolve_column_name(
            columns=frame.columns,
            candidates=["label", "match", "is_match", "target", "matched"],
            target="label",
        )

        parsed_pairs = frame[text_col].astype(str).map(_parse_pair_from_prompt)
        cleaned = pd.DataFrame(
            {
                "resume_text": parsed_pairs.map(lambda item: clean_text(item[0])),
                "jd_text": parsed_pairs.map(lambda item: clean_text(item[1])),
                "label": frame[label_col].map(_coerce_label).astype(int),
            }
        )
    else:
        available = ", ".join(frame.columns)
        raise ValueError(
            "Unsupported dataset schema. Expected either (resume_text, jd_text, label) "
            f"or (text, label). Found columns: {available}"
        )
    cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)
    cleaned = cleaned[(cleaned["resume_text"].str.len() > 20) & (cleaned["jd_text"].str.len() > 20)]
    cleaned["sample_id"] = cleaned.index

    if cleaned["label"].nunique() < 2:
        raise ValueError("Prepared dataset has a single class only; model training requires binary classes.")

    LOGGER.info("Prepared %d usable records.", len(cleaned))
    LOGGER.info("Class distribution: %s", cleaned["label"].value_counts(normalize=True).to_dict())
    return cleaned


def save_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    """Persist prepared artifacts in both CSV and Parquet for interoperability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "resume_jd_clean.csv"
    parquet_path = output_dir / "resume_jd_clean.parquet"

    df.to_csv(csv_path, index=False)
    LOGGER.info("Saved CSV: %s", csv_path)
    try:
        df.to_parquet(parquet_path, index=False)
        LOGGER.info("Saved Parquet: %s", parquet_path)
    except ImportError as exc:
        LOGGER.warning("Parquet export skipped because optional engine is unavailable: %s", exc)


def parse_args() -> argparse.Namespace:
    """Parse script CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare Resume-JD dataset for model training.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="facehuggerapoorv/resume-jd-match",
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where cleaned files are stored.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute end-to-end data preparation."""
    configure_logging()
    args = parse_args()
    prepared_df = prepare_dataframe(dataset_name=args.dataset_name)
    save_outputs(df=prepared_df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
