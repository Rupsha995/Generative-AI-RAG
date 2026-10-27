"""STEP 1 - DATA PREPARATION
Download a news corpus, clean it, and save it as a CSV.

Run:  python data_prep.py
"""
import re
import pandas as pd
from datasets import load_dataset

from config import (CLEAN_PATH, DATASET_NAME, DATASET_VERSION,
                    N_DOCS, MIN_WORDS)


def clean_text(text: str) -> str:
    """Remove noise that would hurt search quality."""
    # "LONDON, England (CNN) -- " style bylines at the start
    text = re.sub(r"^.{0,120}?\(CNN\)\s*(--|-)?\s*", "", text, flags=re.S)
    text = re.sub(r"https?://\S+", " ", text)      # URLs
    text = re.sub(r"<[^>]+>", " ", text)           # leftover HTML tags
    text = re.sub(r"\s+", " ", text).strip()       # extra spaces / newlines
    return text


def main():
    print(f"Downloading {N_DOCS} documents ...")
    ds = load_dataset(DATASET_NAME, DATASET_VERSION, split=f"train[:{N_DOCS}]")

    df = pd.DataFrame({
        "doc_id": range(len(ds)),
        "text": [clean_text(a) for a in ds["article"]],
        # human-written highlights: we use them ONLY for evaluation
        "reference_summary": ds["highlights"],
    })

    before = len(df)
    df = df[df["text"].str.split().str.len() >= MIN_WORDS]   # drop short docs
    df = df.drop_duplicates(subset="text")                   # drop duplicates
    df = df.reset_index(drop=True)
    df["doc_id"] = df.index                                  # re-number cleanly

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Kept {len(df)} of {before} documents -> {CLEAN_PATH}")
    print(f"Average length: {df['text'].str.split().str.len().mean():.0f} words")


if __name__ == "__main__":
    main()
