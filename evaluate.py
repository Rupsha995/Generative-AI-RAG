"""STEP 4 - EVALUATION
(a) Search accuracy: Hit@1, Hit@5, Hit@10, MRR@10 for BM25 vs dense vs hybrid.
(b) Summary quality: ROUGE-1/2/L vs human highlights + a human-rating sheet.

Run:  python evaluate.py                 (full)
      python evaluate.py --skip-summary  (search only, much faster)
"""
import argparse
import time
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from config import CLEAN_PATH, RESULTS_DIR, RANDOM_SEED
from search import HybridSearcher
from summarize import Summarizer


def make_test_set(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Random subset. Query = first human highlight of the doc.
    The highlight is NOT part of the indexed text, so there is no cheating."""
    test = df.sample(n=min(n, len(df)), random_state=RANDOM_SEED).copy()
    test["query"] = test["reference_summary"].str.split("\n").str[0].str.strip()
    return test


def evaluate_search(searcher, test, top_k=10) -> pd.DataFrame:
    rows = []
    for method in ["bm25", "dense", "hybrid"]:
        ranks, t0 = [], time.time()
        for _, row in test.iterrows():
            results = searcher.search(row["query"], top_n=top_k, method=method)
            ids = [r["doc_id"] for r in results]
            ranks.append(ids.index(row["doc_id"]) + 1 if row["doc_id"] in ids else None)
        ms = (time.time() - t0) / len(test) * 1000
        found = [r for r in ranks if r is not None]
        rows.append({
            "method": method,
            "Hit@1": sum(r == 1 for r in found) / len(ranks),
            "Hit@5": sum(r <= 5 for r in found) / len(ranks),
            "Hit@10": len(found) / len(ranks),
            "MRR@10": sum(1 / r for r in found) / len(ranks),
            "ms/query": round(ms, 1),
        })
    return pd.DataFrame(rows).round(3)


def evaluate_summaries(test, n=20, backend="local") -> pd.DataFrame:
    summarizer = Summarizer(backend)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for _, row in test.head(n).iterrows():
        summary = summarizer.summarize([row["text"]], "medium")
        s = scorer.score(row["reference_summary"], summary)
        rows.append({"doc_id": row["doc_id"], "summary": summary,
                     "rouge1": s["rouge1"].fmeasure,
                     "rouge2": s["rouge2"].fmeasure,
                     "rougeL": s["rougeL"].fmeasure})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-summaries", type=int, default=20)
    ap.add_argument("--backend", default="local", choices=["local", "api"])
    ap.add_argument("--skip-summary", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CLEAN_PATH)
    test = make_test_set(df, args.n_test)
    searcher = HybridSearcher(df)

    print("\n=== SEARCH EVALUATION ===")
    metrics = evaluate_search(searcher, test)
    print(metrics.to_string(index=False))
    metrics.to_csv(RESULTS_DIR / "retrieval_metrics.csv", index=False)

    if args.skip_summary:
        return

    print("\n=== SUMMARY EVALUATION (ROUGE) ===")
    summ = evaluate_summaries(test, args.n_summaries, args.backend)
    print(summ[["rouge1", "rouge2", "rougeL"]].mean().round(3).to_string())
    summ.to_csv(RESULTS_DIR / "summary_rouge.csv", index=False)

    # Human evaluation: open this file and fill in scores from 1 (bad) to 5 (great)
    human = summ[["doc_id", "summary"]].copy()
    for col in ["coherence_1_5", "relevance_1_5", "fluency_1_5"]:
        human[col] = ""
    human.to_csv(RESULTS_DIR / "human_eval_template.csv", index=False)
    print(f"\nHuman rating sheet saved to {RESULTS_DIR/'human_eval_template.csv'}")


if __name__ == "__main__":
    main()
