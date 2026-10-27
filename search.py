"""STEP 2 - DOCUMENT SEARCH
Hybrid search = BM25 (keyword matching) + embeddings (meaning matching),
merged with Reciprocal Rank Fusion (RRF).
"""
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import CLEAN_PATH, EMB_PATH, EMBED_MODEL


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, remove stop words ('the', 'and', ...)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


def best_excerpt(text: str, query: str, n_sentences: int = 2) -> str:
    """Pick the sentence(s) that share the most words with the query."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    q = set(tokenize(query))
    scores = [len(q & set(tokenize(s))) for s in sents]
    best = max(range(len(sents)), key=lambda i: (scores[i], -i))
    return " ".join(sents[best:best + n_sentences])


def _ranks(scores: np.ndarray) -> np.ndarray:
    """Convert scores to ranks (0 = best)."""
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


class HybridSearcher:
    def __init__(self, df: pd.DataFrame | None = None):
        self.df = df if df is not None else pd.read_csv(CLEAN_PATH)
        self.texts = self.df["text"].tolist()

        # --- keyword index (BM25) ---
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

        # --- meaning index (embeddings) ---
        self.model = SentenceTransformer(EMBED_MODEL)
        self.emb = self._load_or_build_embeddings()

    def _load_or_build_embeddings(self) -> np.ndarray:
        # Embedding thousands of docs is slow, so we do it once and cache it.
        if EMB_PATH.exists():
            emb = np.load(EMB_PATH)
            if len(emb) == len(self.texts):
                return emb
        print("Building embeddings (one-time, a few minutes on CPU) ...")
        emb = self.model.encode(self.texts, batch_size=64, show_progress_bar=True,
                                normalize_embeddings=True)
        np.save(EMB_PATH, emb)
        return emb

    # ---- individual scorers -------------------------------------------
    def bm25_scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)))

    def dense_scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return self.emb @ q          # cosine similarity (vectors are normalized)

    def hybrid_scores(self, query: str, k: int = 60) -> np.ndarray:
        """Reciprocal Rank Fusion: documents ranked high by BOTH methods win."""
        r1 = _ranks(self.bm25_scores(query))
        r2 = _ranks(self.dense_scores(query))
        return 1 / (k + r1) + 1 / (k + r2)

    # ---- public API ----------------------------------------------------
    def search(self, query: str, top_n: int = 10, method: str = "hybrid") -> list[dict]:
        scorer = {"bm25": self.bm25_scores,
                  "dense": self.dense_scores,
                  "hybrid": self.hybrid_scores}[method]
        scores = scorer(query)
        top = np.argsort(-scores)[:top_n]
        return [{
            "doc_id": int(self.df.iloc[i]["doc_id"]),
            "score": float(scores[i]),
            "text": self.texts[i],
            "excerpt": best_excerpt(self.texts[i], query),
        } for i in top]
