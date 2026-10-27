"""STEP 3 - SUMMARIZATION
Two backends:
  "local" : free, runs on your computer (distilbart). No API key needed.
  "api"   : Claude via the Anthropic API. Needs ANTHROPIC_API_KEY.

The user picks a length: "short", "medium" or "long".
"""
import os
from config import LOCAL_SUMMARY_MODEL, API_MODEL

# (min_tokens, max_tokens) for the local model, and target words for the API
LOCAL_LENGTHS = {"short": (25, 60), "medium": (60, 130), "long": (130, 250)}
API_WORDS = {"short": 40, "medium": 100, "long": 200}


class Summarizer:
    def __init__(self, backend: str = "local"):
        self.backend = backend
        self._pipe = None      # loaded lazily so the app starts fast

    # ---------------- local backend ----------------
    def _local(self, text: str, length: str) -> str:
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("summarization", model=LOCAL_SUMMARY_MODEL)
        lo, hi = LOCAL_LENGTHS[length]
        out = self._pipe(text, min_length=lo, max_length=hi,
                         truncation=True, do_sample=False)
        return out[0]["summary_text"].strip()

    # ---------------- API backend ------------------
    def _api(self, docs: list[str], length: str) -> str:
        import anthropic
        client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        joined = "\n\n".join(f"[Document {i+1}]\n{d[:3000]}" for i, d in enumerate(docs))
        prompt = (f"Summarize the key points of the documents below in about "
                  f"{API_WORDS[length]} words. Be coherent, factual, and do not "
                  f"add information that is not in the documents.\n\n{joined}")
        msg = client.messages.create(model=API_MODEL, max_tokens=800,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()

    # ---------------- public API -------------------
    def summarize(self, docs: list[str], length: str = "medium") -> str:
        """Summarize one or many documents."""
        if self.backend == "api":
            return self._api(docs, length)
        if len(docs) == 1:
            return self._local(docs[0], length)
        # Map-reduce: summarize each doc briefly, then summarize the summaries.
        partial = [self._local(d, "short") for d in docs]
        return self._local(" ".join(partial), length)
