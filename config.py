"""Central settings. Change values here instead of hunting through the code."""
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CLEAN_PATH = DATA_DIR / "clean_docs.csv"      # cleaned corpus
EMB_PATH = DATA_DIR / "embeddings.npy"        # cached document embeddings

DATASET_NAME = "abisee/cnn_dailymail"         # news articles + human-written highlights
DATASET_VERSION = "3.0.0"
N_DOCS = 5000                                 # corpus size (raise for a bigger corpus)
MIN_WORDS = 80                                # drop very short articles

EMBED_MODEL = "all-MiniLM-L6-v2"              # small, fast sentence-embedding model
LOCAL_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"   # light summarization model
API_MODEL = "claude-sonnet-5-5"               # used only if backend="api"

RANDOM_SEED = 42
