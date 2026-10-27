"""STEP 5 (BONUS) - WEB INTERFACE
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from config import CLEAN_PATH
from search import HybridSearcher
from summarize import Summarizer

st.set_page_config(page_title="Document Search & Summary", layout="wide")
PAGE_SIZE = 5


@st.cache_resource(show_spinner="Loading search index ...")
def load_searcher():
    return HybridSearcher()


@st.cache_resource
def load_summarizer(backend):
    return Summarizer(backend)


@st.cache_data
def build_suggestions(n=400):
    """Auto-suggest list = the most common two-word phrases in the corpus."""
    texts = pd.read_csv(CLEAN_PATH)["text"].tolist()
    vec = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=n)
    vec.fit(texts)
    return sorted(vec.vocabulary_.keys())


# ---------------- sidebar settings ----------------
st.sidebar.header("Settings")
method = st.sidebar.selectbox("Search method", ["hybrid", "bm25", "dense"])
length = st.sidebar.radio("Summary length", ["short", "medium", "long"], index=1)
backend = st.sidebar.radio("Summarizer", ["local", "api"],
                           help="'api' needs ANTHROPIC_API_KEY set")

# ---------------- main page ----------------
st.title("Document Search & Summarization")
searcher = load_searcher()

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Search the corpus", key="query",
                      placeholder="e.g. earthquake rescue efforts")

# Auto-suggestion: show phrases that match the last word being typed
if query:
    last = query.split()[-1].lower()
    hints = [s for s in build_suggestions() if s.startswith(last)][:5]
    if hints:
        st.caption("Suggestions:")
        cols = st.columns(len(hints))
        for col, h in zip(cols, hints):
            col.button(h, on_click=lambda h=h: st.session_state.update(query=h))

if query:
    results = searcher.search(query, top_n=50, method=method)

    # ---- summary of the top 5 results ----
    if st.button("Summarize top 5 results"):
        with st.spinner("Summarizing ..."):
            summary = load_summarizer(backend).summarize(
                [r["text"] for r in results[:5]], length)
        st.subheader("Summary")
        st.info(summary)

    # ---- paginated results ----
    pages = (len(results) - 1) // PAGE_SIZE + 1
    page = st.number_input("Page", 1, pages, 1)
    st.write(f"Showing results {(page-1)*PAGE_SIZE+1}-"
             f"{min(page*PAGE_SIZE, len(results))} of {len(results)}")
    for r in results[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]:
        with st.expander(f"Doc {r['doc_id']}  |  score {r['score']:.3f}"):
            st.markdown(f"**Best excerpt:** {r['excerpt']}")
            st.write(r["text"][:1500] + (" ..." if len(r["text"]) > 1500 else ""))
