# pages/Document.py
# Document Topics (Pro) — robust df cutoffs + semantic analysis + SAFE confidence
# Patch: hide sklearn "After pruning, no terms remain" stack trace; show tip instead.

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Optional libs
HAVE_BERTOPIC = False
HAVE_SBERT = False
HAVE_BM25 = False
try:
    from bertopic import BERTopic
    HAVE_BERTOPIC = True
except Exception:
    pass

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SBERT = True
except Exception:
    pass

try:
    from rank_bm25 import BM25Okapi
    HAVE_BM25 = True
except Exception:
    pass

# Fallback models
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import normalize

# =========================== Helpers ===========================
TIP_MSG = "After pruning, no terms remain. Try a lower min_df or a higher max_df."

def get_consolidated_df() -> pd.DataFrame | None:
    df1 = st.session_state.get("consolidated_df")
    if isinstance(df1, pd.DataFrame) and not df1.empty:
        return df1
    df2 = st.session_state.get("consolidated")
    if isinstance(df2, pd.DataFrame) and not df2.empty:
        return df2
    return None

def fold_accents(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\t", " ")
    s = (s.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
           .replace("\u2013","-").replace("\u2014","-").replace("\u2015","-"))
    s = s.replace("-", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sent_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text))
    return [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", text) if s.strip()]

def pick_text_column(df: pd.DataFrame) -> str:
    prefs = [c for c in ["Findings of the evaluation", "full_text", "text"] if c in df.columns]
    return prefs[0] if prefs else df.columns[0]

# Stopwords (EN/ES/FR/PT + evaluation boilerplate)
EN_SW = {"the","of","and","to","in","for","with","on","at","by","from","as","that","this","it","is","are","was","were","be","been",
         "an","a","or","if","but","can","will","may","we","our","their","they","these","those","which","within","across","into"}
ES_SW = {"de","la","el","los","las","y","en","que","del","con","para","por","una","un","se","al","como","su","sus"}
FR_SW = {"de","la","le","les","et","en","que","des","du","aux","dans","pour","par","une","un","au","sur","ses","leurs"}
PT_SW = {"de","da","do","das","dos","e","em","que","para","por","com","uma","um","se","ao","os","as","suas","seus"}
DOMAIN_SW = {"evaluation","evaluations","evaluated","findings","conclusions","recommendations","chapter","annex","abstract",
             "coherence","relevance","effectiveness","efficiency","sustainability","connectedness",
             "figure","fig","table","annexe","appendix","ix","ii","iii","iv","vi","vii","viii","x","xi","xii"}

def make_stopwords(include_es=True, include_fr=True, include_pt=True, extra: List[str] | None = None) -> List[str]:
    sw = set(EN_SW) | set(DOMAIN_SW)
    if include_es: sw |= ES_SW
    if include_fr: sw |= FR_SW
    if include_pt: sw |= PT_SW
    if extra: sw |= {fold_accents(x) for x in extra if x.strip()}
    return sorted(sw)

ROMAN_RE = re.compile(r"\b(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b", re.I)
NUMERIC_RE = re.compile(r"\b\d{1,4}\b")
HEADER_PAT = re.compile(r"\b(findings|conclusions|recommendations|chapter\s+\d+|annex|appendix)\b", re.I)

def detox_text(s: str, remove_numbers=True, remove_roman=True, drop_headers=True) -> str:
    t = fold_accents(s)
    if drop_headers: t = HEADER_PAT.sub(" ", t)
    if remove_roman: t = ROMAN_RE.sub(" ", t)
    if remove_numbers: t = NUMERIC_RE.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_keyword_windows(series_text: pd.Series, keywords: List[str],
                            window: int = 1, whole_word: bool = True, dedup_per_doc: bool = True
                           ) -> Tuple[List[str], List[int]]:
    snippets, idx_ref = [], []
    pats = []
    for kw in [fold_accents(k) for k in keywords if k.strip()]:
        pats.append(re.compile(rf"\b{re.escape(kw)}\b") if whole_word else re.compile(re.escape(kw)))
    for i, txt in series_text.items():
        if not str(txt).strip() or not pats: continue
        sents = sent_split(txt); s_norm = [fold_accents(s) for s in sents]
        seen = set()
        for j, ns in enumerate(s_norm):
            if any(p.search(ns) for p in pats):
                s = max(0, j-window); e = min(len(sents), j+window+1)
                snip = " ".join(sents[s:e]).strip()
                key = fold_accents(snip)[:500]
                if dedup_per_doc and key in seen: continue
                seen.add(key); snippets.append(snip); idx_ref.append(i)
    return snippets, idx_ref

def bm25_prefilter(texts: List[str], keywords: List[str], keep_top_n: int = 400) -> List[int]:
    if not HAVE_BM25 or not keywords or not texts: return list(range(len(texts)))
    toks = [re.findall(r"[a-z0-9]+", fold_accents(t)) for t in texts]
    bm25 = BM25Okapi(toks)
    q = re.findall(r"[a-z0-9]+", fold_accents(" ".join(keywords)))
    scores = bm25.get_scores(q)
    return np.argsort(-np.array(scores))[: min(keep_top_n, len(scores))].tolist()

def ensure_k_valid(k: int, n_items: int) -> int:
    return max(2, min(k, max(2, n_items)))

# --- clamp df params to avoid sklearn InvalidParameterError ---
def clamp_df_params(n_items: int, min_df_int: int, max_df_pct: int) -> tuple[float, float]:
    n = max(int(n_items), 1)
    min_prop = float(min_df_int) / float(n)
    min_prop = max(0.0, min(min_prop, 0.95))
    max_prop = float(max_df_pct) / 100.0
    eps = 1e-3
    max_prop = max(min_prop + eps, min(max_prop, 0.999))
    if not (0.0 <= max_prop <= 1.0):
        max_prop = 0.999
    if max_prop <= min_prop:
        max_prop = min(0.999, min_prop + eps)
    return float(min_prop), float(max_prop)

# ---- TIP-ONLY error wrapper (prevents red stack trace) ----
def fit_with_tip(vectorizer, texts: List[str], which: str):
    """Call vectorizer.fit_transform(texts). If sklearn prunes everything, show a tip and stop."""
    try:
        X = vectorizer.fit_transform(texts)
        return X
    except ValueError as e:
        if "After pruning, no terms remain" in str(e):
            st.info(f"💡 {TIP_MSG}")
            st.stop()
        raise  # any other error: let existing handlers deal with it

# =========================== UI ===========================
st.set_page_config(page_title="Document Topics (Pro)", page_icon="🧩", layout="wide")
st.title("🧩 Document Topics (Pro) — from session")

df = get_consolidated_df()
if df is None:
    st.error("No consolidated dataframe in session. Run the main page first.")
    st.stop()

with st.expander("Preview consolidated dataframe", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# Sidebar
with st.sidebar:
    st.header("Corpus")
    text_col = st.selectbox("Text column", options=list(df.columns), index=list(df.columns).index(pick_text_column(df)))
    scope = st.radio("Scope", ["Whole documents", "Keyword-centered (sentence ± window)"], index=0)

    keywords = []; whole_word = True; window = 1
    if scope.startswith("Keyword"):
        kw_text = st.text_area("Keywords (comma or newline separated)",
                               value="teacher training, professional development, learning outcomes, curriculum", height=110)
        keywords = [w.strip() for w in re.split(r"[,\n]", kw_text) if w.strip()]
        whole_word = st.checkbox("Whole-word boundary", value=True)
        window = st.slider("Sentence window (±)", 1, 3, 1)

    st.markdown("---")
    st.header("Cleaning")
    drop_headers = st.checkbox("Drop section headers (findings/conclusions/annex…)", value=True)
    strip_numbers = st.checkbox("Strip numbers", value=True)
    strip_roman = st.checkbox("Strip roman numerals (I, II, IV…)", value=True)
    inc_es = st.checkbox("Spanish stopwords", value=True)
    inc_fr = st.checkbox("French stopwords", value=True)
    inc_pt = st.checkbox("Portuguese stopwords", value=True)
    extra_sw = st.text_area("Extra stopwords (comma/newline)", value="", height=80)

    st.markdown("---")
    st.header("Prefilter")
    use_bm25 = st.checkbox("BM25 prefilter by keywords (focus corpus)", value=True if scope.startswith("Keyword") else False)
    keep_top_n = st.slider("Keep top-N items", 100, 2000, 600, 50)

    st.markdown("---")
    st.header("Model")
    model_choice = st.selectbox("Topic model",
                                ["BERTopic (SBERT)", "NMF (TF-IDF)", "LDA (Count)"],
                                index=0 if HAVE_BERTOPIC else 1)
    k = st.slider("Number of topics (NMF/LDA) or reduction target (BERTopic)", 2, 50, 12, 1)
    ngram_max = st.slider("N-grams", 1, 3, 2, 1)
    min_df = st.slider("min_df (drop rare terms)", 1, 20, 2, 1)
    max_df = st.slider("max_df (%)", 50, 100, 95, 1)
    max_features = st.slider("Max features", 2000, 30000, 12000, 1000)

    if model_choice.startswith("BERTopic"):
        min_topic_size = st.slider("min_topic_size (BERTopic)", 5, 200, 30, 5)
        reduce_topics = st.checkbox("Reduce topics to ~k", value=True)

    st.markdown("---")
    show_doc_table = st.checkbox("Show per-item assignments", value=False)

# Build corpus
raw_series = df[text_col].fillna("").astype(str)
meta = pd.DataFrame({
    "row_id": np.arange(len(df)),
    "filename": df.get("filename", pd.Series(np.arange(len(df)).astype(str))),
    "year": df.get("year", pd.Series([np.nan]*len(df))),
    "region": (df["unicef_region"] if "unicef_region" in df.columns
               else (df["region"] if "region" in df.columns else pd.Series(["Unknown"]*len(df))))
})

if scope.startswith("Keyword"):
    if not keywords:
        st.warning("Enter at least one keyword for keyword-centered mode."); st.stop()
    snippets, idx_ref = extract_keyword_windows(raw_series, keywords, window=window, whole_word=whole_word, dedup_per_doc=True)
    corpus_df = pd.DataFrame({"text": snippets, "source_row": idx_ref}).merge(meta, left_on="source_row", right_on="row_id", how="left")
    if corpus_df.empty:
        st.error("No keyword-centered snippets found. Try different keywords or increase window."); st.stop()
    st.success(f"Built keyword-centered corpus with **{len(corpus_df)}** snippets from **{corpus_df['filename'].nunique()}** documents.")
else:
    corpus_df = pd.DataFrame({"text": raw_series, "source_row": np.arange(len(raw_series))}).merge(meta, left_on="source_row", right_on="row_id", how="left")
    corpus_df = corpus_df[corpus_df["text"].str.len() >= 40]
    st.success(f"Built whole-document corpus with **{len(corpus_df)}** items.")

# Cleaning
sw = make_stopwords(inc_es, inc_fr, inc_pt, extra=[x for x in re.split(r"[,\n]", extra_sw) if x.strip()])
corpus_df["clean"] = corpus_df["text"].map(lambda x: detox_text(x, strip_numbers, strip_roman, drop_headers))
corpus_df = corpus_df[corpus_df["clean"].str.len() >= 20].reset_index(drop=True)
if corpus_df.empty:
    st.error("Corpus is empty after cleaning. Relax cleaning or pick a richer column (e.g., full_text)."); st.stop()

# Optional BM25 prefilter
if use_bm25 and ((scope.startswith("Keyword") and keywords) or (not scope.startswith("Keyword") and keywords)):
    keep_idx = bm25_prefilter(corpus_df["clean"].tolist(), keywords, keep_top_n=keep_top_n) if HAVE_BM25 else list(range(len(corpus_df)))
    corpus_df = corpus_df.iloc[keep_idx].reset_index(drop=True)
    st.caption(f"BM25 prefilter applied → kept **{len(corpus_df)}** items.")

st.caption(f"Final corpus size: **{len(corpus_df)}** | Avg length: **{int(corpus_df['clean'].str.len().mean())}** chars")

# =========================== Modeling ===========================
topics_df = None
doc_table = None
label_map: Dict[int, str] = {}

def run_bertopic(corpus_df: pd.DataFrame,
                 min_topic_size: int,
                 ngram_max: int,
                 min_df_prop: float,
                 max_df_prop: float,
                 sw: List[str],
                 target_k: int | None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int,str], object, object]:
    texts = corpus_df["clean"].tolist()
    n_items = len(texts)
    if n_items < 5:
        raise RuntimeError("too_small_for_bertopic")

    safe_neighbors = max(2, min(15, n_items - 1))
    hdb_min_cluster = max(2, min(min_topic_size, max(2, n_items // 2)))
    hdb_min_samples = max(1, min(hdb_min_cluster // 2, n_items - 1))

    umap_model = None
    hdb_model = None
    try:
        from umap import UMAP
        umap_model = UMAP(n_neighbors=safe_neighbors, n_components=5, random_state=0, metric="cosine")
    except Exception:
        pass
    try:
        from hdbscan import HDBSCAN
        hdb_model = HDBSCAN(min_cluster_size=hdb_min_cluster,
                            min_samples=hdb_min_samples,
                            metric="euclidean",
                            prediction_data=True)
    except Exception:
        pass

    embed_model = None
    if HAVE_SBERT:
        try:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            embed_model = None

    vectorizer = CountVectorizer(
        stop_words=sw if sw else None,
        ngram_range=(1, ngram_max),
        min_df=min_df_prop,
        max_df=max_df_prop
    )

    topic_model = BERTopic(
        embedding_model=embed_model,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        min_topic_size=hdb_min_cluster,
        calculate_probabilities=True,
        verbose=False,
    )

    topics, probs = topic_model.fit_transform(texts)

    if target_k is not None:
        try:
            topic_model = topic_model.reduce_topics(texts, topics, nr_topics=target_k)
            topics, probs = topic_model.transform(texts)
        except Exception:
            pass

    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].copy()
    info = info.rename(columns={"Topic":"topic", "Count":"count", "Name":"name"})
    topics_df = info[["topic","name","count"]].reset_index(drop=True)

    doc_table = corpus_df[["filename","year","region","text","clean"]].copy()
    doc_table["topic"] = topics
    label_map = {int(r["topic"]): r["name"] for _, r in topics_df.iterrows()}
    doc_table["topic_label"] = doc_table["topic"].map(label_map)

    return topics_df, doc_table, label_map, topics, probs

# Fit model(s)
model_choice_is_bertopic = (model_choice.startswith("BERTopic") and HAVE_BERTOPIC)
if model_choice_is_bertopic:
    min_prop, max_prop = clamp_df_params(len(corpus_df), min_df, max_df)
    try:
        topics_df, doc_table, label_map, topics_raw, probs = run_bertopic(
            corpus_df, min_topic_size, ngram_max, min_prop, max_prop, sw, target_k=(k if 'reduce_topics' in locals() and reduce_topics else None)
        )
    except RuntimeError as e:
        if "too_small_for_bertopic" in str(e):
            st.warning("BERTopic skipped: corpus too small. Falling back to NMF.")
            model_choice_is_bertopic = False
        else:
            st.warning("Retrying BERTopic with permissive settings (n_neighbors<=10, min_cluster=2, min_samples=1).")
            try:
                topics_df, doc_table, label_map, topics_raw, probs = run_bertopic(
                    corpus_df, min_topic_size=2, ngram_max=ngram_max,
                    min_df_prop=1.0/len(corpus_df), max_df_prop=0.999, sw=sw,
                    target_k=(k if 'reduce_topics' in locals() and reduce_topics else None)
                )
            except Exception:
                st.warning("BERTopic still failed; falling back to NMF.")
                model_choice_is_bertopic = False
    except Exception:
        st.warning("BERTopic failed; falling back to NMF.")
        model_choice_is_bertopic = False

if not model_choice_is_bertopic:
    n_items = len(corpus_df)
    k = ensure_k_valid(k, n_items); st.caption(f"Using k = {k} topics on {n_items} items.")
    min_prop, max_prop = clamp_df_params(n_items, min_df, max_df)

    if model_choice.startswith("LDA"):
        vect = CountVectorizer(
            lowercase=True, strip_accents="unicode",
            stop_words=sw if sw else None,
            ngram_range=(1, ngram_max),
            min_df=min_prop, max_df=max_prop, max_features=max_features
        )
        # TIP-ONLY wrapper here:
        X = fit_with_tip(vect, corpus_df["clean"].astype(str).tolist(), which="LDA")
        lda = LatentDirichletAllocation(n_components=k, random_state=0, max_iter=30, learning_method="batch")
        W = lda.fit_transform(X); H = lda.components_; terms = vect.get_feature_names_out()
    else:
        vect = TfidfVectorizer(
            lowercase=True, strip_accents="unicode",
            stop_words=sw if sw else None,
            ngram_range=(1, ngram_max),
            min_df=min_prop, max_df=max_prop, max_features=max_features
        )
        # TIP-ONLY wrapper here:
        X = fit_with_tip(vect, corpus_df["clean"].astype(str).tolist(), which="NMF")
        nmf = NMF(n_components=k, init="nndsvda", random_state=0, max_iter=400)
        W = nmf.fit_transform(X); H = nmf.components_; terms = vect.get_feature_names_out()

    rows = []
    topn = 12
    for t in range(k):
        idx = np.argsort(-H[t])[:topn]
        words = [terms[j] for j in idx]
        label = ", ".join(words[:8])
        rows.append({"topic": t, "name": label, "count": int((W.argmax(axis=1) == t).sum())})
    topics_df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)

    doc_table = corpus_df[["filename","year","region","text","clean"]].copy()
    doc_table["topic"] = W.argmax(axis=1)
    label_map = {int(r.topic): r.name for _, r in topics_df.iterrows()}
    doc_table["topic_label"] = doc_table["topic"].map(label_map)
    doc_table["topic_confidence"] = W.max(axis=1).round(4)

# =========================== Confidence (safe for any BERTopic probs) ===========================
if model_choice_is_bertopic:
    def probs_to_conf(probs_obj, n_items: int) -> np.ndarray | None:
        if probs_obj is None:
            return None
        try:
            arr = np.asarray(probs_obj, dtype=float)
        except Exception:
            arr = None
        if arr is not None:
            if arr.ndim == 2 and arr.shape[0] == n_items:
                return arr.max(axis=1)
            if arr.ndim == 1 and arr.shape[0] == n_items:
                return arr
        try:
            vals = []
            for p in probs_obj:
                if p is None: vals.append(np.nan); continue
                pa = np.asarray(p).ravel()
                vals.append(float(np.max(pa)) if pa.size else np.nan)
            if len(vals) == n_items:
                return np.array(vals, dtype=float)
        except Exception:
            pass
        return None

    conf = probs_to_conf(locals().get("probs", None), n_items=len(corpus_df))
    if conf is not None:
        doc_table["topic_confidence"] = np.nan_to_num(conf, nan=np.nan).round(4)

# =========================== Outputs & Charts ===========================
st.subheader(f"Topics ({'BERTopic' if model_choice_is_bertopic else ('LDA' if model_choice.startswith('LDA') else 'NMF')})")
st.dataframe(topics_df, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.download_button("⬇️ Download topics (CSV)",
        data=topics_df.to_csv(index=False).encode("utf-8"),
        file_name="topics_summary.csv", mime="text/csv", use_container_width=True)
with c2:
    out_docs = doc_table[["filename","year","region","topic","topic_label","topic_confidence"]].copy()
    st.download_button("⬇️ Download per-item topics (CSV)",
        data=out_docs.to_csv(index=False).encode("utf-8"),
        file_name="document_topics.csv", mime="text/csv", use_container_width=True)

st.subheader("Charts")
size_df = topics_df.copy()
st.altair_chart(
    alt.Chart(size_df).mark_bar().encode(
        x=alt.X("count:Q", title="Items"),
        y=alt.Y("name:N", sort="-x", title="Topic"),
        tooltip=["topic","name","count"]
    ).properties(height=max(240, 24*len(size_df))),
    use_container_width=True
)

if "year" in doc_table.columns:
    year_df = (doc_table.assign(Year=pd.to_numeric(doc_table["year"], errors="coerce"))
               .dropna(subset=["Year"])
               .groupby(["Year","topic_label"], as_index=False).size()
               .rename(columns={"size":"count"}))
    if not year_df.empty:
        st.markdown("**Distribution by Year**")
        st.altair_chart(
            alt.Chart(year_df).mark_bar().encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("sum(count):Q", title="Items"),
                color=alt.Color("topic_label:N", title="Topic"),
                tooltip=["Year","topic_label","sum(count)"]
            ).properties(height=320),
            use_container_width=True
        )

if "region" in doc_table.columns:
    reg_df = (doc_table.copy().groupby(["region","topic_label"], as_index=False).size()
              .rename(columns={"size":"count"}))
    if not reg_df.empty:
        st.markdown("**Distribution by Region**")
        st.altair_chart(
            alt.Chart(reg_df).mark_bar().encode(
                x=alt.X("region:N", sort="-y", title="Region"),
                y=alt.Y("sum(count):Q", title="Items"),
                color=alt.Color("topic_label:N", title="Topic"),
                tooltip=["region","topic_label","sum(count)"]
            ).properties(height=320),
            use_container_width=True
        )

# =========================== Semantic Analysis ===========================
st.subheader("Semantic Analysis (Embeddings)")
@st.cache_resource(show_spinner=False)
def _load_sbert():
    if not HAVE_SBERT: return None
    try: return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception: return None

@st.cache_data(show_spinner=False)
def embed_corpus_sbert(texts: List[str]) -> np.ndarray:
    model = _load_sbert()
    if model is None: return np.empty((0,0))
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

@st.cache_data(show_spinner=False)
def embed_corpus_tfidf(texts: List[str]):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9, strip_accents="unicode")
    # TIP-ONLY wrapper here:
    X = fit_with_tip(vec, texts, which="TFIDF-embeddings")
    X = normalize(X)
    return X.toarray().astype(np.float32), vec

@st.cache_data(show_spinner=False)
def embed_query_tfidf(query: str, vectorizer: TfidfVectorizer):
    q = vectorizer.transform([query]); q = normalize(q); return q.toarray().astype(np.float32)[0]

corpus_texts = corpus_df["clean"].tolist()
engine = "SBERT" if _load_sbert() is not None else "TF-IDF"
st.caption(f"Embedding engine: **{engine}**")

if engine == "SBERT":
    EMB = embed_corpus_sbert(corpus_texts)
    if EMB.size == 0:
        EMB, tfidf_vec = embed_corpus_tfidf(corpus_texts); engine = "TF-IDF"
else:
    EMB, tfidf_vec = embed_corpus_tfidf(corpus_texts)

topic_ids = doc_table["topic"].to_numpy()

def topic_centroids(emb: np.ndarray, topic_ids: np.ndarray) -> Dict[int, np.ndarray]:
    cents = {}
    for tid in np.unique(topic_ids):
        mask = (topic_ids == tid)
        if mask.sum() == 0: continue
        c = emb[mask].mean(axis=0); n = np.linalg.norm(c) + 1e-12
        cents[int(tid)] = (c / n).astype(np.float32)
    return cents

centroids = topic_centroids(EMB, topic_ids)

rows = []
for tid, cen in centroids.items():
    mask = (topic_ids == tid); sims = (EMB[mask] @ cen)
    rows.append({"topic": tid, "label": (label_map.get(int(tid), f"Topic {tid}")),
                 "n_items": int(mask.sum()), "mean_sim": float(np.mean(sims)),
                 "median_sim": float(np.median(sims)), "std_sim": float(np.std(sims))})
sem_quality = pd.DataFrame(rows).sort_values("mean_sim", ascending=False).reset_index(drop=True)
st.markdown("**Topic semantic quality (cosine to centroid)**")
st.dataframe(sem_quality, use_container_width=True)

st.markdown("**Top representative documents per topic**")
pick_tid = st.selectbox("Choose a topic",
                        options=[int(t) for t in sem_quality["topic"].tolist()],
                        format_func=lambda t: f"{t} — {label_map.get(int(t), f'Topic {t}')}")
def top_docs_for_topic(tid: int, k: int = 10):
    cen = centroids.get(int(tid));
    if cen is None: return pd.DataFrame()
    mask = (topic_ids == tid); idxs = np.where(mask)[0]
    sims = (EMB[idxs] @ cen); order = np.argsort(-sims)[: min(k, len(sims))]
    out = doc_table.iloc[idxs[order]].copy()
    out.insert(0, "similarity", sims[order].round(4))
    out["snippet"] = out["text"].astype(str).str.slice(0, 240)
    return out[["similarity","filename","year","region","snippet"]]
st.dataframe(top_docs_for_topic(int(pick_tid), k=10), use_container_width=True, height=360)

st.markdown("**Topic ↔ Topic semantic similarity**")
if len(centroids) >= 2:
    tids = sorted(centroids.keys()); C = np.stack([centroids[t] for t in tids], axis=0)
    sim_mat = C @ C.T
    hm = [{"t_i": f"{tids[i]}", "t_j": f"{tids[j]}", "sim": float(sim_mat[i,j])}
          for i in range(len(tids)) for j in range(len(tids))]
    heat = alt.Chart(pd.DataFrame(hm)).mark_rect().encode(
        x=alt.X("t_i:N", title="Topic"), y=alt.Y("t_j:N", title="Topic"),
        color=alt.Color("sim:Q", title="Cosine"), tooltip=["t_i","t_j","sim"]
    ).properties(height=300)
    st.altair_chart(heat, use_container_width=True)
else:
    st.info("Not enough topics to compute a similarity heatmap.")

st.markdown("**Semantic search over corpus**")
col_q1, col_q2 = st.columns([3,1])
with col_q1:
    query = st.text_input("Enter a query (e.g., 'teacher professional development in rural areas')", "")
with col_q2:
    topk = st.slider("Top-k", 5, 50, 12, 1)

if query.strip():
    if engine == "SBERT":
        q_vec = _load_sbert().encode([fold_accents(query)], normalize_embeddings=True)[0].astype(np.float32)
    else:
        q_vec = embed_query_tfidf(fold_accents(query), tfidf_vec)
    sims = EMB @ q_vec; order = np.argsort(-sims)[: topk]
    qres = doc_table.iloc[order].copy()
    qres.insert(0, "similarity", sims[order].round(4))
    qres["snippet"] = qres["text"].astype(str).str.slice(0, 240)
    st.dataframe(qres[["similarity","filename","year","region","topic_label","snippet"]],
                 use_container_width=True, height=420)
else:
    st.caption("Type a query above to see the most semantically similar items.")

st.caption("If BERTopic errors on small corpora, this page now auto-tunes neighbors/cluster sizes, retries permissively, and falls back to NMF so you always get topics.")
