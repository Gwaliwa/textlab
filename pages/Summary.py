# pages/Summary.py
# Semantic summaries + semantic chat + SBERT topic analysis (session-only, offline)
# - Strong heading/TOC de-noising
# - Semantic flow ordering for cleaner summaries
# - Hybrid chat (SBERT + BM25/TF-IDF)
# - SBERT KMeans topics with cosine-to-centroid quality
# - Robust region column handling (unicef_region/region/etc.)
from __future__ import annotations

import re
import unicodedata
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ===== Optional deps (graceful fallbacks) =====
HAVE_BM25 = True
try:
    from rank_bm25 import BM25Okapi
except Exception:
    HAVE_BM25 = False

HAVE_SBERT = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    HAVE_SBERT = False

HAVE_SKLEARN = True
try:
    from sklearn.cluster import KMeans
except Exception:
    HAVE_SKLEARN = False

HAVE_THESAURUS = False
def _noop_expand_query(q: str, k_extra: int = 8) -> List[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", q.lower()))
try:
    # Optional UNICEF domain thesaurus (if you created one)
    from domain_thesaurus import expand_query as _expand_query, rm3_expand as _rm3_expand  # type: ignore
    HAVE_THESAURUS = True
except Exception:
    _expand_query = _noop_expand_query
    _rm3_expand = None

# ======================= Utilities ==========================
def get_consolidated_df() -> pd.DataFrame | None:
    for k in ("consolidated_df", "consolidated"):
        v = st.session_state.get(k)
        if isinstance(v, pd.DataFrame) and not v.empty:
            return v
    return None

def pick_region_col(df: pd.DataFrame) -> str | None:
    """Return the first region-like column name found, or None."""
    for name in ["unicef_region", "region", "Region", "UNICEF Region", "RegionName", "Region_Name"]:
        if name in df.columns:
            return name
    return None

def fold_accents(s: str) -> str:
    if s is None: return ""
    s = s.replace("\r"," ").replace("\t"," ")
    s = (s.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
           .replace("\u2013","-").replace("\u2014","-").replace("\u2015","-"))
    s = s.replace("-", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+"," ", s).strip()
    return s

TOKEN_RE = re.compile(r"[a-z0-9]+")
def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(fold_accents(s))

def sent_split(text: str) -> List[str]:
    text = re.sub(r"\s+"," ", str(text))
    return [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", text) if s.strip()]

def coalesce_cols(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v: parts.append(v)
    return " ".join(parts).strip()

# --- Patterns for noise ---
TOC_WORDS = r"(table of contents|contents|list of tables|list of figures|index|glossary|abbreviations)"
TOC_PAT_A = re.compile(rf"\b{TOC_WORDS}\b", re.I)
DOTTED_LEADER = re.compile(r"\.{3,}\s*\d{1,4}\s*$")
HEAD_NUM_A = re.compile(r"^\s*(\d+(\.\d+){0,4}|[IVXLCM]+)\s*[\.\)]\s+\S")   # "3.1 Title" / "II. Title"
HEAD_NUM_B = re.compile(r"^\s*\S+(?:\s+\S+){0,8}\s+(\d+(\.\d+){0,4})\s*(?:\.{3,}\s*\d+)?\s*$")  # "Title 3.1 .... 45"
ONLY_PUNC_NUM = re.compile(r"^[\s\.\,;:\-–—•*·\(\)\[\]\d]+$")
PAGE_PAT = re.compile(r"^\s*(page\s*\d+(\s*/\s*\d+)?|p\.\s*\d+)\s*$", re.I)
ANNEX_PAT = re.compile(r"^\s*(annex|annexe|appendix|appendice)\b", re.I)
FIG_TAB_PAT = re.compile(r"^\s*(figure|table)\b", re.I)
SEC_PAT = re.compile(r"^\s*(section|chapter|part)\b", re.I)

def _upper_ratio(s: str) -> float:
    letters = re.findall(r"[A-Za-z]", s)
    if not letters: return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)

def looks_like_heading_line(line: str) -> bool:
    """Detect short 'heading-like' lines: numbering, roman numerals, title-case blobs, colons, leaders."""
    if not line or not line.strip(): return True
    raw = line.strip()
    if PAGE_PAT.match(raw): return True
    if TOC_PAT_A.search(fold_accents(raw)): return True
    if ANNEX_PAT.match(raw): return True
    if FIG_TAB_PAT.match(raw): return True
    if SEC_PAT.match(raw): return True
    if DOTTED_LEADER.search(raw): return True
    if HEAD_NUM_A.match(raw): return True
    if HEAD_NUM_B.match(raw): return True
    if ONLY_PUNC_NUM.match(raw): return True
    toks = raw.split()
    if raw.endswith(":"): return True
    if len(toks) <= 12:
        if not re.search(r"[\.!?]$", raw):
            capish = sum(1 for w in toks if w[:1].isupper())
            if capish/len(toks) >= 0.75 or _upper_ratio(raw) >= 0.85:
                return True
    if raw.count(".") >= 6: return True
    toks_simple = tokenize(raw)
    if toks_simple:
        num_ratio = sum(1 for t in toks_simple if re.fullmatch(r"\d+(\.\d+)?", t)) / len(toks_simple)
        if num_ratio >= 0.6: return True
    return False

BULLET_LEAD = re.compile(r"^\s*[\-–—•*·]\s*")
TRAIL_LEADER_NUM = re.compile(r"\.{3,}\s*\d{1,4}\s*$")

def clean_sentence_text(s: str) -> str:
    if not s: return ""
    x = BULLET_LEAD.sub("", s)
    x = TRAIL_LEADER_NUM.sub("", x)
    x = re.sub(r"\s{2,}", " ", x).strip()
    return x

def looks_like_real_sentence(s: str, require_punct: bool, max_tokens: int = 60) -> bool:
    if not s or not s.strip(): return False
    raw = s.strip()
    if looks_like_heading_line(raw): return False
    if require_punct and not re.search(r"[\.!?]$", raw): return False
    if len(tokenize(raw)) <= 8 and _upper_ratio(raw) >= 0.85: return False
    toks = tokenize(raw)
    return 8 <= len(toks) <= max_tokens

def preclean_block(text: str, aggressive: bool = True) -> str:
    """Line-level cleaning for headers/footers/TOC/annex/table/figure/section lines."""
    if not text: return ""
    lines = [l.strip() for l in text.replace("\r", "\n").split("\n")]
    lines = [l for l in lines if l]
    if not lines: return ""
    freq = Counter(lines)
    cleaned = []
    for l in lines:
        if aggressive and 3 <= freq[l] <= len(lines):
            continue
        if looks_like_heading_line(l):
            continue
        cleaned.append(l)
    return "\n".join(cleaned)

# -------- TF-IDF mini (dense) for fallback --------
def build_tfidf(passages: List[str]):
    vocab: Dict[str,int] = {}
    docs_tokens: List[List[str]] = []
    for t in passages:
        toks = tokenize(t)
        docs_tokens.append(toks)
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    V = len(vocab)
    if V == 0:
        return np.zeros((len(passages),0), dtype=np.float32), vocab
    rows = []
    for toks in docs_tokens:
        counts: Dict[int,int] = {}
        for tok in toks:
            j = vocab[tok]; counts[j] = counts.get(j,0)+1
        rows.append(counts)
    M = np.zeros((len(passages), V), dtype=np.float32)
    dfc = np.zeros(V, dtype=np.float32)
    for i, counts in enumerate(rows):
        for j,cnt in counts.items():
            M[i,j] = cnt
        for j in counts.keys():
            dfc[j] += 1.0
    idf = np.log((1.0+len(passages))/(1.0+dfc)) + 1.0
    M *= idf
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    M = M / norms
    return M, vocab

def cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    if A.size == 0 or b.size == 0: return np.zeros(A.shape[0], dtype=np.float32)
    b = b/(np.linalg.norm(b)+1e-9)
    return A @ b

def minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12: return np.zeros_like(x)
    return (x-lo)/(hi-lo)

# ==================== Page setup & CSS ======================
st.set_page_config(page_title="Semantic Summary + Chat + Analysis", page_icon="🧠", layout="wide")
st.title("🧠 Semantic Summaries + Private Chat + Analysis (Session)")

# Fixed bottom chat input + dynamic padding
INPUT_BAR_H = 116
EXTRA_GAP = 28
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  padding-bottom: calc(var(--chat-input-h, {INPUT_BAR_H}px) + {EXTRA_GAP}px) !important;
}}
div[data-testid="stChatInput"] {{
  position: fixed !important; left: 50%; transform: translateX(-50%);
  bottom: 0; max-width: 980px; width: calc(100% - 2rem);
  z-index: 9999; background: var(--background-color);
  border-top: 1px solid rgba(49,51,63,0.2);
  padding: .25rem 0 .5rem 0;
}}
.chat-container {{ max-width: 980px; width: calc(100% - 2rem); margin: 0 auto; }}
</style>
""", unsafe_allow_html=True)
st.components.v1.html("""
<script>
(function(){
  const doc = parent.document;
  const main = doc.querySelector('[data-testid="stAppViewContainer"] > .main');
  const input = doc.querySelector('div[data-testid="stChatInput"]');
  if(!main || !input) return;
  const update = () => {
    const h = input.offsetHeight || 0;
    main.style.setProperty('--chat-input-h', h + 'px');
  };
  new ResizeObserver(update).observe(input);
  window.addEventListener('resize', update);
  update();
})();
</script>
""", height=0)

df = get_consolidated_df()
if df is None:
    st.error("No consolidated dataframe in session. Run the main page first.")
    st.stop()

with st.expander("Preview consolidated dataframe", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# Defaults for text columns
default_text_cols = [c for c in ["Findings of the evaluation","context","contributions","collaborations","innovations","full_text","text"] if c in df.columns]
if not default_text_cols:
    default_text_cols = [c for c in df.columns if c.lower() not in ("filename","country","year","unicef_region","region","filepath")][:2]

# ====================== Sidebar (semantic) ========================
with st.sidebar:
    st.header("Text & Summary Settings")
    text_cols = st.multiselect("Columns to use as text", options=list(df.columns), default=default_text_cols)
    max_sentences = st.slider("Summary sentences", 3, 12, 6)
    per_doc_cap = st.slider("Max sentences per document (summary pool)", 2, 20, 8)
    mmr_lambda = st.slider("MMR λ (relevance vs diversity)", 0.1, 0.9, 0.65, 0.05)
    summary_focus = st.text_input("Optional: focus query (guides the summary)", value="")
    filter_toc = st.checkbox("Filter TOC/heading noise", value=True)
    aggressive_line_clean = st.checkbox("Aggressive heading removal (line-level)", value=True)
    require_sentence_punct = st.checkbox("Require sentence punctuation (. ? !)", value=True,
                                         help="Drop lines that don't end as proper sentences to avoid headings.")
    order_mode = st.radio("Summary ordering", ["Semantic flow", "Relevance"], index=0,
                          help="Semantic flow builds a coherent chain; Relevance is centroid-first.")

    st.markdown("---")
    st.header("Chat Retrieval")
    top_k = st.slider("Top-k passages", 5, 40, 12, 1)
    gate = st.slider("Answer gate (final score)", 0.02, 0.60, 0.12, 0.01)
    use_bm25_pref = st.checkbox("Use BM25 (if installed)", value=HAVE_BM25)
    use_rm3 = st.checkbox("RM3 pseudo-relevance feedback (needs thesaurus)", value=False)
    min_vocab = st.slider("Semantic vocab: min token count", 1, 10, 2, 1,
                          help="Drop expansion tokens seen fewer times in the corpus. Originals kept.")
    if HAVE_SBERT:
        alpha = st.slider("Hybrid weight α (SBERT vs Lexical)", 0.0, 1.0, 0.65, 0.05)
        sbert_min = st.slider("Semantic similarity min (cosine)", 0.00, 1.00, 0.30, 0.01)
    else:
        st.info("SBERT not detected: using TF-IDF for summaries and chat.")
        alpha, sbert_min = 0.0, 0.0

# ==================== Build working corpus ========================
if not text_cols:
    st.warning("Pick at least one text column on the left.")
    st.stop()

work = df.copy()
work["__text__"] = work.apply(lambda r: coalesce_cols(r, text_cols), axis=1).fillna("").astype(str)

# ==================== Sentence pool for summaries =================
def build_sentence_pool(data: pd.DataFrame, per_doc_cap: int) -> pd.DataFrame:
    recs = []
    for _, r in data.iterrows():
        raw = r["__text__"]
        if not str(raw).strip():
            continue
        pre = preclean_block(raw, aggressive=aggressive_line_clean) if filter_toc else raw
        sents = sent_split(pre)
        kept = []
        for s in sents:
            s2 = clean_sentence_text(s)
            if filter_toc and looks_like_heading_line(s2):
                continue
            if not looks_like_real_sentence(s2, require_sentence_punct, max_tokens=60):
                continue
            kept.append(s2)
        if not kept:
            kept = [clean_sentence_text(s) for s in sent_split(raw)]
            kept = [s for s in kept if looks_like_real_sentence(s, False, 60)]
        for s in kept[:per_doc_cap]:
            recs.append({
                "sentence": s,
                "filename": r.get("filename"),
                "country": r.get("country"),
                "region": r.get(pick_region_col(df)) if pick_region_col(df) else r.get("region"),
                "year": r.get("year"),
            })
    return pd.DataFrame(recs)

sent_pool_df = build_sentence_pool(work, per_doc_cap=per_doc_cap)
if sent_pool_df.empty:
    st.warning("No sentences found to summarize. Try lowering filters or unchecking 'Require punctuation'.")
    st.stop()

# ================== SBERT resources (cached) ======================
if HAVE_SBERT:
    @st.cache_resource(show_spinner=False)
    def _load_sbert(name: str = "all-MiniLM-L6-v2"):
        return SentenceTransformer(name)

    @st.cache_resource(show_spinner=False)
    def _encode_texts(texts_tuple: Tuple[str, ...], model_name: str = "all-MiniLM-L6-v2"):
        model = _load_sbert(model_name)
        return model.encode(list(texts_tuple), batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    SBERT_NAME = "all-MiniLM-L6-v2"
    sent_emb = _encode_texts(tuple(sent_pool_df["sentence"].tolist()), SBERT_NAME)
else:
    tfidf_M_s, tfidf_vocab_s = build_tfidf(sent_pool_df["sentence"].tolist())
    sent_emb = None

# ============== Semantic MMR summarization & ordering ==============
def mmr_select(k: int, emb: np.ndarray, q_vec: np.ndarray | None, lam: float = 0.65) -> List[int]:
    n = emb.shape[0]
    if n == 0: return []
    if q_vec is None:
        centroid = emb.mean(axis=0, keepdims=True)
        rel = (emb @ (centroid.T)).ravel()
    else:
        rel = (emb @ (q_vec.reshape(-1,1))).ravel()
    selected: List[int] = []
    candidate = set(range(n))
    if n <= k: return list(range(n))
    sims = emb @ emb.T
    for _ in range(k):
        if not candidate: break
        if not selected:
            idx0 = list(candidate)
            pick = idx0[int(np.argmax(rel[idx0]))]
        else:
            best_score, best_i = -1e9, None
            sel = np.array(selected, dtype=int)
            for i_idx in candidate:
                diversity = float(np.max(sims[i_idx, sel]))
                score = lam * float(rel[i_idx]) - (1.0 - lam) * diversity
                if score > best_score:
                    best_score, best_i = score, i_idx
            pick = int(best_i)
        selected.append(pick); candidate.remove(pick)
    return selected

def order_semantic_flow(selected_idx: List[int], emb: np.ndarray) -> List[int]:
    if not selected_idx: return []
    sub = np.array(selected_idx, dtype=int)
    S = emb[sub, :]
    centroid = S.mean(axis=0); centroid = centroid/(np.linalg.norm(centroid)+1e-9)
    rel = (S @ centroid).ravel()
    order = [int(np.argmax(rel))]
    remaining = set(range(len(sub)))
    remaining.remove(order[0])
    sims = S @ S.T
    while remaining:
        last = order[-1]
        nxt = max(remaining, key=lambda j: sims[last, j])
        order.append(nxt)
        remaining.remove(nxt)
    return [int(sub[i]) for i in order]

def semantic_summary(max_sents: int, focus: str = "") -> pd.DataFrame:
    if HAVE_SBERT:
        model = _load_sbert(SBERT_NAME)
        q_vec = model.encode([focus], normalize_embeddings=True, show_progress_bar=False)[0] if focus.strip() else None
        idx = mmr_select(min(max_sents, len(sent_pool_df)), sent_emb, q_vec, lam=mmr_lambda)
        if order_mode == "Semantic flow":
            idx = order_semantic_flow(idx, sent_emb)
        picked = sent_pool_df.iloc[idx].copy()
        picked["rank"] = np.arange(1, len(picked)+1)
        return picked[["rank","sentence","filename","country","region","year"]]
    else:
        centroid = tfidf_M_s.mean(axis=0)
        rel = cosine_sim_matrix(tfidf_M_s, centroid)
        order = np.argsort(-rel)[:max_sents]
        picked = sent_pool_df.iloc[order].copy()
        picked["rank"] = np.arange(1, len(picked)+1)
        return picked[["rank","sentence","filename","country","region","year"]]

# ======================= Token counts (for query filter) =======================
@st.cache_data(show_spinner=False)
def build_token_counts(texts: Tuple[str, ...]) -> Dict[str,int]:
    bag = Counter()
    for t in texts:
        for tok in tokenize(t):
            bag[tok] += 1
    return dict(bag)

# Build retrieval passages (chat)
def chunk_text(text: str, max_words: int = 140, overlap: int = 40):
    sents = sent_split(text)
    chunks, cur, count = [], [], 0
    for s in sents:
        w = len(tokenize(s))
        if count + w > max_words and cur:
            chunks.append(" ".join(cur))
            ov, ccount = [], 0
            for rs in reversed(cur):
                cw = len(tokenize(rs))
                if ccount + cw >= overlap:
                    ov.insert(0, rs); break
                ov.insert(0, rs); ccount += cw
            cur = ov + [s]; count = sum(len(tokenize(x)) for x in cur)
        else:
            cur.append(s); count += w
    if cur: chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]

PASS_CHUNK = 140; PASS_OVERLAP = 35
passages = []
reg_col_in_df = pick_region_col(df)
for idx, r in work.iterrows():
    text = r["__text__"]
    if not str(text).strip(): continue
    for j, ch in enumerate(chunk_text(text, max_words=PASS_CHUNK, overlap=PASS_OVERLAP)):
        passages.append({
            "pid": f"{idx}-{j}",
            "filename": r.get("filename", idx),
            "country": r.get("country"),
            "region": (r.get(reg_col_in_df) if reg_col_in_df else r.get("region")),
            "year": r.get("year"),
            "text": ch
        })
passages_df = pd.DataFrame(passages)
if passages_df.empty:
    st.warning("No passages available from selected columns.")
    st.stop()

token_counts = build_token_counts(tuple(passages_df["text"].tolist()))

# Build retrievers
if use_bm25_pref and HAVE_BM25:
    bm25 = BM25Okapi([tokenize(t) for t in passages_df["text"].tolist()])
    tfidf_M, tfidf_vocab = None, {}
else:
    bm25 = None
    tfidf_M, tfidf_vocab = build_tfidf(passages_df["text"].tolist())

# SBERT passage embeddings (for chat) if available
if HAVE_SBERT:
    @st.cache_resource(show_spinner=False)
    def _encode_passages(texts_tuple: Tuple[str, ...], model_name: str = "all-MiniLM-L6-v2"):
        model = _load_sbert(model_name)
        return model.encode(list(texts_tuple), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    passages_emb = _encode_passages(tuple(passages_df["text"].tolist()), "all-MiniLM-L6-v2")
else:
    passages_emb = None

# =========================== Layout ==============================
tab_sum, tab_chat, tab_sem = st.tabs([
    "① Semantic Summaries (w/ sources)",
    "② Private Chat (Semantic)",
    "③ Semantic Analysis (SBERT)"
])

# -------------------- Summaries tab --------------------
with tab_sum:
    st.subheader("Semantic Summary")
    picked_df = semantic_summary(max_sentences, summary_focus)
    if picked_df.empty:
        st.info("No sentences selected. Try unchecking 'Require punctuation' or lowering filters.")
    else:
        st.write("**Selected summary sentences (with sources):**")
        st.dataframe(picked_df, use_container_width=True)
        st.download_button("⬇️ Download summary (CSV)",
                           data=picked_df.to_csv(index=False).encode("utf-8"),
                           file_name="semantic_summary_with_sources.csv",
                           mime="text/csv",
                           use_container_width=True)

# ---------------------- Chat tab ----------------------
with tab_chat:
    st.subheader("Private Chat (Semantic, offline)")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_country = st.multiselect("Country", sorted([x for x in df["country"].dropna().unique().tolist()]) if "country" in df.columns else [])
    with fc2:
        region_col = pick_region_col(df)
        sel_region = st.multiselect(
            "Region",
            sorted(df[region_col].dropna().unique().tolist()) if region_col else []
        )
    with fc3:
        sel_year = st.multiselect("Year", sorted([int(x) for x in pd.to_numeric(df["year"], errors="coerce").dropna().unique().tolist()]) if "year" in df.columns else [])

    def expand_tokens_for_query(q: str) -> List[str]:
        base_tokens = tokenize(q)
        try:
            q_terms = _expand_query(q, k_extra=8) if HAVE_THESAURUS else list(base_tokens)
        except Exception:
            q_terms = list(base_tokens)
        if min_vocab > 1:
            q_terms = [t for t in q_terms if (t in base_tokens) or token_counts.get(t, 0) >= min_vocab] or (base_tokens or ["evaluation"])
        return q_terms

    def minmax_arr(x: np.ndarray) -> np.ndarray:
        if x.size == 0: return x
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-12: return np.zeros_like(x)
        return (x-lo)/(hi-lo)

    def retrieve(query: str, k: int) -> pd.DataFrame:
        q_terms = expand_tokens_for_query(query)
        if bm25 is not None:
            lex = np.array(bm25.get_scores(q_terms))
            if use_rm3 and HAVE_THESAURUS and _rm3_expand is not None:
                first_ord = np.argsort(-lex)[: min(k*3, len(lex))]
                try:
                    q_terms2 = _rm3_expand(q_terms, passages_df.iloc[first_ord]["text"].tolist(), n_terms=8)
                except Exception:
                    q_terms2 = q_terms
                if min_vocab > 1:
                    q_terms2 = [t for t in q_terms2 if (t in q_terms) or token_counts.get(t,0) >= min_vocab] or q_terms
                lex = np.array(bm25.get_scores(q_terms2))
        else:
            q_vec = np.zeros(tfidf_M.shape[1], dtype=np.float32)
            for t in q_terms:
                j = tfidf_vocab.get(t)
                if j is not None: q_vec[j] += 1.0
            lex = cosine_sim_matrix(tfidf_M, q_vec)
        lex_norm = minmax_arr(lex)

        if HAVE_SBERT and passages_emb is not None:
            model = _load_sbert("all-MiniLM-L6-v2")
            q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            sem = passages_emb @ q_vec
            if sbert_min > 0 and float(np.max(sem)) < float(sbert_min):
                return pd.DataFrame(columns=list(passages_df.columns)+["score","lex","sem"])
            sem_norm = minmax_arr(sem)
            final = alpha * sem_norm + (1.0 - alpha) * lex_norm
        else:
            sem = None
            final = lex_norm

        order = np.argsort(-final)[: min(k, len(final))]
        out = passages_df.iloc[order].copy()
        out["score"] = final[order]
        out["lex"] = lex_norm[order]
        if sem is not None:
            out["sem"] = sem[order]
        return out.reset_index(drop=True)

    def answer_from_hits(query: str, hits: pd.DataFrame, max_snips: int = 6, max_chars: int = 900) -> Tuple[str, pd.DataFrame]:
        rows = []
        for _, r in hits.head(max_snips).iterrows():
            for s in sent_split(r["text"]):
                s2 = clean_sentence_text(s)
                if filter_toc and looks_like_heading_line(s2):
                    continue
                if not looks_like_real_sentence(s2, True, 60):
                    continue
                rows.append({
                    "sentence": s2,
                    "filename": r.get("filename"),
                    "country": r.get("country"),
                    "region": r.get("region"),
                    "year": r.get("year"),
                })
        if not rows:
            return "", pd.DataFrame()
        cand = pd.DataFrame(rows)
        if HAVE_SBERT:
            model = _load_sbert("all-MiniLM-L6-v2")
            S = model.encode(cand["sentence"].tolist(), normalize_embeddings=True, show_progress_bar=False)
            qv = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            # select + flow order
            idx = mmr_select(k=min(6, len(cand)), emb=S, q_vec=qv, lam=0.70)
            idx = order_semantic_flow(idx, S)
            picked = cand.iloc[idx].copy()
        else:
            tf_s, vocab_s = build_tfidf(cand["sentence"].tolist())
            q_vec = np.zeros(tf_s.shape[1], dtype=np.float32)
            for t in tokenize(query):
                j = vocab_s.get(t)
                if j is not None: q_vec[j] += 1.0
            sims = cosine_sim_matrix(tf_s, q_vec)
            order = np.argsort(-sims)[: min(6, len(sims))]
            picked = cand.iloc[order].copy()

        answer = " ".join(picked["sentence"].tolist())
        if len(answer) > max_chars: answer = answer[:max_chars].rsplit(" ", 1)[0] + "…"
        return answer, picked

    if "summary_chat_history" not in st.session_state:
        st.session_state.summary_chat_history = []

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown("### Chat")
    for msg in st.session_state.summary_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
        src_tbl = msg.get("sources_df")
        if isinstance(src_tbl, pd.DataFrame) and not src_tbl.empty:
            with st.expander("Top sources", expanded=False):
                st.dataframe(src_tbl, use_container_width=True)

    user_q = st.chat_input("Ask about the loaded documents…")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_q:
        st.session_state.summary_chat_history.append({"role":"user","content":user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        hits = retrieve(user_q, k=top_k)

        if hits.empty or float(hits["score"].max()) < float(gate):
            reply = "I’m not confident I can answer that from the loaded text. Try more specific terms, broaden filters, or lower the answer gate."
            st.session_state.summary_chat_history.append({"role":"assistant","content":reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            answer, picked = answer_from_hits(user_q, hits, max_snips=6)
            if not answer:
                reply = "I found relevant passages but couldn’t synthesize a concise answer. Try a more specific question."
                st.session_state.summary_chat_history.append({"role":"assistant","content":reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
            else:
                src_cols = ["filename","year","region","country"]
                src_df = hits[src_cols + ["score"]].head(6).copy()
                reply = f"{answer}\n\n**Sources:** " + ", ".join(
                    f"{r.get('filename')} ({r.get('year')}, {r.get('region')})" for _, r in src_df.iterrows()
                )
                st.session_state.summary_chat_history.append({"role":"assistant","content":reply, "sources_df": src_df})
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    with st.expander("Top sources", expanded=False):
                        st.dataframe(src_df, use_container_width=True)

# ------------------ Semantic Analysis tab -------------------
with tab_sem:
    st.subheader("Semantic Analysis (Embeddings)")
    st.caption("Embedding engine: **SBERT**. Topic quality = cosine(doc → topic centroid).")

    if not HAVE_SBERT:
        st.error("SBERT not detected. Install `sentence-transformers` to run semantic topic analysis.")
        st.stop()
    if not HAVE_SKLEARN:
        st.error("scikit-learn not detected. Install `scikit-learn` for KMeans clustering.")
        st.stop()

    gran = st.radio("Granularity", ["Documents", "Informative sentences"], index=0, horizontal=True)

    if gran == "Documents":
        texts = work["__text__"].astype(str).tolist()
        reg_col = pick_region_col(work)
        # take only meta columns that exist
        base_cols = [c for c in ["filename", "country", "year"] if c in work.columns]
        meta = work[base_cols].copy()
        meta["region"] = work[reg_col] if reg_col else None
    else:
        texts = sent_pool_df["sentence"].astype(str).tolist()
        meta = pd.DataFrame({
            "filename": sent_pool_df["filename"].fillna("(unknown)"),
            "country": sent_pool_df["country"],
            "region": sent_pool_df["region"],
            "year":   sent_pool_df["year"],
        })

    n_items = len(texts)
    if n_items < 2:
        st.info("Not enough items to cluster.")
        st.stop()

    n_default = min(8, max(2, n_items//10 or 2))
    k = st.slider("Number of topics (KMeans clusters)", 2, max(2, min(40, n_items)), n_default, 1)

    with st.spinner("Embedding & clustering…"):
        model = _load_sbert("all-MiniLM-L6-v2")
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        k = min(k, n_items)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(emb)

        emb_norm = emb
        topic_rows = []
        doc_rows = []

        STOP = {
            "the","and","of","to","in","for","on","with","a","an","is","are","was","were","be","by","as","that","this","it",
            "from","or","at","we","our","their","they","these","those","has","have","had","not","but","can","will",
            "de","la","el","en","los","las","les","le","des","du","et","que","un","une","y","con","para","por",
            "evaluation","evaluacion","évaluation","findings","chapter","annex","annexe","appendix","section","sections",
            "table","figure","unicef","programme","program","project","report","coherence","relevance","effectiveness",
            "efficiency","sustainability","connectedness"
        }

        def topic_terms(idxs: List[int], topn=6) -> str:
            bag = Counter()
            for i in idxs:
                for t in tokenize(texts[i]):
                    if t in STOP or len(t) < 3: continue
                    bag[t] += 1
            return ", ".join([w for w,_ in bag.most_common(topn)]) if bag else "(noisy/short)"

        for t in range(k):
            idxs = np.where(labels == t)[0].tolist()
            if not idxs: continue
            centroid = np.mean(emb_norm[idxs, :], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            sims = (emb_norm[idxs, :] @ centroid).astype(float)
            topic_rows.append({
                "topic": t,
                "size": len(idxs),
                "quality_mean": float(np.mean(sims)),
                "quality_median": float(np.median(sims)),
                "quality_min": float(np.min(sims)),
                "quality_max": float(np.max(sims)),
                "label_terms": topic_terms(idxs, topn=6)
            })
            for i, s in zip(idxs, sims):
                doc_rows.append({
                    "topic": t,
                    "item_index": i,
                    "similarity": float(s),
                    "text_preview": (texts[i][:240] + "…") if len(texts[i]) > 240 else texts[i],
                })

        topic_df = pd.DataFrame(topic_rows).sort_values(["quality_mean","size"], ascending=[False, False]).reset_index(drop=True)
        doc_df = pd.DataFrame(doc_rows)
        doc_df = doc_df.merge(meta.reset_index(drop=True), left_on="item_index", right_index=True, how="left")

    st.markdown("#### Topic semantic quality (cosine to centroid)")
    st.dataframe(topic_df, use_container_width=True)
    st.download_button(
        "⬇️ Download topic table (CSV)",
        topic_df.to_csv(index=False).encode("utf-8"),
        file_name="semantic_topics_quality.csv",
        mime="text/csv",
        use_container_width=True
    )

    chart = alt.Chart(topic_df).mark_bar().encode(
        x=alt.X("topic:O", title="Topic"),
        y=alt.Y("quality_mean:Q", title="Mean cosine → centroid", scale=alt.Scale(domain=[0,1])),
        tooltip=[
            alt.Tooltip("topic:O"),
            alt.Tooltip("size:Q"),
            alt.Tooltip("quality_mean:Q", format=".3f"),
            alt.Tooltip("quality_median:Q", format=".3f"),
            alt.Tooltip("quality_min:Q", format=".3f"),
            alt.Tooltip("quality_max:Q", format=".3f"),
            alt.Tooltip("label_terms:N", title="Top terms"),
        ]
    ).properties(height=280)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Assignments (per item) with similarity & sources")
    cols = ["topic","similarity","filename","country","region","year","text_preview"]
    cols = [c for c in cols if c in doc_df.columns] + [c for c in doc_df.columns if c not in cols]
    st.dataframe(doc_df[cols].sort_values(["topic","similarity"], ascending=[True, False]), use_container_width=True)
    st.download_button(
        "⬇️ Download assignments (CSV)",
        doc_df[cols].to_csv(index=False).encode("utf-8"),
        file_name="semantic_topic_assignments.csv",
        mime="text/csv",
        use_container_width=True
    )
