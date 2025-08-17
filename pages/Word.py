# pages/2_Keyword_Analyzer.py
# Keyword Analyzer — reads consolidated_df from session (no CSV upload)
# - Pick one text column
# - Enter keywords
# - Exact vs AI/Semantic matching (SBERT if available; else RapidFuzz; else fallback)
# - Counts per document
# - Aggregations by Year & Region
# - Charts
# - EXTRA: previews + downloads for BOTH Exact & Semantic:
#          (a) per-document totals (non-zero)
#          (b) per-document SNIPPETS (±1 sentence around matches), non-zero

from __future__ import annotations
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Optional engines ----------
HAVE_RAPIDFUZZ = False
try:
    from rapidfuzz import process, fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    pass

HAVE_SBERT = False
SBERT_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SBERT = True
except Exception:
    pass

# ========================= Helpers =========================
def get_consolidated_df() -> pd.DataFrame | None:
    """Safely pull a consolidated dataframe from session without boolean coercion."""
    df1 = st.session_state.get("consolidated_df", None)
    if isinstance(df1, pd.DataFrame) and not df1.empty:
        return df1
    df2 = st.session_state.get("consolidated", None)
    if isinstance(df2, pd.DataFrame) and not df2.empty:
        return df2
    return None

def fold_accents(s: str) -> str:
    """Lower + remove accents + normalize whitespace/dashes for robust matching."""
    if s is None:
        return ""
    s = s.replace("\r", " ").replace("\t", " ")
    s = s.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2015", "-")
    s = s.replace("-", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", s)

def generate_ngrams(tokens: List[str], n_min: int = 1, n_max: int = 3) -> List[str]:
    out = []
    N = len(tokens)
    for n in range(n_min, n_max + 1):
        for i in range(N - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out

@st.cache_data(show_spinner=False)
def build_vocab_from_corpus(text_series: pd.Series, min_count: int = 2, n_max: int = 3, max_vocab: int = 20000) -> List[str]:
    """Phrase vocab (1–3 grams) from the selected text column for semantic expansion."""
    cnt = Counter()
    for txt in text_series.fillna("").astype(str).tolist():
        norm = fold_accents(txt)
        toks = tokenize_words(norm)
        grams = generate_ngrams(toks, 1, n_max)
        cnt.update(grams)
    items = [(p, c) for p, c in cnt.items() if c >= min_count and 3 <= len(p) <= 40]
    items.sort(key=lambda x: (-x[1], x[0]))
    return [p for p, _ in items[:max_vocab]]

@st.cache_data(show_spinner=False)
def sbert_model_load(name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def semantic_expand_keywords(
    keywords: List[str],
    vocab: List[str],
    topk: int = 12,
    threshold: float = 0.55,
) -> Dict[str, List[str]]:
    """
    Expand each keyword with semantically similar phrases from the corpus vocab.
    Prefers SBERT; falls back to RapidFuzz; then simple variants.
    Returns dict: keyword -> [expanded phrases including normalized original].
    """
    expansions: Dict[str, List[str]] = {k: [] for k in keywords}
    if not keywords or not vocab:
        return expansions

    base_norm = {k: fold_accents(k) for k in keywords}

    global SBERT_MODEL
    if HAVE_SBERT and SBERT_MODEL is None:
        try:
            SBERT_MODEL = sbert_model_load()
        except Exception:
            pass

    if HAVE_SBERT and SBERT_MODEL is not None:
        kw_emb = SBERT_MODEL.encode([base_norm[k] for k in keywords], normalize_embeddings=True)
        vb_emb = SBERT_MODEL.encode(vocab, normalize_embeddings=True)
        sims = (kw_emb @ vb_emb.T)
        for i, k in enumerate(keywords):
            scores = sims[i]
            idx = np.where(scores >= threshold)[0]
            if idx.size == 0:
                top_idx = np.argsort(-scores)[:topk]
                picks = [vocab[j] for j in top_idx]
            else:
                top_idx = idx[np.argsort(-scores[idx])][:topk]
                picks = [vocab[j] for j in top_idx]
            uniq, seen = [], set()
            for p in [base_norm[k]] + picks:
                if p and p not in seen:
                    uniq.append(p); seen.add(p)
            expansions[k] = uniq
        return expansions

    if HAVE_RAPIDFUZZ:
        for k in keywords:
            target = base_norm[k]
            scored = process.extract(
                target, vocab, scorer=fuzz.token_set_ratio, score_cutoff=int(threshold * 100)
            )
            scored.sort(key=lambda x: (-x[1], x[0]))
            picks = [m for m, s, _ in scored[:topk]]
            expansions[k] = [target] + [p for p in picks if p != target]
        return expansions

    # Minimal fallback: inflections
    def simple_variants(w: str) -> List[str]:
        out = {w}
        if w.endswith("y") and len(w) > 3:
            out.add(w[:-1] + "ies")
        if not w.endswith("s"):
            out.add(w + "s")
        out.add(w + "ing")
        out.add(w + "ed")
        return list(out)

    for k in keywords:
        base = base_norm[k]
        variants = simple_variants(base)
        picks = [v for v in variants if v in vocab][:topk]
        expansions[k] = [base] + [p for p in picks if p != base]
    return expansions

def count_occurrences_normtext(normtext: str, phrases: List[str], whole_word=True) -> int:
    if not normtext or not phrases:
        return 0
    total = 0
    for ph in phrases:
        if not ph:
            continue
        pat = re.compile(rf"\b{re.escape(ph)}\b") if whole_word else re.compile(re.escape(ph))
        total += len(pat.findall(normtext))
    return total

# --- New: sentence split + snippet extraction (± window) ---
def sent_split(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text))
    return [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", text) if s.strip()]

def extract_snippets_for_doc(
    raw_text: str,
    phrases: List[str],
    window: int = 1,
    whole_word: bool = True
) -> List[str]:
    """
    Return unique snippets for any phrase match with ±window sentences.
    Matching is done on accent-folded lowercased sentences; snippets are returned in ORIGINAL text.
    """
    if not raw_text or not phrases:
        return []
    sents_raw = sent_split(raw_text)
    if not sents_raw:
        return []
    sents_norm = [fold_accents(s) for s in sents_raw]
    pats = [re.compile(rf"\b{re.escape(p)}\b") if whole_word else re.compile(re.escape(p)) for p in phrases if p]
    out, seen = [], set()
    for j, ns in enumerate(sents_norm):
        if any(p.search(ns) for p in pats):
            s = max(0, j - window)
            e = min(len(sents_raw), j + window + 1)
            snip = " ".join(sents_raw[s:e]).strip()
            key = fold_accents(snip)[:500]
            if key in seen:
                continue
            seen.add(key)
            out.append(snip)
    return out

# ========================= UI =========================
st.set_page_config(page_title="Keyword Analyzer (from session)", page_icon="🧮", layout="wide")
st.title("🧮 Keyword Analyzer (from session)")

# ---------- Pull dataframe from session ----------
df = get_consolidated_df()
if df is None:
    st.error("No consolidated dataframe found in session. Go to the main page, run the pipeline, then come back.")
    st.stop()

with st.expander("Preview consolidated dataframe", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Matching")
    match_type = st.radio(
        "Match type",
        ["Exact (whole word/phrase)", "AI / Semantic"],
        index=0,
    )
    whole_word = st.checkbox("Whole-word/phrase boundary", value=True)
    ngram_max = st.slider("Semantic vocab: max n-gram", 1, 3, 3, 1, help="Used in AI/Semantic mode")
    min_count_vocab = st.slider("Semantic vocab: min phrase count", 1, 10, 2, 1, help="Used in AI/Semantic mode")
    topk_expansion = st.slider("Max expansions per keyword", 3, 30, 12, 1, help="Used in AI/Semantic mode")
    threshold = st.slider("Semantic similarity threshold", 0.30, 0.90, 0.55, 0.01, help="SBERT cosine or RapidFuzz ratio/100")

    st.markdown("---")
    st.header("Charts")
    stack_by = st.radio("Stack bars by", ["keyword", "region"], index=0)
    show_doc_table = st.checkbox("Show per-document table", value=False)
    show_preview_expansions = st.checkbox("Show expansions preview (AI/Semantic)", value=True)

# ---------- Choose text & meta columns ----------
cols = list(df.columns)

text_defaults = [c for c in ["Findings of the evaluation", "full_text", "text"] if c in cols]
default_text = text_defaults[0] if text_defaults else cols[0]
text_col = st.selectbox("Text column to analyze", options=cols, index=cols.index(default_text))

meta_cols = ["(none)"] + cols
def _auto(colnames, keys):
    s = {str(c).strip().lower(): c for c in colnames}
    for k in keys:
        if k in s:
            return s[k]
    return "(none)"

year_col_default = _auto(cols, ["year"])
region_col_default = _auto(cols, ["region", "unicef_region"])
filename_col_default = _auto(cols, ["filename", "pdf", "name"])
country_col_default = _auto(cols, ["country"])

c1, c2, c3 = st.columns(3)
with c1:
    year_col = st.selectbox("Year column", options=meta_cols,
                            index=meta_cols.index(year_col_default) if year_col_default in meta_cols else 0)
with c2:
    region_col = st.selectbox("Region column", options=meta_cols,
                              index=meta_cols.index(region_col_default) if region_col_default in meta_cols else 0)
with c3:
    filename_col = st.selectbox("Filename column", options=meta_cols,
                                index=meta_cols.index(filename_col_default) if filename_col_default in meta_cols else 0)

# ---------- Keywords input ----------
st.subheader("Enter keywords")
kw_text = st.text_area(
    "Keywords (comma or newline separated)",
    value="learning outcomes, teacher training, curriculum, attendance",
    height=100,
)
keywords = [w.strip() for w in re.split(r"[,\n]", kw_text) if w.strip()]
if not keywords:
    st.warning("Please enter at least one keyword.")
    st.stop()

st.info(f"Analyzing **{text_col}** with {len(keywords)} keyword(s).")

# ---------- Prepare working frame ----------
work = df.copy()
work["_text_raw"] = work[text_col].fillna("").astype(str)
work["_text_norm"] = work["_text_raw"].map(fold_accents)

if year_col != "(none)":
    work["_year"] = work[year_col]
else:
    work["_year"] = np.nan
if region_col != "(none)":
    work["_region"] = work[region_col].astype(str)
else:
    work["_region"] = "Unknown"
if filename_col != "(none)":
    work["_filename"] = work[filename_col].astype(str)
else:
    work["_filename"] = np.arange(len(work)).astype(str)

# Country is optional (no extra UI to preserve appearance)
if country_col_default != "(none)" and country_col_default in cols:
    work["_country"] = work[country_col_default].astype(str)
else:
    work["_country"] = "Unknown"

# ---------- Build expansions for SELECTED mode (for charts) ----------
engine_note = ""
if match_type.startswith("AI"):
    vocab = build_vocab_from_corpus(work["_text_norm"], min_count=min_count_vocab, n_max=ngram_max, max_vocab=20000)
    expansions = semantic_expand_keywords(keywords, vocab, topk=topk_expansion, threshold=threshold)
    if HAVE_SBERT:
        engine_note = "Semantic engine: SBERT"
    elif HAVE_RAPIDFUZZ:
        engine_note = "Semantic engine: RapidFuzz"
    else:
        engine_note = "Semantic engine: basic variants (no SBERT/RapidFuzz installed)"
else:
    expansions = {k: [fold_accents(k)] for k in keywords}
    engine_note = "Exact matching (accent-insensitive, lowercased)"

st.caption(engine_note)

if match_type.startswith("AI") and show_preview_expansions:
    with st.expander("Preview: expansions per keyword", expanded=False):
        for k in keywords:
            st.markdown(f"**{k}** → {', '.join(expansions.get(k, [])[:30])}")

# ---------- Count per-document (SELECTED mode) ----------
rows = []
for _, r in work.iterrows():
    normtext = r["_text_norm"]
    rec = {
        "filename": r["_filename"],
        "year": r["_year"],
        "region": r["_region"],
        "country": r["_country"],
    }
    total = 0
    for k in keywords:
        phrases = expansions.get(k, [])
        c = count_occurrences_normtext(normtext, phrases, whole_word=whole_word)
        rec[k] = c
        total += c
    rec["total_count"] = total
    rows.append(rec)

doc_counts = pd.DataFrame(rows)
try:
    doc_counts["year"] = pd.to_numeric(doc_counts["year"], errors="coerce").astype("Int64")
except Exception:
    pass

# ---------- Aggregations for charts ----------
agg_by_year = (
    doc_counts.groupby("year", dropna=False)[keywords].sum().reset_index().rename(columns={"year": "Year"})
)
agg_by_region = (
    doc_counts.groupby("region", dropna=False)[keywords].sum().reset_index().rename(columns={"region": "Region"})
)
long_year = agg_by_year.melt(id_vars=["Year"], var_name="keyword", value_name="count")
long_region = agg_by_region.melt(id_vars=["Region"], var_name="keyword", value_name="count")

# ---------- Results (selected mode) ----------
st.subheader("Results")

m1, m2, m3 = st.columns(3)
m1.metric("Docs analyzed", len(doc_counts))
m2.metric("Unique regions", doc_counts["region"].nunique(dropna=False))
try:
    yr_min = int(pd.to_numeric(doc_counts["year"], errors="coerce").min())
    yr_max = int(pd.to_numeric(doc_counts["year"], errors="coerce").max())
    m3.metric("Year range", f"{yr_min}–{yr_max}")
except Exception:
    m3.metric("Year range", "n/a")

# Optional per-document table
if show_doc_table:
    st.markdown("**Per-document counts (selected mode)**")
    st.dataframe(doc_counts, use_container_width=True, height=400)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download per-document counts (selected mode, CSV)",
        data=doc_counts.to_csv(index=False).encode("utf-8"),
        file_name="keyword_counts_per_document_selected.csv",
        mime="text/csv",
        use_container_width=True
    )
with c2:
    agg_total = (
        doc_counts.assign(_total=doc_counts[keywords].sum(axis=1))
        .groupby(["year","region"], dropna=False)["_total"].sum().reset_index()
        .rename(columns={"_total":"total"})
    )
    st.download_button(
        "⬇️ Download totals by Year & Region (selected mode)",
        data=agg_total.to_csv(index=False).encode("utf-8"),
        file_name="totals_by_year_region_selected.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------- Charts (selected mode) ----------
st.subheader("Charts")

if stack_by == "keyword":
    chart_year = (
        alt.Chart(long_year)
        .mark_bar()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("sum(count):Q", title="Count"),
            color=alt.Color("keyword:N", title="Keyword"),
            tooltip=["Year", "keyword", "sum(count)"]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_year, use_container_width=True)

    chart_region = (
        alt.Chart(long_region)
        .mark_bar()
        .encode(
            x=alt.X("Region:N", sort="-y", title="Region"),
            y=alt.Y("sum(count):Q", title="Count"),
            color=alt.Color("keyword:N", title="Keyword"),
            tooltip=["Region", "keyword", "sum(count)"]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_region, use_container_width=True)
else:
    agg_by_year_region = (
        doc_counts.assign(_total=doc_counts[keywords].sum(axis=1))
        .groupby(["year","region"], dropna=False)["_total"].sum().reset_index()
        .rename(columns={"year":"Year","region":"Region","_total":"count"})
    )

    chart_year = (
        alt.Chart(agg_by_year_region)
        .mark_bar()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("sum(count):Q", title="Count"),
            color=alt.Color("Region:N", title="Region"),
            tooltip=["Year", "Region", "sum(count)"]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_year, use_container_width=True)

    chart_region = (
        alt.Chart(agg_by_year_region)
        .mark_bar()
        .encode(
            x=alt.X("Region:N", sort="-y", title="Region"),
            y=alt.Y("sum(count):Q", title="Count"),
            color=alt.Color("Region:N", legend=None),
            tooltip=["Region", "sum(count)"]
        )
        .properties(height=320)
    )
    st.altair_chart(chart_region, use_container_width=True)

st.caption("This page reads `st.session_state['consolidated_df']` (or 'consolidated') from the main app. Run the pipeline there first, then analyze here.")

# ======================================================================
# EXTRA: Compute BOTH exact & semantic totals + SNIPPETS (±1 sentence),
#        preview non-zero tables, then enable downloads.
# ======================================================================

st.markdown("---")
st.subheader("Extra downloads & previews")

# Build expansions for both modes INDEPENDENT of the sidebar selection (to not change appearance)
exp_exact = {k: [fold_accents(k)] for k in keywords}

try:
    vocab_all = build_vocab_from_corpus(work["_text_norm"], min_count=min_count_vocab, n_max=ngram_max, max_vocab=20000)
    exp_sem = semantic_expand_keywords(keywords, vocab_all, topk=topk_expansion, threshold=threshold)
except Exception:
    exp_sem = {k: [fold_accents(k)] for k in keywords}

def compute_doc_counts(exp_map: Dict[str, List[str]], whole_word: bool) -> pd.DataFrame:
    rows = []
    for _, r in work.iterrows():
        normtext = r["_text_norm"]
        rec = {"filename": r["_filename"], "year": r["_year"], "region": r["_region"], "country": r["_country"]}
        total = 0
        for k in keywords:
            phrases = exp_map.get(k, [])
            c = count_occurrences_normtext(normtext, phrases, whole_word=whole_word)
            rec[k] = c
            total += c
        rec["total_count"] = total
        rows.append(rec)
    out = pd.DataFrame(rows)
    try:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    except Exception:
        pass
    return out

def compute_doc_snippets(exp_map: Dict[str, List[str]], whole_word: bool, window: int = 1) -> pd.DataFrame:
    """Per-document consolidated snippet strings (±window), across all keywords."""
    rows = []
    for _, r in work.iterrows():
        raw = r["_text_raw"]
        norm = r["_text_norm"]
        all_snips = []
        total = 0
        for k in keywords:
            phrases = exp_map.get(k, [])
            if not phrases:
                continue
            total += count_occurrences_normtext(norm, phrases, whole_word=whole_word)
            snips = extract_snippets_for_doc(raw, phrases, window=window, whole_word=whole_word)
            all_snips.extend(snips)
        # de-duplicate while preserving order
        seen, uniq = set(), []
        for s in all_snips:
            key = fold_accents(s)[:500]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
        rows.append({
            "filename": r["_filename"],
            "year": r["_year"],
            "region": r["_region"],
            "country": r["_country"],
            "snippet_count": len(uniq),
            "total_count": total,
            "snippets": " ⏐ ".join(uniq)  # validator-friendly, single cell
        })
    out = pd.DataFrame(rows)
    try:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    except Exception:
        pass
    return out

# Compute totals
doc_counts_exact = compute_doc_counts(exp_exact, whole_word=whole_word)
doc_counts_sem   = compute_doc_counts(exp_sem,   whole_word=whole_word)

# Compute snippets (±1 sentence)
doc_snips_exact = compute_doc_snippets(exp_exact, whole_word=whole_word, window=1)
doc_snips_sem   = compute_doc_snippets(exp_sem,   whole_word=whole_word, window=1)

# Filter non-zero
nz_exact_tot = doc_counts_exact[doc_counts_exact["total_count"] > 0].copy()
nz_sem_tot   = doc_counts_sem[doc_counts_sem["total_count"] > 0].copy()

nz_exact_snip = doc_snips_exact[(doc_snips_exact["total_count"] > 0) & (doc_snips_exact["snippet_count"] > 0)].copy()
nz_sem_snip   = doc_snips_sem[(doc_snips_sem["total_count"] > 0) & (doc_snips_sem["snippet_count"] > 0)].copy()

# --- PREVIEWS before downloads ---
st.markdown("### Preview non-zero tables")

# Totals previews
px1, px2 = st.columns(2)
with px1:
    st.caption(f"EXACT totals — {len(nz_exact_tot)} documents")
    if nz_exact_tot.empty:
        st.info("No exact matches.")
    else:
        st.dataframe(nz_exact_tot.sort_values("total_count", ascending=False),
                     use_container_width=True, height=300)

with px2:
    st.caption(f"SEMANTIC totals — {len(nz_sem_tot)} documents")
    if nz_sem_tot.empty:
        st.info("No semantic matches.")
    else:
        st.dataframe(nz_sem_tot.sort_values("total_count", ascending=False),
                     use_container_width=True, height=300)

# Snippets previews
ps1, ps2 = st.columns(2)
with ps1:
    st.caption(f"EXACT snippets (±1 sentence) — {len(nz_exact_snip)} documents")
    if nz_exact_snip.empty:
        st.info("No exact snippet matches.")
    else:
        st.dataframe(nz_exact_snip.sort_values("snippet_count", ascending=False),
                     use_container_width=True, height=320)

with ps2:
    st.caption(f"SEMANTIC snippets (±1 sentence) — {len(nz_sem_snip)} documents")
    if nz_sem_snip.empty:
        st.info("No semantic snippet matches.")
    else:
        st.dataframe(nz_sem_snip.sort_values("snippet_count", ascending=False),
                     use_container_width=True, height=320)

# --- DOWNLOADS ---
d1, d2 = st.columns(2)
with d1:
    label_e_tot = f"⬇️ Download EXACT totals (non-zero, {len(nz_exact_tot)} docs)"
    st.download_button(
        label_e_tot,
        data=nz_exact_tot.to_csv(index=False).encode("utf-8"),
        file_name="keyword_exact_counts_per_document_nonzero.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=nz_exact_tot.empty
    )
    label_e_snip = f"⬇️ Download EXACT snippets (non-zero, {len(nz_exact_snip)} docs)"
    st.download_button(
        label_e_snip,
        data=nz_exact_snip.to_csv(index=False).encode("utf-8"),
        file_name="keyword_exact_snippets_per_document_nonzero.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=nz_exact_snip.empty
    )

with d2:
    label_s_tot = f"⬇️ Download SEMANTIC totals (non-zero, {len(nz_sem_tot)} docs)"
    st.download_button(
        label_s_tot,
        data=nz_sem_tot.to_csv(index=False).encode("utf-8"),
        file_name="keyword_semantic_counts_per_document_nonzero.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=nz_sem_tot.empty
    )
    label_s_snip = f"⬇️ Download SEMANTIC snippets (non-zero, {len(nz_sem_snip)} docs)"
    st.download_button(
        label_s_snip,
        data=nz_sem_snip.to_csv(index=False).encode("utf-8"),
        file_name="keyword_semantic_snippets_per_document_nonzero.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=nz_sem_snip.empty
    )
