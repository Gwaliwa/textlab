# app.py — CSV/Excel links → PDFs → TXT → Sections (Findings / Everything / Custom)
# High-recall Findings extractor:
#  - Accent-folded regex + fuzzy line match (RapidFuzz if available, else difflib)
#  - Broader multilingual synonyms (EN/ES/FR/PT/AR/RU/zh/DE/IT)
#  - Numbered/bulleted heading tolerance, heading-like boundary detection
#  - TOC fallback (PyMuPDF), layout (font-size) fallback, page-window sweep
#  - Optional OCR fallback (pypdfium2 + pytesseract)
#
# pip install streamlit pandas requests pymupdf pdfplumber pypdfium2 pytesseract pillow langdetect
# Optional (better fuzzy): pip install rapidfuzz

from __future__ import annotations
import io, os, re, time, string, zipfile, unicodedata, statistics
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from urllib.parse import urlparse, unquote, urljoin, parse_qs

import requests
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Fuzzy (optional) ----------
try:
    from rapidfuzz import fuzz  # pip install rapidfuzz
    HAVE_FUZZ = True
except Exception:
    import difflib
    HAVE_FUZZ = False

# ---------- PDF engines ----------
BACKEND = None
try:
    import fitz  # PyMuPDF (fast; needed for TOC/layout fallback)
    BACKEND = "pymupdf"
except Exception:
    try:
        import pdfplumber
        BACKEND = "pdfplumber"
    except Exception:
        BACKEND = None

# ---------- OCR fallback ----------
try:
    import pypdfium2 as pdfium
    from PIL import Image
    import pytesseract
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

# ---------- Language detect ----------
try:
    from langdetect import detect as _detect_lang
    HAVE_LANGDETECT = True
except Exception:
    HAVE_LANGDETECT = False


# ==================== Downloader helpers ====================
SAFE_CHARS = f"-_.() {string.ascii_letters}{string.digits}"

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def guess_filename_from_url(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name or "document.pdf"
    if not name.lower().endswith(".pdf"): name += ".pdf"
    return name

def sanitize_filename(name: str) -> str:
    cleaned = "".join(c for c in name if c in SAFE_CHARS).strip()
    return cleaned or "document.pdf"

def pick_filename(url: str, resp: requests.Response) -> str:
    cd = resp.headers.get("Content-Disposition", "") or ""
    m = re.search(r"filename\*\s*=\s*UTF-8''([^;\r\n]+)", cd, re.IGNORECASE)
    if m: return sanitize_filename(unquote(m.group(1)))
    m = re.search(r"filename\*\s*=\s*[^']*''([^;\r\n]+)", cd, re.IGNORECASE)
    if m: return sanitize_filename(unquote(m.group(1)))
    m = re.search(r'filename\s*=\s*"([^"\r\n]+)"', cd, re.IGNORECASE)
    if m: return sanitize_filename(unquote(m.group(1)))
    m = re.search(r"filename\s*=\s*([^;\r\n]+)", cd, re.IGNORECASE)
    if m: return sanitize_filename(unquote(m.group(1).strip()))
    return sanitize_filename(guess_filename_from_url(url))

def looks_like_pdf(resp: requests.Response, url: str, force: bool) -> bool:
    if force: return True
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "application/pdf" in ctype: return True
    try:
        if pick_filename(url, resp).lower().endswith(".pdf"): return True
    except Exception:
        pass
    return url.lower().endswith(".pdf")

def resolve_pdf_from_html(html_text: str, base_url: str) -> Optional[str]:
    m = re.search(r'http-equiv="refresh"[^>]*content="\d+;\s*url=([^"]+)"', html_text, re.IGNORECASE)
    if m: return urljoin(base_url, m.group(1))
    m = re.search(r'href=[\'"]([^\'"]+?\.pdf(?:\?[^\'"]*)?)[\'"]', html_text, re.IGNORECASE)
    if m: return urljoin(base_url, m.group(1))
    m = re.search(r'(?:iframe|embed)[^>]+src=[\'"]([^\'"]+?\.pdf(?:\?[^\'"]*)?)[\'"]', html_text, re.IGNORECASE)
    if m: return urljoin(base_url, m.group(1))
    return None

def transform_direct_pdf_url(url: str) -> str:
    try:
        parsed = urlparse(url); host = (parsed.netloc or "").lower(); path = parsed.path or ""; qs = parse_qs(parsed.query)
        if "dropbox.com" in host:
            if qs.get("dl", ["0"])[0] != "1": return url + ("&" if parsed.query else "?") + "dl=1"
        if "drive.google.com" in host:
            m = re.search(r"/file/d/([^/]+)/view", path)
            if m: return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
            if "open" in path and "id" in qs: return f"https://drive.google.com/uc?export=download&id={qs['id'][0]}"
        if any(h in host for h in ["sharepoint.com","onedrive.live.com","1drv.ms","box.com"]):
            if "download=1" not in url.lower(): return url + ("&" if parsed.query else "?") + "download=1"
        if "github.com" in host and "/blob/" in path:
            return url.replace("/blob/", "/raw/")
    except Exception:
        pass
    return url

def sniff_for_pdf_link(resp: requests.Response) -> Optional[str]:
    try:
        head = b""
        for chunk in resp.iter_content(chunk_size=8192):
            head += chunk
            if len(head) > 1024 * 1024: break
        text = head.decode("utf-8", errors="ignore")
        return resolve_pdf_from_html(text, resp.url)
    except Exception:
        return None

def download_one(
    url: str,
    out_dir: Path,
    timeout: int,
    force: bool,
    user_agent: str,
    verify_ssl: bool,
    chunk_size: int,
    max_bytes: Optional[int],
    max_retries: int,
    follow_html: bool
) -> Tuple[str, str, Optional[str]]:
    if not isinstance(url, str) or not url.strip():
        return ("skipped", "", "blank url")
    url = url.strip()
    session = requests.Session()
    headers = {"User-Agent": user_agent, "Accept": "*/*"}
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            with session.get(url, headers=headers, stream=True, timeout=timeout, verify=verify_ssl) as r:
                if r.status_code >= 400: return ("error", "", f"HTTP {r.status_code}")
                if not looks_like_pdf(r, url, force):
                    if follow_html:
                        cand = sniff_for_pdf_link(r)
                        if cand:
                            return download_one(cand, out_dir, timeout, force, user_agent, verify_ssl, chunk_size, max_bytes, 0, False)
                    ctype = r.headers.get("Content-Type"); return ("skipped", "", f"Not a PDF (Content-Type: {ctype})")
                fname = pick_filename(r.url, r)
                dest = out_dir / fname
                base, ext = dest.stem, dest.suffix; i = 1
                while dest.exists(): dest = out_dir / f"{base}_{i}{ext}"; i += 1
                bytes_written = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk: continue
                        f.write(chunk); bytes_written += len(chunk)
                        if max_bytes and bytes_written > max_bytes: raise RuntimeError("Exceeded max_bytes limit")
                return ("ok", str(dest), None)
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries: time.sleep(min(2 ** attempt, 3))
    return ("error", "", last_err or "unknown error")

def read_table(file, csv_encoding: str | None = None, csv_delimiter: Optional[str] = None) -> pd.DataFrame:
    name = getattr(file, "name", "uploaded"); lower = name.lower()
    try: file.seek(0)
    except Exception: pass
    data = file.read()
    if lower.endswith((".xlsx",".xls")) or data[:2]==b"PK":
        try: return pd.read_excel(io.BytesIO(data))
        except Exception: pass
    encodings_to_try = [csv_encoding] if csv_encoding else ["utf-8","utf-8-sig","cp1252","latin1"]
    sep = None if not csv_delimiter else csv_delimiter
    last_err = None
    for enc in encodings_to_try:
        try:
            text = data.decode(enc)
            return pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e; continue
    try:
        text = data.decode("latin1", errors="replace")
        return pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
    except Exception:
        pass
    try: return pd.read_excel(io.BytesIO(data))
    except Exception as e: raise RuntimeError(f"Unable to parse file. Last error: {last_err}")

def find_candidate_link_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        lcol = str(col).lower()
        if any(k in lcol for k in ["url","link","href"]):
            candidates.append(col); continue
        vals = df[col].dropna().astype(str)
        if len(vals)==0: continue
        hits = vals.str.match(r"^https?://", na=False).sum()
        if hits/max(len(vals),1) >= 0.25: candidates.append(col)
    seen, out = set(), []
    for c in candidates:
        if c not in seen:
            out.append(c); seen.add(c)
    return out


# ==================== Extraction (PyMuPDF / pdfplumber / OCR) ====================
def extract_text_pymupdf(pdf_path: Path, max_pages: int = 0) -> str:
    text = []
    with fitz.open(pdf_path) as doc:
        end = doc.page_count if max_pages <= 0 else min(max_pages, doc.page_count)
        for i in range(end):
            text.append(doc.load_page(i).get_text("text") or "")
    return "\n".join(text).strip()

def extract_text_pdfplumber(pdf_path: Path, max_pages: int = 0) -> str:
    import pdfplumber as _pl
    text = []
    with _pl.open(str(pdf_path)) as pdf:
        end = len(pdf.pages) if max_pages <= 0 else min(max_pages, len(pdf.pages))
        for i in range(end):
            text.append(pdf.pages[i].extract_text() or "")
    return "\n".join(text).strip()

def ocr_pdf(pdf_path: Path, dpi: int = 200, langs: str = "eng", max_pages: int = 0) -> str:
    if not HAVE_OCR: return ""
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        n = len(pdf)
        end = n if max_pages <= 0 else min(max_pages, n)
        out = []
        for i in range(end):
            page = pdf[i]
            pil = page.render(scale=dpi/72.0).to_pil()
            txt = pytesseract.image_to_string(pil, lang=langs) or ""
            if txt: out.append(txt)
        return "\n".join(out).strip()
    except Exception:
        return ""

def extract_text_hybrid(pdf_path: Path, max_pages: int, auto_ocr_min_chars: int, ocr_dpi: int, ocr_langs: str) -> str:
    txt = ""
    if BACKEND == "pymupdf":
        try: txt = extract_text_pymupdf(pdf_path, max_pages=max_pages)
        except Exception: txt = ""
    if not txt and BACKEND == "pdfplumber":
        try: txt = extract_text_pdfplumber(pdf_path, max_pages=max_pages)
        except Exception: txt = ""
    if (not txt or len(txt) < auto_ocr_min_chars) and HAVE_OCR:
        ocr_txt = ocr_pdf(pdf_path, dpi=ocr_dpi, langs=ocr_langs, max_pages=max_pages)
        if len(ocr_txt) > len(txt): txt = ocr_txt
    return txt or ""


# ==================== Language detection (never None) ====================
def _script_hint(sample: str) -> str | None:
    for ch in sample:
        name = unicodedata.name(ch, "")
        if "CYRILLIC" in name:   return "ru"
        if "ARABIC" in name:     return "ar"
        if "HEBREW" in name:     return "he"
        if "DEVANAGARI" in name: return "hi"
        if "THAI" in name:       return "th"
        if "HANGUL" in name:     return "ko"
        if "HIRAGANA" in name or "KATAKANA" in name: return "ja"
        if "CJK UNIFIED" in name: return "zh"
    return None

def detect_lang_fast(text: str, sample_chars: int = 4000) -> str:
    s = (text or "").strip()
    if not s: return "unknown"
    s = s[:sample_chars]
    hint = _script_hint(s)
    if hint: return hint
    try:
        if HAVE_LANGDETECT:
            out = _detect_lang(s)
            return out or "unknown"
    except Exception:
        pass
    ascii_ratio = sum(1 for c in s if ord(c) < 128) / max(1, len(s))
    return "en" if ascii_ratio > 0.97 else "unknown"


# ==================== Findings extractor (supercharged) ====================
# Base starts/stops
FINDINGS_STARTS = [
    # English
    "findings of the evaluation","key findings","main findings","findings and analysis","results and findings","evaluation findings",
    "findings and conclusions","synthesis of findings","summary of findings",
    # Spanish
    "hallazgos y análisis de datos","hallazgos de la evaluación","hallazgos","hallazgos clave","principales hallazgos",
    "resultados y hallazgos","resultados de la evaluación",
    # French
    "constats","principaux constats","résultats et constats","résultats de l’évaluation","constatations",
    # Portuguese
    "em destaque","resultados e conclusões preliminares","constatações","achados","achados da avaliação","principais achados",
    # Arabic
    "النتائج","نتائج التقييم","أبرز النتائج","الاستنتاجات والنتائج",
    # Russian
    "основные результаты","результаты","выводы","итоги оценки",
    # Chinese
    "主要发现","研究发现","评估发现",
    # German
    "wesentliche feststellungen","ergebnisse der bewertung","schlüsselbefunde","befunde",
    # Italian
    "risultati della valutazione","principali risultati","risultati e constatazioni","risultati",
    # Generic
    "findings"
]
FINDINGS_STOPS = [
    # English
    "conclusions and recommendations","recommendations","recommendation","conclusions","limitations","scope and limitations",
    "annex","annexes","appendix","appendices","methodology","methods","discussion",
    # FR/ES/PT/IT/DE
    "recommandations","conclusiones","conclusões finais","conclusioni","schlussfolgerungen","empfehlungen",
    "méthodologie","methodologie","metodologia","metodología","methoden",
    "anexo","anexos","annexe","appendice","anhang",
    "discussion and conclusions","lessons learned","implications",
    # Arabic
    "التوصيات","الخلاصة","الاستنتاجات","المنهجية","الملحق",
    # Russian
    "рекомендации","заключения","методология","приложение",
    # Chinese
    "建议","结论","方法论","附录"
]

# Extended lists
FINDINGS_STARTS_EXT = FINDINGS_STARTS + [
    "summary of findings","synthesis of findings","assessment findings",
    "findings overview","evidence and findings",
    "síntesis de hallazgos","resumen de hallazgos",
    "principaux constats","synthèse des constats",
    "achados principais","síntese dos achados",
    "zusammenfassung der feststellungen",
    "principali risultati","sintesi dei risultati",
]
FINDINGS_STOPS_EXT  = FINDINGS_STOPS + [
    "lecciones aprendidas","leçons apprises","lições aprendidas","lezioni apprese",
    "schlussfolgerungen und empfehlungen"
]

# Root regex for starts/stops (incl. stems)
FINDINGS_ROOT_RE  = re.compile(
    r"\b(findings?|hallazg\w+|constat\w+|constataç\w+|achad\w+|résultat\w+|resultad\w+|"
    r"вывод\w+|результ\w+|النتائج|发现|feststell\w+|befund\w+|risultat\w+)\b", re.IGNORECASE)
STOPS_ROOT_RE = re.compile(
    r"\b(recommend(ation|ations)?|recommandation\w*|recomendaç\w+|recomendaci\w+|empfehlung\w*|"
    r"conclusion\w*|schlussfolger\w*|заключени\w+|结论|"
    r"methodolog\w*|méthodolog\w*|metodolog\w*|method\w*|方法论|المنهجية|методолог\w+|"
    r"annex\w*|appendix|appendice\w*|anhang|附录|discussion)\b",
    re.IGNORECASE
)

# Heading-like detector
_HEADING_LIKE = re.compile(
    r"""^\s*(?:[•\-\u2013\u2014]\s*)?(?:[IVXLCDM]+|[A-Z]|\d+(?:\.\d+){0,3})[\)\.]?\s+
        (?:[A-Z][A-Z0-9 &/\-\u2013\u2014]{2,}|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,8}))\s*$""",
    re.VERBOSE
)

def _dehyphenate(text: str) -> str:
    text = re.sub(r'(\w)[\-\u2010-\u2015]\n(\w)', r'\1\2', text)
    text = text.replace('\r','')
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("-\n", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def _build_heading_regex(terms: list[str]) -> re.Pattern:
    alts = []
    for t in terms:
        t = _fold(t).lower()
        if t: alts.append(re.escape(t))
    if not alts: alts = [r"$^"]
    # allow bullets/numbering
    pat = r"^\s*(?:[•\-–—]\s*|\(?[A-Za-z]{1,4}\)|[A-Za-z]{1,4}[\)\.]|\(?\d+(?:\.\d+){0,3}\)?[\)\.]?)?\s*(?:" + "|".join(alts) + r")\b"
    return re.compile(pat, flags=re.IGNORECASE)

def _heading_like_candidates(lines: list[str], start_line: int = 0) -> list[int]:
    idxs = []
    for i in range(start_line, len(lines)):
        ln = _fold(lines[i])
        if len(ln) > 140:
            continue
        if _HEADING_LIKE.match(ln):
            idxs.append(i)
    return idxs

def _next_boundary_after(lines: list[str], start_idx: int) -> int:
    cands = _heading_like_candidates(lines, start_idx + 1)
    return cands[0] if cands else -1

def _line_offsets(lines_raw: list[str]) -> list[int]:
    offs, total = [0], 0
    for ln in lines_raw[:-1]:
        total += len(ln) + 1
        offs.append(total)
    return offs

def _find_start_line(text: str, terms: list[str], fuzzy_thresh: int = 74) -> int:
    raw = _dehyphenate(text)
    lines = raw.split("\n")
    folded = [_fold(ln) for ln in lines]

    # regex
    rx = _build_heading_regex(terms)
    for i, ln in enumerate(folded):
        if rx.search(ln):
            return i

    # fuzzy (root-gated)
    term_fold = [_fold(t).lower() for t in terms if t.strip()]
    best_i, best_s = -1, -1
    for i, ln in enumerate(folded):
        low = ln.lower()
        if not FINDINGS_ROOT_RE.search(low) and all(w not in low for w in ("finding","hallazg","constat","achad","résultat","result","вывод","результ","النتائج","发现","feststell","befund","risultat")):
            continue
        for t in term_fold:
            if HAVE_FUZZ:
                s = max(fuzz.partial_ratio(low, t), fuzz.token_set_ratio(low, t))
            else:
                s = int(100 * difflib.SequenceMatcher(None, low, t).ratio())
            if _HEADING_LIKE.match(ln): s += 8  # small boost if heading-like
            if s > best_s:
                best_s, best_i = s, i
    return best_i if best_s >= fuzzy_thresh else -1

def extract_block_by_roots(text: str) -> str:
    if not text: return ""
    raw = text.replace("\r","")
    m = FINDINGS_ROOT_RE.search(raw)
    if not m: return ""
    start_line = raw.rfind("\n", 0, m.start())
    a = 0 if start_line == -1 else start_line + 1
    stop = STOPS_ROOT_RE.search(raw, pos=a+1)
    b = stop.start() if stop else len(raw)
    if b - a < 40 and stop:
        return ""
    return raw[a:b].strip()

def extract_findings_via_toc(pdf_path: Path, max_pages_from_toc: int = 12) -> str:
    if BACKEND != "pymupdf": return ""
    try:
        doc = fitz.open(str(pdf_path))
        toc = doc.get_toc() or []
        if not toc: return ""
        best = None
        for level, title, page in toc:
            title_fold = _fold(title).lower()
            if FINDINGS_ROOT_RE.search(title_fold) or any(k in title_fold for k in ("finding","hallazg","constat","achad","résultat","result","вывод","результ","النتائج","发现","feststell","befund","risultat")):
                best = page; break
            for t in FINDINGS_STARTS_EXT:
                t_fold = _fold(t).lower()
                if HAVE_FUZZ:
                    sc = max(fuzz.partial_ratio(title_fold, t_fold), fuzz.token_set_ratio(title_fold, t_fold))
                    if sc >= 85: best = page; break
                else:
                    if difflib.SequenceMatcher(None, title_fold, t_fold).ratio() >= 0.80:
                        best = page; break
            if best is not None: break
        if best is None: return ""
        sp = max(1, int(best)) - 1
        ep = min(doc.page_count - 1, sp + max_pages_from_toc - 1)
        parts = []
        for p in range(sp, ep + 1):
            parts.append(doc.load_page(p).get_text("text") or "")
        return "\n".join(parts).strip()
    except Exception:
        return ""

def extract_findings_layout_pymupdf(pdf_path: Path, max_span_pages: int = 10) -> str:
    if BACKEND != "pymupdf": return ""
    try:
        doc = fitz.open(str(pdf_path))
        for pno in range(min(doc.page_count, 40)):
            page = doc.load_page(pno)
            d = page.get_text("dict")
            sizes = []
            for b in d.get("blocks", []):
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        sizes.append(s.get("size", 0))
            if not sizes: continue
            median = sorted(sizes)[len(sizes)//2]
            for b in d.get("blocks", []):
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        txt = _fold(s.get("text",""))
                        if not txt: continue
                        if s.get("size", 0) >= max(median*1.08, 12.5) and FINDINGS_ROOT_RE.search(txt.lower()):
                            parts = [page.get_text("text") or ""]
                            end_page = min(doc.page_count-1, pno + max_span_pages - 1)
                            for q in range(pno+1, end_page+1):
                                parts.append(doc.load_page(q).get_text("text") or "")
                            return "\n".join(parts).strip()
        return ""
    except Exception:
        return ""

def extract_findings_supercharged(pdf_path: Path, text: str, extra_starts=None, extra_stops=None) -> str:
    """Order:
       1) start → next heading-like boundary
       2) neighbor (Methodology → Recommendations)
       3) root-block slice
       4) TOC pages (PyMuPDF)
       5) Layout (font-size) fallback
       6) Page-window sweep (findings-ish density)
    """
    if not text: return ""
    text = _dehyphenate(text)
    starts = (FINDINGS_STARTS_EXT + (extra_starts or []))
    stops  = (FINDINGS_STOPS_EXT  + (extra_stops  or []))

    raw_lines = text.split("\n")
    offs = _line_offsets(raw_lines)

    # 1) start → next boundary
    si = _find_start_line(text, starts, fuzzy_thresh=72)
    if si != -1:
        stop_line = _find_start_line("\n".join(raw_lines[si+1:]), stops, fuzzy_thresh=70)
        if stop_line != -1:
            sj = si + 1 + stop_line
        else:
            nb = _next_boundary_after(raw_lines, si)
            sj = nb if nb != -1 else -1
        a = offs[si] if si < len(offs) else 0
        b = (offs[sj] if (sj != -1 and sj < len(offs)) else len(text))
        if b - a >= 40:
            return text[a:b].strip()

    # 2) neighbor window
    meth = _find_start_line(text, ["methodology","methodologie","méthodologie","metodologia","metodología","المنهجية","методология","方法论","methoden"], fuzzy_thresh=70)
    reco = _find_start_line(text, ["recommendations","recommandations","recomendações","recomendaciones","empfehlungen","raccomandazioni","التوصيات","рекомендации","建议"], fuzzy_thresh=70)
    if meth != -1 and reco != -1 and reco > meth:
        a = offs[meth] if meth < len(offs) else 0
        b = offs[reco] if reco < len(offs) else len(text)
        if b - a >= 40:
            return text[a:b].strip()

    # 3) root-block slice
    rb = extract_block_by_roots(text)
    if rb: return rb

    # 4) TOC
    toc_try = extract_findings_via_toc(pdf_path, max_pages_from_toc=12)
    if toc_try: return toc_try

    # 5) Layout heuristics
    layout_try = extract_findings_layout_pymupdf(pdf_path, max_span_pages=10)
    if layout_try: return layout_try

    # 6) Page-window sweep
    try:
        if BACKEND == "pymupdf":
            doc = fitz.open(str(pdf_path))
            N = min(doc.page_count, 60)
            def score_block(s: str) -> float:
                s_low = _fold(s).lower()
                hits = 0
                for kw in FINDINGS_STARTS_EXT:
                    if kw.lower() in s_low: hits += 2
                if "finding" in s_low: hits += 1
                roots = len(FINDINGS_ROOT_RE.findall(s_low))
                stops = len(STOPS_ROOT_RE.findall(s_low))
                return hits + roots - 0.2 * stops
            best_s, best_text = -1e9, ""
            window = 4
            for i in range(0, N):
                parts = []
                for j in range(i, min(N, i+window)):
                    parts.append(doc.load_page(j).get_text("text") or "")
                blk = "\n".join(parts).strip()
                sc = score_block(blk)
                if sc > best_s and len(blk) > 80:
                    best_s, best_text = sc, blk
            if best_text:
                return best_text
    except Exception:
        pass

    return ""


# ==================== Custom-headings extractor ====================
def extract_sections_by_keywords(text: str, ordered_map: List[Tuple[str, List[str]]]) -> Dict[str, str]:
    result = {h: "" for h, _ in ordered_map}
    if not text: return result
    t = text.replace("-\n", "").replace("\r", "")
    low = t.lower()
    hits: List[Tuple[int, str]] = []
    for heading, kws in ordered_map:
        best_pos = None
        for kw in kws:
            kw = kw.strip()
            if not kw: continue
            pos = low.find(kw.lower())
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
        if best_pos is not None:
            ls = t.rfind("\n", 0, best_pos)
            best_pos = 0 if ls == -1 else (ls + 1)
            hits.append((best_pos, heading))
    if not hits: return result
    hits.sort(key=lambda x: x[0])
    for i, (start_pos, heading) in enumerate(hits):
        end_pos = hits[i + 1][0] if i + 1 < len(hits) else len(t)
        result[heading] = t[start_pos:end_pos].strip()
    return result


# ==================== UI ====================
st.set_page_config(page_title="Bulk PDF Downloader → TXT → Sections", page_icon="📥", layout="wide")
st.title("📝 Text Lab")

with st.sidebar:
    st.header("Download")
    max_workers = st.slider("Parallel downloads", 1, 16, 6)
    timeout = st.slider("Timeout (sec)", 5, 120, 30)
    force_pdf = st.checkbox("Force download if not labeled PDF", value=False)
    follow_html = st.checkbox("Find PDFs in landing pages", value=True)
    direct_fixes = st.checkbox("Direct-download fixes (Drive/Dropbox/etc.)", value=True)
    verify_ssl = st.checkbox("Verify SSL", value=True)
    max_bytes = st.number_input("Max bytes/file (0 = no limit)", min_value=0, value=0, step=1)
    user_agent = st.text_input("User-Agent", value="Mozilla/5.0 (compatible; PDFDownloader/1.0)")
    csv_encoding = st.selectbox("CSV encoding", ["Auto (try common)", "utf-8","utf-8-sig","cp1252","latin1"], index=0)
    csv_delimiter = st.text_input("CSV delimiter (blank = auto)", value="")

    st.markdown("---")
    st.header("Extraction / OCR")
    max_pages = st.slider("Max pages per PDF (0 = all)", 0, 60, 12)
    auto_ocr_min_chars = st.slider("Auto-OCR if extracted text < N chars", 0, 2000, 120, 10)
    ocr_dpi = st.slider("OCR render DPI", 120, 300, 220, 10)
    ocr_langs = st.text_input("Tesseract languages", "eng+spa+fra+por+deu+ita+ara+rus+chi_sim")
    extract_workers = st.slider("Parallel extract workers", 1, 16, 8)

    st.markdown("---")
    st.header("Language (optional)")
    lang_sample = st.slider("Language detection sample chars", 500, 8000, 3000, 500)

left, right = st.columns([2,1])
with left:
    uploaded = st.file_uploader(
        "Upload CSV or Excel with PDF links + (optionally) Year/Country/Region",
        type=["csv","xlsx","xls"],
        accept_multiple_files=False
    )
with right:
    base_out_dir = st.text_input("Base output folder", value="downloads")
    add_timestamp = st.checkbox("Add timestamp subfolder", value=True)
    zip_after = st.checkbox("Zip downloaded PDFs", value=False)

mode = st.radio(
    "Section mode",
    ["Evaluation Findings (default)", "Everything (single column)", "Custom headings (up to 4)"],
    index=0,
    horizontal=True
)

# Extra keywords (optional)
extra_findings_starts, extra_findings_stops = [], []
missing_policy = "insert_full"
if mode == "Evaluation Findings (default)":
    with st.expander("Tweak Findings keywords (optional)"):
        s = st.text_area("Extra 'Findings' start phrases (comma/newline separated)", value="")
        t = st.text_area("Extra stop phrases (comma/newline separated)", value="")
        extra_findings_starts = [w.strip() for w in re.split(r"[,\n]", s) if w.strip()]
        extra_findings_stops  = [w.strip() for w in re.split(r"[,\n]", t) if w.strip()]
    missing_policy = st.radio(
        "When Findings missing after all attempts:",
        ["Insert full document text", "Skip this document"],
        index=0, horizontal=True
    )
    missing_policy = "insert_full" if missing_policy.startswith("Insert") else "skip"

# Custom headings (user-entered)
ordered_map: List[Tuple[str, List[str]]] = []
if mode == "Custom headings (up to 4)":
    st.subheader("Custom headings (enter simple keywords; order matters)")
    st.caption("Add up to 4 rows. In **keywords**, enter comma- or newline-separated phrases near that heading.")
    default_rows = pd.DataFrame([
        {"heading": "context", "keywords": "situation update, context, background"},
        {"heading": "contributions", "keywords": "major contributions, results achieved"},
        {"heading": "collaborations", "keywords": "UN collaboration, partnerships"},
        {"heading": "innovations", "keywords": "lessons learned, innovation"},
    ])
    headings_kw_df = st.data_editor(
        default_rows, num_rows="dynamic", use_container_width=True,
        column_config={
            "heading": st.column_config.TextColumn("heading"),
            "keywords": st.column_config.TextColumn("keywords (comma/newline separated)"),
        },
        key="headings_kw_editor",
    )
    for _, r in headings_kw_df.iterrows():
        h = str(r.get("heading","")).strip()
        k = str(r.get("keywords","")).strip()
        if not h or not k: continue
        kw_list = [s.strip() for s in re.split(r"[,\n]", k) if s.strip()]
        if kw_list: ordered_map.append((h, kw_list))


# ==================== Pipeline ====================
if uploaded is not None:
    # Read table
    try:
        df = read_table(uploaded, None if csv_encoding.startswith("Auto") else csv_encoding, (csv_delimiter or None))
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        df = None

    if df is not None and not df.empty:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # Link column
        candidates = find_candidate_link_columns(df)
        try:
            default_index = list(df.columns).index(candidates[0]) if candidates else 0
        except ValueError:
            default_index = 0
        link_col = st.selectbox("Select the column with PDF links", options=list(df.columns), index=default_index)

        # Metadata columns
        st.subheader("Map metadata columns (from your uploaded file)")
        meta_cols = ["(none)"] + list(df.columns)

        def _auto(colnames, keys):
            s = {str(c).strip().lower(): c for c in colnames}
            for k in keys:
                if k in s: return s[k]
            return "(none)"

        year_col_default = _auto(df.columns, ["year"])
        country_col_default = _auto(df.columns, ["country", "country name"])
        region_col_default = _auto(df.columns, ["region", "unicef_region"])

        c1, c2, c3 = st.columns(3)
        with c1:
            year_col = st.selectbox("Year column", options=meta_cols,
                                    index=meta_cols.index(year_col_default) if year_col_default in meta_cols else 0)
        with c2:
            country_col = st.selectbox("Country column", options=meta_cols,
                                       index=meta_cols.index(country_col_default) if country_col_default in meta_cols else 0)
        with c3:
            region_col = st.selectbox("Region column", options=meta_cols,
                                      index=meta_cols.index(region_col_default) if region_col_default in meta_cols else 0)

        # Collect URLs + metadata mapping
        urls_series = df[link_col].dropna().astype(str).str.strip()
        urls_series = urls_series[urls_series.str.match(r"^https?://")]
        source_urls = urls_series.tolist()
        final_urls = [transform_direct_pdf_url(u) for u in source_urls] if direct_fixes else source_urls

        meta_by_source: Dict[str, Dict[str, object]] = {}
        src_indices = urls_series.index.tolist()
        for src, idx in zip(source_urls, src_indices):
            if src not in meta_by_source:
                row = df.loc[idx]
                y = (None if year_col == "(none)" else row.get(year_col))
                c = (None if country_col == "(none)" else row.get(country_col))
                r = (None if region_col == "(none)" else row.get(region_col))
                meta_by_source[src] = {
                    "year": (int(y) if pd.notna(y) and str(y).isdigit() else (y if pd.notna(y) else None)),
                    "country": (str(c).strip() if pd.notna(c) else None),
                    "region": (str(r).strip() if pd.notna(r) else None),
                }

        st.info(f"Found {len(source_urls)} link(s) in **{link_col}**.")
        go = st.button("🚀 Download + Extract + Consolidate")

        if go and source_urls:
            ts_name = time.strftime("%Y%m%d-%H%M%S")
            out_dir = Path(base_out_dir)
            if add_timestamp: out_dir = out_dir / ts_name
            ensure_dir(out_dir)

            # ---------- DOWNLOAD ----------
            results = []
            progress = st.progress(0)
            status_area = st.empty()
            start = time.time()

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        download_one, f_url, out_dir, timeout, force_pdf, user_agent, verify_ssl,
                        1024 * 256, None if (max_bytes or 0) <= 0 else max_bytes, 2, follow_html
                    ): (s_url, f_url)
                    for s_url, f_url in zip(source_urls, final_urls)
                }
                done = 0
                total = len(futs)
                for fut in as_completed(futs):
                    s_url, f_url = futs[fut]
                    try:
                        status, path, err = fut.result()
                    except Exception as e:
                        status, path, err = ("error", "", str(e))
                    results.append({"source_url": s_url, "final_url": f_url, "status": status, "pdf_saved_path": path, "error": err or ""})
                    done += 1
                    progress.progress(done/max(total,1))
                    status_area.write(f"Processed {done}/{total} — last: {s_url} → {status}")

            st.success(f"Downloaded in {time.time()-start:.1f}s → `{out_dir}`")

            if zip_after:
                zip_path = out_dir.with_suffix(".zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in out_dir.glob("*.pdf"):
                        if p.is_file(): zf.write(p, arcname=p.name)
                st.info(f"ZIP created: `{zip_path}`")

            res_df = pd.DataFrame(results)
            st.subheader("Download results")
            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                "⬇️ Download results log (CSV)",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name=f"download_log_{ts_name}.csv",
                mime="text/csv",
                use_container_width=True
            )

            # ---------- EXTRACT + CONSOLIDATE ----------
            st.subheader("Extracting text and consolidating…")
            if BACKEND is None and not HAVE_OCR:
                st.error("No text backend. Install PyMuPDF (`pip install pymupdf`) or pdfplumber; OCR needs pypdfium2 + pytesseract.")
            else:
                txt_dir = ensure_dir(out_dir / "texts")
                ok = res_df[res_df["status"].eq("ok")]
                items = [(r["source_url"], r["final_url"], Path(r["pdf_saved_path"])) for _, r in ok.iterrows()]

                rows = []
                skipped_rows = []
                prog2 = st.progress(0.0)

                def process_one(item):
                    src_url, fin_url, pdf_path = item
                    try:
                        text = extract_text_hybrid(
                            pdf_path,
                            max_pages=max_pages,
                            auto_ocr_min_chars=auto_ocr_min_chars,
                            ocr_dpi=ocr_dpi,
                            ocr_langs=ocr_langs or "eng",
                        )
                        lang = detect_lang_fast(text, sample_chars=lang_sample)

                        # save TXT
                        txt_name = pdf_path.with_suffix(".txt").name
                        (txt_dir / txt_name).write_text(text or "", encoding="utf-8")

                        meta = meta_by_source.get(src_url, {"year": None, "country": None, "region": None})

                        row = {
                            "filename": pdf_path.name,
                            "source_url": src_url,
                            "final_url": fin_url,
                            "pdf_saved_path": str(pdf_path),
                            "txt_saved_path": str(txt_dir / txt_name),
                            "detected_lang": (lang or "unknown"),
                            "year": meta.get("year"),
                            "country": meta.get("country"),
                            "region": meta.get("region"),
                        }

                        if mode == "Everything (single column)":
                            row["full_text"] = text or ""
                            return row

                        if mode == "Custom headings (up to 4)":
                            sections = extract_sections_by_keywords(text or "", ordered_map)
                            row.update(sections)
                            return row

                        # Findings: supercharged pipeline
                        findings = extract_findings_supercharged(
                            pdf_path,
                            text or "",
                            extra_starts=extra_findings_starts,
                            extra_stops=extra_findings_stops
                        )
                        missing = (len((findings or "").strip()) == 0)

                        if missing:
                            if missing_policy == "insert_full":
                                findings = text or ""
                                row["Findings of the evaluation"] = findings
                                row["findings_missing"] = False
                                row["filled_with_full_text"] = True
                                return row
                            else:  # skip
                                return {
                                    "_skip": True,
                                    "filename": pdf_path.name,
                                    "source_url": src_url,
                                    "year": meta.get("year"),
                                    "country": meta.get("country"),
                                    "region": meta.get("region"),
                                    "reason": "Findings not detected"
                                }

                        row["Findings of the evaluation"] = findings
                        row["findings_missing"] = False
                        row["filled_with_full_text"] = False
                        return row

                    except Exception as e:
                        return {
                            "_skip": True,
                            "filename": pdf_path.name, "source_url": src_url,
                            "year": None, "country": None, "region": None,
                            "reason": f"error: {e}"
                        }

                if items:
                    with ThreadPoolExecutor(max_workers=extract_workers) as ex:
                        futs = [ex.submit(process_one, it) for it in items]
                        done, total = 0, len(futs)
                        for fut in as_completed(futs):
                            r = fut.result()
                            if isinstance(r, dict) and r.get("_skip"):
                                skipped_rows.append(r)
                            else:
                                rows.append(r)
                            done += 1
                            prog2.progress(done/max(1,total))

                consolidated = pd.DataFrame(rows) if rows else pd.DataFrame()
                if not consolidated.empty:
                    st.success(f"Consolidated {len(consolidated)} file(s).")
                    st.dataframe(consolidated.head(50), use_container_width=True)

                    # Save for other pages
                    st.session_state["consolidated"] = consolidated.copy()
                    st.session_state["consolidated_df"] = consolidated.copy()

                    # Export consolidated & TXT ZIP
                    st.download_button(
                        "💾 Download Consolidated CSV",
                        data=consolidated.to_csv(index=False).encode("utf-8"),
                        file_name=f"consolidated_{ts_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    txt_zip = io.BytesIO()
                    with zipfile.ZipFile(txt_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                        for p in txt_dir.glob("*.txt"):
                            if p.is_file(): zf.write(p, arcname=p.name)
                    txt_zip.seek(0)
                    st.download_button(
                        "⬇️ Download all TXT (ZIP)",
                        data=txt_zip.getvalue(),
                        file_name=f"texts_{ts_name}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

                # Skipped report (if any)
                if skipped_rows:
                    st.warning(f"{len(skipped_rows)} file(s) skipped (Findings not detected).")
                    skipped_df = pd.DataFrame(skipped_rows)
                    st.dataframe(skipped_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download skipped list (CSV)",
                        data=skipped_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"skipped_{ts_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        elif go and not source_urls:
            st.warning("No valid http(s) links detected in the chosen column.")


# ==================== Diagnostics ====================
with st.expander("Environment diagnostics", expanded=False):
    st.write(f"PDF backend: **{BACKEND or 'none'}** (PyMuPDF fastest; needed for TOC/layout fallback)")
    st.write(f"OCR available: **{HAVE_OCR}** (pypdfium2 + pytesseract + Pillow + system Tesseract)")
    st.write(f"Language detection available: **{HAVE_LANGDETECT}**")
    st.write(f"Fuzzy matching: **{'RapidFuzz' if HAVE_FUZZ else 'difflib'}**")
