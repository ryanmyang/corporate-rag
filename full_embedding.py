#!/usr/bin/env python
"""
make_chunks.py – Streamlit PDF → semantic chunks → embeddings → uploadable JSON
• Logs every stage under data/embedding_logs/<file>/<timestamp>/
• GPT‑4o Vision OCR fallback for slide pages (<1 000 chars)
• Doc‑level GPT‑3.5 metadata  → company / quarter / medium / sections
• Chunk‑level GPT‑3.5 metadata → section classification only
"""

from __future__ import annotations
import os, re, json, time, base64, logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import fitz                       # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
from rich.console import Console

# ───── env & OpenAI ───────────────────────────────────────────────────────
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)
EMBED_MODEL  = "text-embedding-ada-002"
META_MODEL   = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"

EMBED_DIM  = 1536
MAX_TOK    = 700
OVERLAP_TOK = 100
LOG_ROOT   = Path("data/embedding_logs")

console = Console()
enc      = tiktoken.get_encoding("cl100k_base")
sentence_end = re.compile(r"(?<=[.!?])\s+")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("chunker")

# ───── helpers: dirs & OCR ────────────────────────────────────────────────
def create_run_dir(stem: str) -> Path:
    d = LOG_ROOT / stem / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d

def ocr_slide(page: fitz.Page) -> str:
    pix = page.get_pixmap(dpi=200)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role":"system","content":"Extract all visible text from this slide."},
            {"role":"user","content":[
                {"type":"image_url",
                 "image_url":{"url":f"data:image/png;base64,{b64}","detail":"auto"}}
            ]}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# ───── Stage 1: PDF→raw text ──────────────────────────────────────────────
def pdf_to_text(pdf: Path, run_dir: Path) -> str:
    doc = fitz.open(pdf); parts=[]
    for i, page in enumerate(doc, 1):
        txt = page.get_text() or ""
        if len(txt)<1_000:
            log.info("p%d short – using OCR", i)
            try: txt = ocr_slide(page)
            except Exception as e: log.warning("OCR p%d failed: %s", i, e)
        parts.append(txt)
    raw = "\n".join(parts)
    (run_dir/"01_text.txt").write_text(raw, encoding="utf-8")
    return raw

# ───── Stage 2: doc‑level metadata ────────────────────────────────────────
DOC_META_PROMPT = (
    "Extract high‑level metadata for the following document. "
    "Return JSON with keys:\n"
    " company   : ticker or company name\n"
    " quarter   : e.g. \"Q4 2024\" or null\n"
    " medium    : one of [\"transcript\",\"presentation\",\"report\",\"article\"]\n"
    " sections  : ordered list of top‑level section titles present in the doc\n"
    "If unsure, use null. Respond with *only* JSON."
)

def extract_doc_meta(text: str, run_dir: Path) -> Dict[str, str | list]:
    resp = client.chat.completions.create(
        model=META_MODEL,
        messages=[{"role":"system","content":DOC_META_PROMPT},
                  {"role":"user","content":text[:2000]}],
        temperature=0
    )
    meta = json.loads(resp.choices[0].message.content)
    (run_dir/"02_doc_meta.json").write_text(json.dumps(meta, indent=2))
    return meta

# ───── Stage 3: chunking ──────────────────────────────────────────────────
def semantic_chunks(text: str, run_dir: Path) -> List[str]:
    sents = sentence_end.split(text)
    chunks,buff,buff_tok=[], "",0
    for s in sents:
        tl=len(enc.encode(s))
        if buff_tok+tl<=MAX_TOK:
            buff+=s+" "; buff_tok+=tl
        else:
            chunks.append(buff.strip())
            overlap=enc.encode(buff)[-OVERLAP_TOK:] if buff_tok>OVERLAP_TOK else []
            buff=enc.decode(overlap)+s+" "; buff_tok=len(overlap)+tl
    if buff.strip(): chunks.append(buff.strip())
    (run_dir/"03_chunks.json").write_text(json.dumps(chunks,indent=2))
    return chunks

# ───── Stage 4: per‑chunk section classification ──────────────────────────
def classify_section(chunk:str, sections:List[str]) -> str|None:
    prompt = (
        "Given the list of known section titles, return which section this chunk "
        "belongs to. Respond with exactly one of the titles or null.\n"
        f"Sections: {sections}"
    )
    resp = client.chat.completions.create(
        model=META_MODEL,
        messages=[{"role":"system","content":prompt},
                  {"role":"user","content":chunk[:1200]}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().strip('"')

def chunk_metadata(chunks:List[str], doc_meta:Dict, run_dir:Path)->List[str]:
    sections = doc_meta.get("sections") or []
    sec_for_chunk=[]
    for c in chunks:
        sec_for_chunk.append(classify_section(c, sections))
    (run_dir/"04_chunk_meta.json").write_text(
        json.dumps(sec_for_chunk,indent=2))
    return sec_for_chunk

# ───── Stage 5: embeddings ────────────────────────────────────────────────
def embed_chunks(chunks:List[str], run_dir:Path)->List[List[float]]:
    vecs=[]
    for i in range(0,len(chunks),16):
        resp=client.embeddings.create(model=EMBED_MODEL,
                                      input=chunks[i:i+16])
        vecs.extend([d.embedding for d in resp.data])
    emb_log=[{"idx":i,"tokens":len(enc.encode(c))} for i,c in enumerate(chunks)]
    (run_dir/"05_embeddings.json").write_text(json.dumps(emb_log,indent=2))
    return vecs

# ───── Stage 6: assemble docs & batch ─────────────────────────────────────
def assemble_docs(stem:str,chunks,vecs,doc_meta,chunk_secs,pdf_name,run_dir):
    docs=[]
    for i,(txt,vec,sec) in enumerate(zip(chunks,vecs,chunk_secs)):
        docs.append({
            "@search.action":"upload",
            "id":f"{stem}_{i:04}",
            "content":txt,
            "contentVector":vec,
            "company":doc_meta.get("company"),
            "quarter":doc_meta.get("quarter"),
            "medium":doc_meta.get("medium"),
            "section":sec,
            "sourceFile":pdf_name
        })
    path=run_dir/"06_upload_batch.json"
    path.write_text(json.dumps({"value":docs},ensure_ascii=False))
    return path

# ───── orchestrator ───────────────────────────────────────────────────────
def process(pdf: Path)->Path:
    stem = pdf.stem.lower(); run_dir = create_run_dir(stem)
    t0=time.perf_counter()

    raw        = pdf_to_text(pdf, run_dir)
    doc_meta   = extract_doc_meta(raw, run_dir)
    chunks     = semantic_chunks(raw, run_dir)
    chunk_secs = chunk_metadata(chunks, doc_meta, run_dir)
    vecs       = embed_chunks(chunks, run_dir)
    out_path   = assemble_docs(stem,chunks,vecs,doc_meta,chunk_secs,pdf.name,run_dir)

    console.print(f"[green]Done in {time.perf_counter()-t0:0.1f}s → {out_path}")
    return out_path

# ───── Streamlit UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="PDF → Chunk JSON", layout="wide")
st.title("Local RAG Pre‑processor")

upload = st.file_uploader("Upload a PDF", type=["pdf"])
if upload and st.button("Process"):
    tmp = Path(upload.name); tmp.write_bytes(upload.getbuffer())
    out_json = process(tmp)
    st.success(f"JSON ready: {out_json}")
    st.code(out_json.read_text()[:1000]+" …", language="json")