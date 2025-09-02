# trascrizione.py
import os
import re
import time
import math
import tempfile
from datetime import timedelta
import textwrap
from typing import Optional

import streamlit as st
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import ffmpeg

# ---------------------------
# Utilità di formattazione
# ---------------------------
def seconds_to_hms(seconds: float) -> str:
    seconds = max(0, float(seconds or 0.0))
    return str(timedelta(seconds=int(round(seconds))))

def seconds_to_minutes_label(seconds: float, decimals: int = 1) -> str:
    minutes = (seconds or 0.0) / 60.0
    return f"{minutes:.{decimals}f} min"

def safe_probe_duration(path: str) -> float:
    try:
        meta = ffmpeg.probe(path)
        d = float(meta["format"]["duration"])
        return max(0.0, d)
    except Exception:
        return 0.0

# ---------------------------
# Pulizia & rifinitura locale del testo
# ---------------------------
TS_LINE_RE = re.compile(r'^\s*\[\d{2}:\d{2}(?::\d{2})?\]\s*', flags=re.MULTILINE)

def remove_square_bracket_timestamps(text: str) -> str:
    return TS_LINE_RE.sub('', text)

def dedup_adjacent_words(text: str) -> str:
    return re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

def normalize_spacing_punct(text: str) -> str:
    t = text
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r' +([,;:.!?])', r'\1', t)
    t = re.sub(r'([,;:.!?])(?![\s\)\]\}])', r'\1 ', t)
    t = re.sub(r'([(\[{]) +', r'\1', t)
    t = re.sub(r' +([)\]}])', r'\1', t)
    t = re.sub(r'([.!?]){3,}', '…', t)
    t = re.sub(r'\s*\n\s*', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def split_sentences(text: str):
    t = re.sub(r'\s*\n\s*', ' ', text)
    t = re.sub(r'([.!?…])([^\s])', r'\1 \2', t)
    parts = re.split(r'(?<=[.!?…])\s+', t)
    return [p.strip() for p in parts if p.strip()]

def capitalize_sentences(parts):
    out = []
    for s in parts:
        s = s.strip()
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        out.append(s)
    return out

def wrap_lines(text: str, width: int = 100) -> str:
    wrapped = []
    for para in text.split("\n"):
        wrapped.append(textwrap.fill(para, width=width) if para.strip() else "")
    return "\n".join(wrapped)

def refine_text(
    text: str,
    remove_ts: bool = False,
    dedup_words: bool = True,
    fix_spacing: bool = True,
    capitalize: bool = True,
    line_break_each_sentence: bool = True,
    wrap_width: Optional[int] = 100,
) -> str:
    t = text or ""
    if remove_ts:
        t = remove_square_bracket_timestamps(t)
    if dedup_words:
        t = dedup_adjacent_words(t)
    if fix_spacing:
        t = normalize_spacing_punct(t)

    if line_break_each_sentence:
        parts = split_sentences(t)
        if capitalize:
            parts = capitalize_sentences(parts)
        t = "\n".join(parts)
    elif capitalize:
        parts = split_sentences(t)
        t = " ".join(capitalize_sentences(parts))

    if wrap_width and wrap_width > 20:
        t = wrap_lines(t, width=int(wrap_width))
    return t.strip()

# ---------------------------
# Download/caching modello
# ---------------------------
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "faster_whisper")
os.makedirs(MODEL_CACHE, exist_ok=True)

MODEL_MB = {
    "tiny": 77,
    "base": 142,
    "small": 461,
    "medium": 1500,
    "large-v3": 3100,
}

def _repo_candidates(size: str):
    # Prova prima i repo più recenti, poi fallback storico
    base = size.replace("_", "-")
    return [
        f"Systran/faste
