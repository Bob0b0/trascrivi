#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App Streamlit: Trascrizione audio (robusta) con split automatico se la durata supera il limite.
- Nessuna dipendenza extra oltre a: streamlit, faster-whisper (già nel requirements), ffmpeg/ffprobe (da apt).
- Se l'audio > MAX_MINUTES, lo dividiamo in blocchi CHUNK_MINUTES con OVERLAP_SECONDS di sovrapposizione.
- Ogni blocco è convertito in WAV 16 kHz mono e trascritto; i timecode vengono riallineati e uniti.
"""

from __future__ import annotations

import os
import io
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Optional

import streamlit as st

# opzionale: se presente, usiamo faster-whisper
try:
    from faster_whisper import WhisperModel
except Exception as e:  # pragma: no cover
    WhisperModel = None  # type: ignore


# ==========================
# Configurazione di base
# ==========================
APP_TITLE = "Trascrizione audio (robusta)"
MAX_MINUTES = 60                  # limite hard dell'app (prima dello split)
CHUNK_MINUTES_DEFAULT = 30        # durata blocchi generati quando si splitta
OVERLAP_SECONDS_DEFAULT = 2       # sovrapposizione per evitare tagli di parole
TARGET_SR = 16_000                # sample rate per la trascrizione

ALLOWED_EXT = [
    "mp3", "m4a", "aac", "wav", "flac", "ogg", "opus", "wma"
]

# ==========================
# Utility
# ==========================

def _run(cmd: List[str]) -> str:
    """Esegue un comando e ritorna stdout come stringa (solleva eccezione in caso di errore)."""
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def check_ffmpeg() -> Dict[str, str]:
    out = {}
    try:
        out["ffmpeg"] = _run(["ffmpeg", "-version"]).splitlines()[0]
    except Exception:
        out["ffmpeg"] = "NON TROVATO"
    try:
        out["ffprobe"] = _run(["ffprobe", "-version"]).splitlines()[0]
    except Exception:
        out["ffprobe"] = "NON TROVATO"
    return out


def ffprobe_duration_sec(path: str) -> float:
    data = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "format=duration", "-of", "json", path
    ])
    dur = float(json.loads(data)["format"]["duration"])
    return dur


def split_audio(
    in_path: str,
    out_dir: str,
    chunk_minutes: int = CHUNK_MINUTES_DEFAULT,
    overlap_sec: int = OVERLAP_SECONDS_DEFAULT,
    target_sr: int = TARGET_SR,
) -> List[Dict[str, float | str]]:
    """Divide l'audio in parti con una piccola sovrapposizione.
    Ritorna una lista di dict: {path, start, end} per ciascun blocco.
    """
    dur = ffprobe_duration_sec(in_path)
    chunk = chunk_minutes * 60
    parts = []
    start = 0.0
    i = 1
    while start < dur:
        this_len = min(chunk + overlap_sec, dur - start)
        out_path = str(Path(out_dir) / f"part_{i:03d}.wav")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-i", in_path,
            "-t", str(this_len),
            "-ar", str(target_sr), "-ac", "1", "-c:a", "pcm_s16le",
            out_path,
        ]
        subprocess.check_call(cmd)
        parts.append({"path": out_path, "start": start, "end": start + this_len})
        start += chunk  # l'offset globale NON include l'overlap
        i += 1
    return parts


# ==========================
# Trascrizione con faster-whisper
# ==========================

@dataclass
class Seg:
    start: float
    end: float
    text: str


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, compute_type: str):
    if WhisperModel is None:
        raise RuntimeError(
            "La libreria 'faster-whisper' non è disponibile. Aggiungila a requirements.txt"
        )
    # device="auto": sceglie GPU se disponibile, altrimenti CPU
    return WhisperModel(model_name, device="auto", compute_type=compute_type)


def transcribe_file(
    audio_path: str,
    model_name: str = "small",
    compute_type: str = "int8",
    language: str | None = None,
    beam_size: int = 5,
) -> List[Seg]:
    model = load_model(model_name, compute_type)
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )
    out: List[Seg] = []
    for s in segments:  # type: ignore[attr-defined]
        out.append(Seg(start=float(s.start), end=float(s.end), text=s.text.strip()))
    return out


# ==========================
# Formattazioni export
# ==========================

def _fmt_ts_srt(t: float) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: List[Seg]) -> str:
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_fmt_ts_srt(s.start)} --> {_fmt_ts_srt(s.end)}")
        lines.append(s.text)
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments: List[Seg]) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        a = _fmt_ts_srt(s.start).replace(",", ".")
        b = _fmt_ts_srt(s.end).replace(",", ".")
        lines.append(f"{a} --> {b}")
        lines.append(s.text)
        lines.append("")
    return "\n".join(lines)


def segments_to_text(segments: List[Seg]) -> str:
    return " ".join(s.text for s in segments).strip()


# ==========================
# Interfaccia Streamlit
# ==========================

st.set_page_config(page_title=APP_TITLE, page_icon="🎧", layout="centered")

st.title(APP_TITLE)
st.caption("Carica un file audio. L'app esegue i controlli preliminari e solo se passa li trascrive.")

with st.expander("Controlli preliminari", expanded=True):
    checks = check_ffmpeg()
    col1, col2 = st.columns(2)
    col1.write("ffmpeg:")
    col1.code(checks.get("ffmpeg", ""))
    col2.write("ffprobe:")
    col2.code(checks.get("ffprobe", ""))

# Sidebar: opzioni
st.sidebar.header("Opzioni")
model_name = st.sidebar.selectbox("Modello", ["tiny", "base", "small", "medium"], index=2)
compute_type = st.sidebar.selectbox("Precisione (CPU=consigliato int8)", ["int8", "int8_float16", "float16", "float32"], index=0)
lang_opt = st.sidebar.selectbox("Lingua", ["auto (rileva)", "it", "en", "es", "fr", "de"], index=0)
chunk_minutes = st.sidebar.number_input("Minuti per blocco (se split)", 10, 60, CHUNK_MINUTES_DEFAULT, 5)
overlap_sec = st.sidebar.number_input("Sovrapposizione (s)", 0, 10, OVERLAP_SECONDS_DEFAULT, 1)

uploaded = st.file_uploader(
    "Seleziona un file",
    type=ALLOWED_EXT,
    help="Limite 200MB per file • " + ", ".join(ext.upper() for ext in ALLOWED_EXT),
)

if uploaded is not None:
    # Salva il file su disco temporaneo
    tmp_suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
        tmp.write(uploaded.read())
        src_path = tmp.name

    # Metadati e durata
    try:
        duration_sec = ffprobe_duration_sec(src_path)
    except Exception as e:
        st.error(f"Impossibile leggere i metadata: {e}")
        st.stop()

    duration_min = duration_sec / 60.0

    st.info(f"Durata: **{duration_min:.1f} min** · Dimensione: **{uploaded.size/1_000_000:.1f} MB**")

    # Se supera il limite, proponi split automatico
    if duration_min > MAX_MINUTES:
        st.error(f"Impossibile processare il file: Audio troppo lungo ({duration_min:.0f} min). Limite: {MAX_MINUTES} min.")
        auto = st.checkbox(
            f"Dividi automaticamente in blocchi da {chunk_minutes} min e procedi",
            value=True,
        )
        if auto and st.button("Avvia frazionamento e trascrizione"):
            with st.status("Divisione e trascrizione in corso...", expanded=True) as status:
                with tempfile.TemporaryDirectory() as td:
                    parts = split_audio(src_path, td, int(chunk_minutes), int(overlap_sec), TARGET_SR)
                    st.write(f"Creati {len(parts)} blocchi")

                    all_segments: List[Seg] = []
                    for idx, p in enumerate(parts, start=1):
                        st.write(f"• Trascrivo parte {idx}/{len(parts)}")
                        segs = transcribe_file(
                            p["path"],
                            model_name=model_name,
                            compute_type=compute_type,
                            language=None if lang_opt.startswith("auto") else lang_opt,
                        )
                        # riallinea i timecode: offset globale = (indice-1) * chunk_minutes
                        offset = (idx - 1) * int(chunk_minutes) * 60
                        for s in segs:
                            all_segments.append(Seg(start=s.start + offset, end=s.end + offset, text=s.text))

                status.update(label="Merge completato", state="complete")

            # Esporta
            base = Path(uploaded.name).with_suffix("").name
            srt = segments_to_srt(all_segments)
            vtt = segments_to_vtt(all_segments)
            txt = segments_to_text(all_segments)

            st.subheader("Download")
            st.download_button("Scarica .txt", txt, file_name=f"{base}.txt")
            st.download_button("Scarica .srt", srt, file_name=f"{base}.srt")
            st.download_button("Scarica .vtt", vtt, file_name=f"{base}.vtt")

            st.subheader("Anteprima testo")
            st.text_area("", txt, height=240)

    else:
        # Flusso normale (nessuno split)
        if st.button("Trascrivi"):
            with st.status("Trascrizione in corso...", expanded=True) as status:
                # Converte direttamente in wav 16k mono temporaneo
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpwav:
                    wav_path = tmpwav.name
                subprocess.check_call([
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", src_path,
                    "-ar", str(TARGET_SR), "-ac", "1", "-c:a", "pcm_s16le",
                    wav_path,
                ])
                segs = transcribe_file(
                    wav_path,
                    model_name=model_name,
                    compute_type=compute_type,
                    language=None if lang_opt.startswith("auto") else lang_opt,
                )
                status.update(label="Trascrizione completata", state="complete")

            base = Path(uploaded.name).with_suffix("").name
            srt = segments_to_srt(segs)
            vtt = segments_to_vtt(segs)
            txt = segments_to_text(segs)

            st.subheader("Download")
            st.download_button("Scarica .txt", txt, file_name=f"{base}.txt")
            st.download_button("Scarica .srt", srt, file_name=f"{base}.srt")
            st.download_button("Scarica .vtt", vtt, file_name=f"{base}.vtt")

            st.subheader("Anteprima testo")
            st.text_area("", txt, height=240)

# Footer minimale
st.markdown(
    """
    <hr/>
    <small>Se l'audio supera i 60 minuti, l'app esegue automaticamente lo split in blocchi (di default 30') con 2s di sovrapposizione e unisce i risultati mantenendo i timecode continui.</small>
    """,
    unsafe_allow_html=True,
)
