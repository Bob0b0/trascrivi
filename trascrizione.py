# trascrizione.py
from __future__ import annotations

import io
import os
import re
import time
import tempfile
from typing import Iterable, Tuple, Optional

import numpy as np
import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg

st.set_page_config(
    page_title="Trascrizione audio by Roberto M.",
    page_icon="🗣️",
    layout="centered",
)

# ---------- Prompt suggerito (copiabile) ----------
PROMPT_SUGGERITO = """Sei un editor professionale. Ripulisci la seguente trascrizione SENZA cambiare il significato.
Regole:
- Correggi ortografia, punteggiatura e maiuscole.
- Elimina intercalari/esitazioni (es. “ehm”, “cioè”, “diciamo”), riempitivi e ripetizioni accidentali.
- Unisci spezzature dovute alla trascrizione e rimuovi eventuali indicazioni non testuali (rumori, timecode).
- Mantieni citazioni e termini tecnici; non inventare contenuti.
- Organizza il testo in paragrafi brevi e leggibili. Vai a capo quando cambia tema.
- Facoltativo: se emergono sezioni chiarissime, puoi aggiungere titoletti Markdown concisi (non forzare).
- NON riassumere e NON aggiungere commenti.
Restituisci solo il testo finale pulito in italiano.
"""

# ---------- Utility ----------
MODEL_CHOICES = ["tiny", "base", "small", "medium"]  # large disabilitato
DISABLED_MODELS = {"large", "large-v1", "large-v2", "large-v3"}

LANG_CHOICES = {
    "auto": "Rilevamento automatico",
    "it": "Italiano",
    "en": "English",
    "fr": "Français",
    "de": "Deutsch",
    "es": "Español",
    "pt": "Português",
}

def format_ts(seconds: float, for_vtt: bool = False) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    sep = "." if for_vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"

def build_srt(segments: Iterable) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_ts(seg.start, for_vtt=False)
        end = format_ts(seg.end, for_vtt=False)
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def build_vtt(segments: Iterable) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_ts(seg.start, for_vtt=True)
        end = format_ts(seg.end, for_vtt=True)
        text = (seg.text or "").strip()
        if not text:
            continue
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def tidy_text(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?]", t):
        return t[:1].upper() + t[1:]
    parts = re.split(r"([.!?]+)\s*", t)
    sentences = []
    for i in range(0, len(parts), 2):
        fragment = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if not fragment:
            continue
        fragment = fragment[:1].upper() + fragment[1:]
        sentences.append(fragment + punct)
    out_lines = []
    bucket = []
    for s in sentences:
        bucket.append(s)
        if len(bucket) >= 3:
            out_lines.append(" ".join(bucket))
            bucket = []
    if bucket:
        out_lines.append(" ".join(bucket))
    return "\n\n".join(out_lines).strip()

def probe_duration(file_path: str) -> Optional[float]:
    try:
        info = ffmpeg.probe(file_path)
        return float(info["format"]["duration"])
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_model(name: str, compute_type: str = "int8") -> WhisperModel:
    if name in DISABLED_MODELS:
        st.warning("Il modello 'large' è disabilitato. Uso 'medium'.")
        name = "medium"
    return WhisperModel(name, compute_type=compute_type)

def transcribe_file(
    file_path: str,
    model_name: str,
    language: str,
    task: str,
    beam_size: int = 5,
    vad_filter: bool = True,
) -> Tuple[str, list, dict]:
    model = load_model(model_name, compute_type="int8")
    lang = None if language == "auto" else language
    segments_iter, info = model.transcribe(
        file_path,
        task="translate" if task == "translate" else "transcribe",
        language=lang,
        vad_filter=vad_filter,
        beam_size=beam_size,
        temperature=0.0,
        best_of=5,
    )
    segments, texts = [], []
    for seg in segments_iter:
        segments.append(seg)
        if seg.text:
            texts.append(seg.text.strip())
    full_text = " ".join(texts).strip()
    return full_text, segments, {
        "language": info.language,
        "language_probability": info.language_probability,
    }

def bytes_download(data: str, filename: str, mime: str = "text/plain") -> None:
    st.download_button(
        "Scarica " + filename,
        data=data.encode("utf-8"),
        file_name=filename,
        mime=mime,
        use_container_width=True,
    )

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Opzioni")
model_name = st.sidebar.selectbox(
    "Modello Whisper",
    options=MODEL_CHOICES,
    index=MODEL_CHOICES.index("medium"),
    help="Modelli disponibili su questa app. 'large' è disabilitato per limiti di memoria.",
)
language_key = st.sidebar.selectbox(
    "Lingua audio",
    options=list(LANG_CHOICES.keys()),
    format_func=lambda k: LANG_CHOICES[k],
    index=0,
)
task = st.sidebar.radio(
    "Operazione",
    options=["transcribe", "translate"],
    format_func=lambda v: "Trascrivi (stessa lingua)" if v == "transcribe" else "Traduci in Italiano",
    index=0,
)
beam_size = st.sidebar.slider("Beam size", 1, 10, 5)
vad_filter = st.sidebar.toggle("VAD (voice activity detection)", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Se il file NON è in italiano, imposta la lingua corretta o lascia **Rilevamento automatico**.")

# ---------- UI principale ----------
st.title("Trascrizione audio by Roberto M.")

with st.expander("Opzioni avanzate (riassunto)", expanded=False):
    st.write(
        f"**Modello**: `{model_name}` · **Lingua**: `{LANG_CHOICES[language_key]}` · "
        f"**Operazione**: `{'Trascrizione' if task=='transcribe' else 'Traduzione in IT'}` · "
        f"**Beam**: {beam_size} · **VAD**: {'ON' if vad_filter else 'OFF'}"
    )

st.subheader("Carica un file audio/video")
uploaded = st.file_uploader(
    "Drag and drop file here",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
    help="Limite 200MB per file",
)

col_a, col_b = st.columns(2)
with_timestamps = col_a.checkbox(
    "Includi timestamp nei file di output (SRT/VTT)",
    value=False,
    help="Disattivato di default. Abilitalo solo se ti serve SRT/VTT con i timecode.",
)
auto_improve = col_b.checkbox(
    "Migliora e formatta automaticamente al termine",
    value=True,
    help="Pulisce il testo con semplici regole (maiuscole, spazi, a capo).",
)

# Suggerimento rapido vicino ai controlli (come da UI iniziale)
st.caption("Suggerimento: carica un file, lascia i timecode disattivati (default) e attiva ‘Migliora e formatta automaticamente’ per ottenere subito un testo pulito.")

start_btn = st.button("Avvia trascrizione", type="primary", use_container_width=True)

# ---------- Elaborazione ----------
if uploaded and start_btn:
    t0 = time.time()
    status = st.status("Inizio elaborazione…", expanded=True)
    status.write("① Carico il modello…")

    if model_name in DISABLED_MODELS:
        st.warning("Il modello 'large' è disabilitato in questa app. Uso 'medium'.")
        model_name = "medium"

    _ = load_model(model_name)
    status.update(label="① Modello pronto.", state="running")

    status.write("② Preparo il file…")
    suffix = "." + uploaded.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    audio_dur = probe_duration(tmp_path)
    if audio_dur is not None:
        mins = int(audio_dur // 60); secs = int(round(audio_dur % 60))
        status.write(f"Durata audio: **{mins:02d}:{secs:02d}**")
    status.update(label="② File temporaneo creato.", state="running")

    status.write("③ Trascrivo… (potrebbe richiedere qualche minuto)")
    progress = st.progress(0)
    for p in range(0, 70, 5):
        progress.progress(p); time.sleep(0.02)

    try:
        text_raw, segments, info = transcribe_file(
            tmp_path, model_name, language_key, task, beam_size, vad_filter
        )
        progress.progress(90)

        status.write("④ Post-processo il testo…")
        final_text = tidy_text(text_raw) if auto_improve else text_raw

        st.subheader("Risultato")
        st.caption(
            f"Rilevato: **{info.get('language','?')}** "
            f"(p={info.get('language_probability', 0):.2f}) · Modello: `{model_name}`"
        )
        st.text_area("Testo", value=final_text, height=300)

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            bytes_download(final_text, "trascrizione.txt", "text/plain")
        if with_timestamps:
            srt = build_srt(segments); vtt = build_vtt(segments)
            with col2: bytes_download(srt, "trascrizione.srt", "text/srt")
            with col3: bytes_download(vtt, "trascrizione.vtt", "text/vtt")

        # ----- Suggerimento di prompt (sempre disponibile, copiabile) -----
        with st.expander("💡 Suggerimento di prompt (per rifinitura con un LLM)"):
            st.caption("Se vuoi rifinire ulteriormente il testo con un LLM (o se disattivi l'opzione di miglioramento automatico), copia questo prompt:")
            st.code(PROMPT_SUGGERITO, language="markdown")

        progress.progress(100)
        elapsed = time.time() - t0
        if audio_dur is not None:
            st.success(
                f"Completato in **{elapsed:.1f}s** · Durata audio **{audio_dur:.1f}s** "
                f"(~{(elapsed/max(audio_dur,1))*100:.0f}% real-time)."
            )
        else:
            st.success(f"Completato in **{elapsed:.1f}s**.")

        status.update(label="✅ Elaborazione completata.", state="complete")

    except Exception as e:
        status.update(label="Errore durante la trascrizione.", state="error")
        st.error(f"Si è verificato un errore: {e}")
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
