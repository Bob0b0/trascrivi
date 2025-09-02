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
    """Ritorna la durata in secondi usando ffmpeg.probe; 0 se non disponibile."""
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
    # "la la la prova" -> "la prova"
    return re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

def normalize_spacing_punct(text: str) -> str:
    t = text
    # spazi multipli -> singolo
    t = re.sub(r'[ \t]+', ' ', t)
    # niente spazio prima di ,;:.!? e uno dopo se manca
    t = re.sub(r' +([,;:.!?])', r'\1', t)
    t = re.sub(r'([,;:.!?])(?![\s\)\]\}])', r'\1 ', t)
    # parentesi e punte: niente spazi interni strani
    t = re.sub(r'([(\[{]) +', r'\1', t)
    t = re.sub(r' +([)\]}])', r'\1', t)
    # puntini ripetuti -> ellissi
    t = re.sub(r'([.!?]){3,}', '…', t)
    # normalizza a capo
    t = re.sub(r'\s*\n\s*', '\n', t)
    # riduci righe vuote consecutive
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def split_sentences(text: str):
    # unisci righe spezzate prima di segmentare
    t = re.sub(r'\s*\n\s*', ' ', text)
    # spazio dopo fine frase se manca
    t = re.sub(r'([.!?…])([^\s])', r'\1 \2', t)
    # split conservando i segni di fine frase
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
# UI
# ---------------------------
st.set_page_config(page_title="Trascrizione audio by Roberto M.", page_icon="📝", layout="centered")
st.title("Trascrizione audio by Roberto M.")

st.markdown(
    "Carica un file audio/video, scegli modello e opzioni, quindi avvia la trascrizione. "
    "Durante l’elaborazione vedrai una barra di avanzamento con la **stima del tempo residuo**."
)

# Sidebar — selezioni e guida
st.sidebar.header("⚙️ Impostazioni")

# Scelta modello
MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v3"]
model_size = st.sidebar.selectbox("Modello", MODEL_OPTIONS, index=1, help="Modelli piccoli = più veloci, meno accurati. Modelli grandi = più accurati, più lenti.")

# Precisione (compute_type)
PRECISION_OPTIONS = ["int8", "int8_float32", "int16", "float16", "float32"]
compute_type = st.sidebar.selectbox("Precisione (compute_type)", PRECISION_OPTIONS, index=0)

# Lingua
LANG_DISPLAY = [
    "Auto (rilevamento)",
    "it — Italiano",
    "en — English",
    "es — Español",
    "fr — Français",
    "de — Deutsch",
    "pt — Português",
]
lang_display = st.sidebar.selectbox("Lingua dell'audio", LANG_DISPLAY, index=0)

LANG_MAP = {
    "it — Italiano": "it",
    "en — English": "en",
    "es — Español": "es",
    "fr — Français": "fr",
    "de — Deutsch": "de",
    "pt — Português": "pt",
}
language = None if lang_display.startswith("Auto") else LANG_MAP.get(lang_display, None)

with st.sidebar.expander("ℹ️ Guida rapida (modelli, lingua, precisione)", expanded=False):
    st.markdown(
        """
**Quale modello scegliere**
- **tiny/base** → **molto veloci**, **meno accurati** → perfetti per bozze e file brevi.
- **small** → **buon compromesso** su CPU normali.
- **medium** → **più accurato**, **più lento**. Consigliato con **GPU**.
- **large-v3** → **massima qualità**, più pesante (RAM/VRAM), ideale per audio lunghi o difficili.

**Download vs velocità**
- Al **primo uso** il modello viene **scaricato** (una tantum). Influisce solo sull’**avvio**.
- La **velocità di trascrizione** dipende da **dimensione del modello** e **hardware** (GPU ≫ CPU).

**Consigli pratici**
- Se **conosci la lingua** dell’audio, **selezionala** per una stabilità leggermente migliore; altrimenti lascia **Auto**.
- **Precisione**: lascia **Int8 (default)** su CPU per il miglior rapporto **velocità/qualità**.
        """
    )
st.sidebar.caption("Suggerimento: per file lunghi usa modelli piccoli su CPU, oppure una GPU per i modelli grandi.")

# Opzioni avanzate
with st.expander("Opzioni avanzate", expanded=False):
    word_timestamps = st.checkbox("Parole con timestamp (più lento)", value=False)
    beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5)
    vad_filter = st.checkbox("VAD filter (migliora segmenti di parlato)", value=True)

uploaded = st.file_uploader(
    "Carica un file audio/video",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm"],
    accept_multiple_files=False,
)

start_button = st.button("▶️ Avvia trascrizione", type="primary", disabled=uploaded is None)

# Output placeholders
progress_bar = st.progress(0, text="In attesa del file…")
eta_box = st.empty()
status_box = st.empty()
output_box = st.empty()
metrics_box = st.empty()
error_box = st.empty()

final_text = ""
audio_seconds = 0.0

if start_button:
    if uploaded is None:
        error_box.error("Nessun file selezionato.")
    else:
        # Salva su file temporaneo
        try:
            suffix = os.path.splitext(uploaded.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
        except Exception as e:
            error_box.error(f"Impossibile salvare il file caricato: {e}")
            st.stop()

        # Durata totale del file (per la barra di avanzamento)
        total_duration = safe_probe_duration(tmp_path)

        # Carica il modello
        status_box.info(f"Caricamento modello **{model_size}** (compute_type: **{compute_type}**)…")
        load_t0 = time.time()
        try:
            model = WhisperModel(model_size, device="auto", compute_type=compute_type)
        except Exception as e:
            error_box.error(f"Errore nel caricamento del modello: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            st.stop()
        load_elapsed = time.time() - load_t0

        # Avvio trascrizione

