import io
import os
import time
import math
import re
from datetime import timedelta

import streamlit as st
import ffmpeg
from faster_whisper import WhisperModel


# ---------- Utils ----------
def seconds_to_mmss(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s//60:02d}:{s%60:02d}"

def tidy_text(text: str) -> str:
    """Pulizia leggera + formattazione a capoversi."""
    if not text:
        return ""

    # spazi, punteggiatura, doppie spaziature
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\s+([,;:.!?])", r"\1", t)
    t = re.sub(r"([,;])([^\s])", r"\1 \2", t)

    # parole ripetute immediate (es. "che che", "eh eh")
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.IGNORECASE)

    # spezza in frasi e metti a capo
    sentences = re.split(r"(?<=[.!?…])\s+", t.strip())
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    t = "\n\n".join(sentences)

    # righe non troppo lunghe (soft wrap mentale; Streamlit fa già il wrap)
    return t.strip()

def probe_duration(file_path: str) -> float:
    """Durata in secondi (usa ffmpeg)."""
    info = ffmpeg.probe(file_path)
    return float(info["format"]["duration"])

def human_size(num_bytes: int) -> str:
    for unit in ["B","KB","MB","GB"]:
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f}{unit}" if unit != "B" else f"{num_bytes}B"
        num_bytes /= 1024.0


# ---------- UI ----------
st.set_page_config(page_title="Trascrizione audio by Roberto M.", page_icon="📝", layout="wide")

st.title("Trascrizione audio by Roberto M.")
st.write("Carica un file audio/video, scegli modello e opzioni, quindi avvia la trascrizione. "
         "Durante l’elaborazione vedrai una barra di avanzamento con la **stima del tempo residuo**.")

with st.sidebar:
    st.header("Opzioni")
    model_size = st.selectbox(
        "Modello",
        ["tiny", "base", "small"],
        index=1,
        help="Modelli più piccoli si caricano più in fretta ma sono meno accurati. "
             "‘base’ è un buon compromesso su CPU."
    )
    compute_type = st.selectbox(
        "Precisione (compute_type)",
        ["int8", "int8_float16", "float16", "float32"],
        index=0,
        help="Lascia **int8** (default) per massime prestazioni su CPU."
    )
    lang_map = {
        "Auto (rileva)": None,
        "Italiano (it)": "it",
        "English (en)": "en",
        "Español (es)": "es",
        "Français (fr)": "fr",
        "Deutsch (de)": "de",
    }
    language_label = st.selectbox("Lingua", list(lang_map.keys()), index=0)
    language = lang_map[language_label]

    st.markdown("---")
    st.subheader("Suggerimenti")
    st.markdown(
        "- Scegli **Base** o **Tiny** se sei su CPU.\n"
        "- Se conosci la lingua, **selezionala** per partire più rapidamente.\n"
        "- Lascia la precisione su **Int8** per velocità ottimale."
    )

with st.expander("Opzioni avanzate", expanded=False):
    show_timestamps = st.checkbox("Mostra timecode (disattivato di default)", value=False)
    auto_tidy = st.checkbox("Migliora e formatta automaticamente", value=True)
    beam_size = st.slider("Beam size", 1, 5, 1, help="1 è più veloce, >1 leggermente più accurato.")
    vad_filter = st.checkbox("VAD (filtra silenzi/rumori)", value=True)

st.subheader("Carica un file audio/video")
uploaded = st.file_uploader(
    "Drag and drop file here",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
    accept_multiple_files=False
)

start_btn = st.button("🚀 Avvia trascrizione", disabled=(uploaded is None))

hint = st.caption("Suggerimento: carica un file, lascia i timecode disattivati (default) e attiva "
                  "**'Migliora e formattata automaticamente'** per ottenere subito un testo pulito.")

# ---------- Main logic ----------
if start_btn and uploaded is not None:
    # salva file temporaneo
    tmp_dir = "tmp_inputs"
    os.makedirs(tmp_dir, exist_ok=True)
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # durata e dimensione
    try:
        total_duration = probe_duration(input_path)
    except Exception as e:
        st.error(f"Impossibile leggere la durata del file: {e}")
        st.stop()

    st.info(f"Caricato: **{uploaded.name}** · {human_size(len(uploaded.getbuffer()))} · "
            f"Durata: **{seconds_to_mmss(total_duration)}**")

    # carica modello
    with st.status(f"Caricamento modello **{model_size}** (compute_type: **{compute_type}**)...", expanded=True) as s:
        try:
            t0_load = time.time()
            model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
            load_time = time.time() - t0_load
            s.update(label=f"Modello **{model_size}** pronto in {seconds_to_mmss(load_time)}.", state="complete")
        except Exception as e:
            st.error(f"Errore nel caricamento del modello: {e}")
            st.stop()

    # trascrizione streaming con ETA
    st.subheader("Elaborazione")
    progress = st.progress(0.0, text="Inizializzazione…")
    eta_box = st.empty()

    collected_segments = []
    plain_text_parts = []

    t0 = time.time()
    processed_sec = 0.0

    try:
        segments, info = model.transcribe(
            input_path,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        lang_detected = info.language if hasattr(info, "language") else language or "auto"

        for seg in segments:
            collected_segments.append(seg)
            plain_text_parts.append(seg.text)

            processed_sec = float(seg.end)
            frac = min(1.0, processed_sec / total_duration) if total_duration > 0 else 0.0
            elapsed = time.time() - t0
            remaining = (elapsed / max(frac, 1e-6)) - elapsed if frac > 0 else 0.0

            progress.progress(frac, text=f"Avanzamento: {int(frac*100)}%")
            eta_box.write(f"⏳ Stima tempo residuo: **{seconds_to_mmss(remaining)}**")

        total_time = time.time() - t0

    except Exception as e:
        st.error(f"Errore durante la trascrizione: {e}")
        st.stop()

    progress.progress(1.0, text="Completato")
    eta_box.empty()

    # compone testo
    raw_text = " ".join(plain_text_parts).strip()

    if show_timestamps:
        st.write("#### Trascrizione con timecode")
        ts_lines = []
        for s in collected_segments:
            ts_lines.append(f"[{seconds_to_mmss(s.start)} → {seconds_to_mmss(s.end)}] {s.text.strip()}")
        st.text_area("Output (con timecode)", "\n".join(ts_lines), height=240)

    # miglioramento automatico
    final_text = tidy_
