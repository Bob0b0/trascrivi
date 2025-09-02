import os
import io
import re
import json
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import streamlit as st
from faster_whisper import WhisperModel

# --------------------------- Config pagina --------------------------- #
st.set_page_config(
    page_title="Trascrizione audio by Roberto M.",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Utilità --------------------------- #
def format_timestamp(secs: float, srt: bool = True) -> str:
    """Converte secondi → HH:MM:SS,mmm (SRT) o HH:MM:SS.mmm (VTT)."""
    if secs < 0:
        secs = 0
    millis = int(round(secs * 1000))
    h = millis // 3_600_000
    m = (millis % 3_600_000) // 60_000
    s = (millis % 60_000) // 1000
    ms = millis % 1000
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    else:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def build_srt(segments: List[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg["start"], srt=True)
        end = format_timestamp(seg["end"], srt=True)
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"

def build_vtt(segments: List[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp(seg["start"], srt=False)
        end = format_timestamp(seg["end"], srt=False)
        text = seg["text"].strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines).strip() + "\n"

_caps_re = re.compile(r"([.!?])\s+([a-zàèéìòóù])")

def tidy_text(text: str) -> str:
    """Pulizia leggera + formattazione frasi."""
    if not text:
        return ""
    t = text

    # Spaziature e segni
    t = re.sub(r"\s+", " ", t)                         # spazi multipli
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)             # spazio prima di punteggiatura
    t = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", t)      # spazio dopo punteggiatura
    t = t.strip()

    # Maiuscole dopo punto
    def _cap(match):
        return f"{match.group(1)} {match.group(2).upper()}"
    t = _caps_re.sub(_cap, t)

    # Primo carattere maiuscolo
    if t:
        t = t[0].upper() + t[1:]

    return t.strip()

@dataclass
class Options:
    model_name: str = "base"
    compute_type: str = "int8"
    language: Optional[str] = None       # None = auto
    vad_filter: bool = True
    beam_size: int = 5
    word_timestamps: bool = False
    include_timestamps: bool = False     # SRT/VTT
    auto_improve: bool = True

# --------------------------- Cache modello --------------------------- #
@st.cache_resource(show_spinner=False)
def load_model_cached(model_name: str, compute_type: str) -> WhisperModel:
    # Nota: su Streamlit Cloud conviene 'int8' su CPU
    return WhisperModel(model_name, compute_type=compute_type)

# --------------------------- Sidebar --------------------------- #
st.sidebar.header("⚙️ Opzioni")

model_name = st.sidebar.selectbox(
    "Modello Whisper",
    ["tiny", "base", "small", "medium", "large-v3"],
    index=1,
    help="Modelli più grandi = più lenti ma più accurati. "
         "Su ambienti con poca RAM evita 'large-v3'.",
)

compute_type = st.sidebar.selectbox(
    "Compute type",
    ["int8", "int8_float16", "int8_float32", "float16", "float32"],
    index=0,
    help="Int8 è il migliore compromesso su CPU."
)

language_choice = st.sidebar.selectbox(
    "Lingua (lascia 'auto' se non sei sicuro)",
    ["auto", "it", "en", "fr", "de", "es", "pt", "ro", "pl", "nl", "sv", "ru", "uk", "tr"],
    index=0,
)
language = None if language_choice == "auto" else language_choice

vad_filter = st.sidebar.toggle(
    "VAD (voice activity detection)",
    value=True,
    help="Filtra i segmenti silenziosi per una segmentazione più pulita."
)

beam_size = st.sidebar.slider(
    "Beam size",
    min_value=1, max_value=10, value=5, step=1,
    help="Valori più alti = un pizzico più accurato ma più lento."
)

st.sidebar.divider()

include_timestamps = st.sidebar.toggle(
    "Includi timestamp nei file di output (SRT/VTT)",
    value=False,
    help="Disattivato di default. Se attivato, avrai i sottotitoli .srt/.vtt."
)

auto_improve = st.sidebar.toggle(
    "Migliora e formatta automaticamente",
    value=True,
    help="Applica una pulizia leggera: spazi, punteggiatura, maiuscole."
)

st.sidebar.divider()
st.sidebar.subheader("📌 Legenda & Suggerimenti")
st.sidebar.caption(
    "- **Modello**: scegli in base all'hardware; *base/small* vanno bene per la maggior parte dei casi.\n"
    "- **Compute type**: su CPU usa **int8**.\n"
    "- **Lingua**: lascia *auto* se non sei sicuro.\n"
    "- **VAD**: migliora la segmentazione eliminando silenzi.\n"
    "- **Beam size**: 4–6 è un buon compromesso.\n"
    "- **Timestamp**: attivalo solo se vuoi SRT/VTT.\n"
    "- **Migliora**: sistema il testo senza dover copiare/incollare."
)

# --------------------------- Header principale --------------------------- #
st.title("Trascrizione audio by Roberto M.")

with st.expander("Opzioni avanzate (riassunto)", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Modello", model_name)
    col2.metric("Compute", compute_type)
    col3.metric("Beam", beam_size)
    st.caption("Queste impostazioni si modificano nella **sidebar**.")

# --------------------------- Upload --------------------------- #
uploaded = st.file_uploader(
    "Carica un file audio/video",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm"],
    help="Limite 200MB per file. Formati comuni supportati da ffmpeg."
)

start_btn = st.button("Avvia trascrizione", type="primary", disabled=uploaded is None)

# --------------------------- Esecuzione --------------------------- #
if start_btn and uploaded is not None:
    opts = Options(
        model_name=model_name,
        compute_type=compute_type,
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        include_timestamps=include_timestamps,
        auto_improve=auto_improve,
    )

    try:
        with st.status("Inizio elaborazione…", expanded=True) as status:
            st.write("① Carico il modello…")
            model = load_model_cached(opts.model_name, opts.compute_type)
            st.write("✅ Modello pronto.")

            # Salva il file su disco temporaneo
            st.write("② Preparo il file…")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            st.write(f"✅ File temporaneo creato.")

            st.write("③ Trascrivo… potrebbe richiedere un po’ di tempo.")
            segments_iter, info = model.transcribe(
                tmp_path,
                language=opts.language,
                beam_size=opts.beam_size,
                vad_filter=opts.vad_filter,
                word_timestamps=False,  # più leggero
            )

            segments: List[dict] = []
            raw_text_parts: List[str] = []

            for seg in segments_iter:
                d = {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                segments.append(d)
                raw_text_parts.append(seg.text)

            raw_text = " ".join(part.strip() for part in raw_text_parts).strip()

            st.write("④ Pulizia e formattazione…")
            final_text = tidy_text(raw_text) if opts.auto_improve else raw_text

            # Prepara output
            downloads = []

            # TXT pulito
            txt_bytes = final_text.encode("utf-8")
            downloads.append(("Scarica testo (.txt)", f"{os.path.splitext(uploaded.name)[0]}_pulito.txt", txt_bytes, "text/plain"))

            # Sottotitoli opzionali
            if opts.include_timestamps and segments:
                srt_str = build_srt(segments)
                vtt_str = build_vtt(segments)
                downloads.append(("Scarica sottotitoli SRT", f"{os.path.splitext(uploaded.name)[0]}.srt", srt_str.encode("utf-8"), "text/plain"))
                downloads.append(("Scarica sottotitoli VTT", f"{os.path.splitext(uploaded.name)[0]}.vtt", vtt_str.encode("utf-8"), "text/vtt"))

            # Segmenti JSON
            seg_json = json.dumps({"segments": segments}, ensure_ascii=False, indent=2).encode("utf-8")
            downloads.append(("Scarica segmenti JSON", f"{os.path.splitext(uploaded.name)[0]}_segments.json", seg_json, "application/json"))

            st.write("⑤ Pronto! Genero i pulsanti di download…")
            status.update(label="Elaborazione completata ✅", state="complete")

        # ------------------- Output UI ------------------- #
        st.subheader("Anteprima testo")
        st.caption("Questa è la versione **pulita** (se attivata l’opzione).")
        st.text_area("Testo", value=final_text, height=260)

        st.subheader("Download")
        cols = st.columns(min(3, len(downloads)))
        for i, (label, fname, data, mime) in enumerate(downloads):
            with cols[i % len(cols)]:
                st.download_button(label, data=data, file_name=fname, mime=mime, use_container_width=True)

        with st.expander("Dettaglio segmenti"):
            st.dataframe(
                [{"start": format_timestamp(s["start"], True), "end": format_timestamp(s["end"], True), "testo": s["text"].strip()} for s in segments],
                use_container_width=True,
            )

    except Exception as e:
        st.error("Qualcosa è andato storto durante l’elaborazione.")
        st.exception(e)
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
else:
    st.caption("Suggerimento: carica un file, lascia i timecode **disattivati** (default) e attiva **Migliora e formatta automaticamente** per ottenere subito un testo pulito.")
