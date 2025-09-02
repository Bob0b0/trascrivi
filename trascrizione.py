import os
import io
import re
import time
import tempfile
import textwrap
import streamlit as st
from faster_whisper import WhisperModel


# --------------------------- Utilità ---------------------------------

def tidy_text(text: str) -> str:
    """Pulisce e rende più leggibile la trascrizione."""
    # spaziature
    text = re.sub(r"\s+", " ", text).strip()
    # separa in frasi
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    # capitalizza e rifinisce
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    # paragrafi ogni 4–6 frasi circa
    out = []
    chunk, count = [], 0
    for s in sentences:
        chunk.append(s)
        count += 1
        if count >= 5:
            out.append(" ".join(chunk))
            chunk, count = [], 0
    if chunk:
        out.append(" ".join(chunk))
    return "\n\n".join(out)


def ts_srt(seconds: float) -> str:
    """Timestamp SRT 00:00:00,000"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{ts_srt(s.start)} --> {ts_srt(s.end)}")
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines)


def build_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{ts_srt(s.start).replace(',', '.')} --> {ts_srt(s.end).replace(',', '.')}")
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines)


# --------------------------- UI ---------------------------------

st.set_page_config(page_title="Trascrizione audio by Roberto M.", layout="centered")
st.title("Trascrizione audio by Roberto M.")

with st.expander("Opzioni avanzate", expanded=False):
    model_name = st.selectbox(
        "Modello Whisper",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=1,
        help="Scegli 'base' per bilanciare qualità/velocità.",
    )
    lang = st.selectbox(
        "Lingua dell'audio",
        ["auto", "it", "en", "fr", "de", "es", "pt"],
        index=0,
        help="Lascia 'auto' se non sei sicuro.",
    )

uploaded = st.file_uploader(
    "Carica un file audio/video",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
)

with_ts = st.checkbox(
    "Includi timestamp nei file di output (SRT/VTT)",
    value=False,
    help="Disattivato di default.",
)
auto_improve = st.checkbox(
    "Migliora e formatta automaticamente al termine",
    value=True,
    help="Pulizia spazi, capitalizzazione frasi e paragrafazione leggera.",
)

start = st.button("Avvia trascrizione", disabled=uploaded is None)

# --------------------------- Logica ---------------------------------

if start and uploaded:
    # salva file temporaneo
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}")
    tmp_in.write(uploaded.getvalue())
    tmp_in.flush()
    tmp_in.close()

    # carica modello
    t0 = time.time()
    with st.status(f"Caricamento modello {model_name} (compute_type: int8)…", expanded=False) as status:
        try:
            model = WhisperModel(model_name, compute_type="int8")
        except Exception as e:
            status.update(label="Errore nel caricamento del modello", state="error")
            st.error(f"Impossibile caricare il modello: {e}")
            os.unlink(tmp_in.name)
            st.stop()
        else:
            dt = time.time() - t0
            status.update(label=f"Modello {model_name} pronto in {dt:0.2f}s", state="complete")

    # trascrivi
    st.info(f"File caricato: **{uploaded.name}** · Dimensione: {len(uploaded.getvalue())/1_048_576:0.1f}MB")
    st.write("Elaborazione…")

    try:
        segments, info = model.transcribe(
            tmp_in.name,
            language=None if lang == "auto" else lang,
            vad_filter=True,
        )
        seg_list = list(segments)
    except Exception as e:
        st.error(f"Errore durante la trascrizione: {e}")
        os.unlink(tmp_in.name)
        st.stop()

    os.unlink(tmp_in.name)

    # testo puro
    raw_text = " ".join(s.text.strip() for s in seg_list).strip()

    # miglioramento automatico se richiesto
    final_text = tidy_text(raw_text) if auto_improve else raw_text

    # preview
    st.subheader("Testo")
    st.text_area("Trascrizione", value=final_text, height=350)

    # download TXT
    st.download_button(
        "Scarica testo (.txt)",
        data=final_text.encode("utf-8"),
        file_name=os.path.splitext(uploaded.name)[0] + ".txt",
        mime="text/plain",
    )

    # opzionale: SRT/VTT
    if with_ts:
        srt_text = build_srt(seg_list)
        vtt_text = build_vtt(seg_list)
        st.download_button(
            "Scarica sottotitoli SRT",
            data=srt_text.encode("utf-8"),
            file_name=os.path.splitext(uploaded.name)[0] + ".srt",
            mime="application/x-subrip",
        )
        st.download_button(
            "Scarica sottotitoli VTT",
            data=vtt_text.encode("utf-8"),
            file_name=os.path.splitext(uploaded.name)[0] + ".vtt",
            mime="text/vtt",
        )

    st.success("Completato ✅")
