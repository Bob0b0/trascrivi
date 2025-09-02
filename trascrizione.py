import os
import io
import re
import json
import time
import tempfile
from dataclasses import dataclass
from typing import List, Optional

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
    if secs < 0:
        secs = 0
    millis = int(round(secs * 1000))
    h = millis // 3_600_000
    m = (millis % 3_600_000) // 60_000
    s = (millis % 60_000) // 1000
    ms = millis % 1000
    return (f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            if srt else f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}")

def build_srt(segments: List[dict]) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(
            f"{i}\n"
            f"{format_timestamp(seg['start'], True)} --> {format_timestamp(seg['end'], True)}\n"
            f"{seg['text'].strip()}\n"
        )
    return "\n".join(out).strip() + "\n"

def build_vtt(segments: List[dict]) -> str:
    out = ["WEBVTT", ""]
    for seg in segments:
        out.append(
            f"{format_timestamp(seg['start'], False)} --> {format_timestamp(seg['end'], False)}\n"
            f"{seg['text'].strip()}\n"
        )
    return "\n".join(out).strip() + "\n"

_caps_re = re.compile(r"([.!?])\s+([a-zàèéìòóù])")
def tidy_text(text: str) -> str:
    if not text:
        return ""
    t = text
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", t)
    t = t.strip()
    def _cap(m): return f"{m.group(1)} {m.group(2).upper()}"
    t = _caps_re.sub(_cap, t)
    if t:
        t = t[0].upper() + t[1:]
    return t.strip()

def human_time(secs: float | int | None) -> str:
    if not secs or secs < 0:
        return "00:00"
    secs = int(round(secs))
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

@dataclass
class Options:
    model_name: str = "base"
    compute_type: str = "int8"
    language: Optional[str] = None
    vad_filter: bool = True
    beam_size: int = 5
    include_timestamps: bool = False
    auto_improve: bool = True

# --------------------------- Cache modello --------------------------- #
@st.cache_resource(show_spinner=False)
def load_model_cached(model_name: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, compute_type=compute_type)

# --------------------------- Sidebar --------------------------- #
st.sidebar.header("⚙️ Opzioni")

model_name = st.sidebar.selectbox(
    "Modello Whisper",
    ["tiny", "base", "small", "medium", "large-v3"],
    index=1,
)

compute_type = st.sidebar.selectbox(
    "Compute type",
    ["int8", "int8_float16", "int8_float32", "float16", "float32"],
    index=0,
)

language_choice = st.sidebar.selectbox(
    "Lingua (lascia 'auto' se non sei sicuro)",
    ["auto", "it", "en", "fr", "de", "es", "pt", "ro", "pl", "nl", "sv", "ru", "uk", "tr"],
    index=0,
)
language = None if language_choice == "auto" else language_choice

vad_filter = st.sidebar.toggle("VAD (voice activity detection)", True)
beam_size = st.sidebar.slider("Beam size", 1, 10, 5, 1)

st.sidebar.divider()
include_timestamps = st.sidebar.toggle(
    "Includi timestamp nei file di output (SRT/VTT)", value=False
)
auto_improve = st.sidebar.toggle("Migliora e formatta automaticamente", value=True)

st.sidebar.divider()
st.sidebar.subheader("📌 Legenda & Suggerimenti")
st.sidebar.caption(
    "- **int8** consigliato su CPU\n"
    "- **VAD** migliora la segmentazione\n"
    "- **Timestamp** off di default: attivalo solo se vuoi SRT/VTT"
)

# --------------------------- Header --------------------------- #
st.title("Trascrizione audio by Roberto M.")

with st.expander("Opzioni avanzate (riassunto)", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Modello", model_name)
    col2.metric("Compute", compute_type)
    col3.metric("Beam", beam_size)
    st.caption("Modifica dalla **sidebar**.")

# --------------------------- Upload --------------------------- #
uploaded = st.file_uploader(
    "Carica un file audio/video",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm"],
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

    t_all_start = time.perf_counter()
    try:
        with st.status("Inizio elaborazione…", expanded=True) as status:
            st.write("① Carico il modello…")
            model = load_model_cached(opts.model_name, opts.compute_type)
            st.write("✅ Modello pronto.")

            st.write("② Preparo il file…")
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            st.write("✅ File temporaneo creato.")

            st.write("③ Trascrivo…")
            t_tx_start = time.perf_counter()

            # Barra di avanzamento
            prog_text = st.empty()
            prog_bar = st.progress(0.0)

            segments_iter, info = model.transcribe(
                tmp_path,
                language=opts.language,
                beam_size=opts.beam_size,
                vad_filter=opts.vad_filter,
                word_timestamps=False,
            )

            audio_duration = getattr(info, "duration", None)  # in secondi
            if audio_duration:
                st.info(f"Durata audio: **{human_time(audio_duration)}**")

            segments: List[dict] = []
            raw_parts: List[str] = []
            last_t = 0.0

            for seg in segments_iter:
                d = {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text}
                segments.append(d)
                raw_parts.append(seg.text)

                # Avanzamento
                last_t = max(last_t, seg.end or 0.0)
                if audio_duration and audio_duration > 0:
                    ratio = min(last_t / audio_duration, 1.0)
                    prog_bar.progress(ratio)
                    prog_text.markdown(
                        f"**Avanzamento**: {human_time(last_t)} / {human_time(audio_duration)}"
                    )

            prog_bar.progress(1.0)
            prog_text.markdown(
                f"**Trascrizione completata** · audio {human_time(audio_duration)}"
                if audio_duration else "**Trascrizione completata**"
            )

            raw_text = " ".join(p.strip() for p in raw_parts).strip()

            t_tx_end = time.perf_counter()
            t_tidy_start = time.perf_counter()
            st.write("④ Pulizia e formattazione…")
            final_text = tidy_text(raw_text) if opts.auto_improve else raw_text
            t_tidy_end = time.perf_counter()

            # Output
            st.write("⑤ Preparo i download…")
            downloads = []

            txt_bytes = final_text.encode("utf-8")
            base = os.path.splitext(uploaded.name)[0]
            downloads.append(("Scarica testo (.txt)", f"{base}_pulito.txt", txt_bytes, "text/plain"))

            if opts.include_timestamps and segments:
                srt_str = build_srt(segments)
                vtt_str = build_vtt(segments)
                downloads.append(("Scarica sottotitoli SRT", f"{base}.srt", srt_str.encode("utf-8"), "text/plain"))
                downloads.append(("Scarica sottotitoli VTT", f"{base}.vtt", vtt_str.encode("utf-8"), "text/vtt"))

            seg_json = json.dumps({"segments": segments}, ensure_ascii=False, indent=2).encode("utf-8")
            downloads.append(("Scarica segmenti JSON", f"{base}_segments.json", seg_json, "application/json"))

            status.update(label="Elaborazione completata ✅", state="complete")

        # ------------------- Riepilogo tempi ------------------- #
        t_all_end = time.perf_counter()
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Durata audio", human_time(audio_duration))
        colB.metric("Trascrizione", human_time(t_tx_end - t_tx_start))
        colC.metric("Pulizia/formatting", human_time(t_tidy_end - t_tidy_start))
        colD.metric("Tempo totale", human_time(t_all_end - t_all_start))

        # ------------------- Output UI ------------------- #
        st.subheader("Anteprima testo")
        st.caption("Versione **pulita** (se attivata l’opzione).")
        st.text_area("Testo", value=final_text, height=260)

        st.subheader("Download")
        cols = st.columns(min(3, len(downloads)))
        for i, (label, fname, data, mime) in enumerate(downloads):
            with cols[i % len(cols)]:
                st.download_button(label, data=data, file_name=fname, mime=mime, use_container_width=True)

        with st.expander("Dettaglio segmenti"):
            st.dataframe(
                [{"start": format_timestamp(s["start"]), "end": format_timestamp(s["end"]), "testo": s["text"].strip()} for s in segments],
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
    st.caption("Suggerimento: carica un file, lascia i timecode **disattivati** e attiva **Migliora e formatta automaticamente** per ottenere subito un testo pulito.")
