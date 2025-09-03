# trascrizione.py
# UI Streamlit per trascrivere file audio/video con faster-whisper
# Sidebar con Modello/Lingua/CPU – timestamp OFF di default, miglioramento automatico

import os
import io
import time
import math
import tempfile
from datetime import timedelta
from typing import List, Tuple

import streamlit as st
from faster_whisper import WhisperModel

try:
    import ffmpeg  # per leggere durata con ffprobe
except Exception:
    ffmpeg = None


# ------------------------- CONFIG -------------------------

st.set_page_config(
    page_title="Trascrizione audio by Roberto M.",
    page_icon="📝",
    layout="wide",
)

st.title("Trascrizione audio by Roberto M.")


# ------------------------- UTILS -------------------------

def human_time(seconds: float) -> str:
    seconds = max(0, float(seconds))
    return str(timedelta(seconds=int(seconds)))


def probe_duration(file_path: str) -> float:
    """Durata file (sec) con ffprobe; fallback 0."""
    if ffmpeg is None:
        return 0.0
    try:
        meta = ffmpeg.probe(file_path)
        dur = meta.get("format", {}).get("duration", None)
        if dur is not None:
            return float(dur)
        for s in meta.get("streams", []):
            if "duration" in s and s["duration"] is not None:
                return float(s["duration"])
    except Exception:
        pass
    return 0.0


def tidy_text(text: str) -> str:
    """Pulizia leggera: spazi, righe spezzate, maiuscole all'inizio paragrafo."""
    if not text:
        return ""

    import re

    t = text.replace(" \n", "\n").replace("\n ", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)

    lines = [ln.strip() for ln in t.splitlines()]
    paras: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if ln:
            buf.append(ln)
        else:
            if buf:
                paras.append(" ".join(buf))
                buf = []
            paras.append("")
    if buf:
        paras.append(" ".join(buf))
    t = "\n\n".join(paras)

    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)

    def cap_first(s: str) -> str:
        for i, ch in enumerate(s):
            if ch.isalpha():
                return s[:i] + ch.upper() + s[i + 1 :]
        return s

    t = "\n\n".join(cap_first(p) if p else "" for p in t.split("\n\n"))
    return t.strip()


def improve_text(text: str, preset: str, custom_rules: str) -> str:
    """Migliora/formatta senza LLM: cosmetica soft."""
    t = tidy_text(text)
    if preset == "Pulito neutro":
        return t

    import re

    if preset == "Paragrafi leggibili":
        t = re.sub(r"([.!?])\s+(?=[A-ZÀ-ÖØ-Þ])", r"\1\n\n", t)
        return tidy_text(t)

    if preset == "Appunti sintetici":
        sentences = re.split(r"(?<=[.!?])\s+", t)
        bullets = []
        for s in sentences:
            s = s.strip("-• ").strip()
            if not s:
                continue
            bullets.append(f"• {s}" if len(s) <= 180 else s)
        return "\n".join(bullets).strip()

    if custom_rules.strip():
        t = re.sub(r"\s{3,}", "  ", t)
    return t


def to_srt(segments: List[Tuple[float, float, str]]) -> str:
    def ts(x: float) -> str:
        ms = int((x - int(x)) * 1000)
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, (start, end, txt) in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(txt.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_vtt(segments: List[Tuple[float, float, str]]) -> str:
    def ts(x: float) -> str:
        ms = int((x - int(x)) * 1000)
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for (start, end, txt) in segments:
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(txt.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, compute_type: str) -> WhisperModel:
    # device=cpu per Streamlit Cloud; compute_type selezionabile in sidebar
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


# ------------------------- SIDEBAR: IMPOSTAZIONI -------------------------

with st.sidebar:
    st.header("Impostazioni")

    model_size = st.selectbox(
        "Modello Whisper",
        options=["tiny", "base", "small", "medium"],  # large disabilitato
        index=1,
        help="Modelli più grandi = migliore qualità ma richiedono più tempo/RAM.",
    )

    compute_choice = st.selectbox(
        "Calcolo (CPU)",
        options=["int8 (consigliato)", "float32 (più qualità, più RAM)"],
        index=0,
        help="Se hai limiti di memoria su Streamlit Cloud, lascia int8.",
    )
    compute_type = "int8" if compute_choice.startswith("int8") else "float32"

    language = st.selectbox(
        "Lingua dell'audio",
        options=["auto", "it", "en", "fr", "de", "es", "pt", "ro"],
        index=0,
        help="Se non sei sicuro, lascia **auto**.",
    )

    initial_prompt = st.text_area(
        "Prompt iniziale (opzionale)",
        value="",
        height=90,
        placeholder="Esempio: L'audio è in italiano; nomi propri: Paolo Ricca; usa punteggiatura naturale.",
    )

    with st.expander("Legenda & note"):
        st.markdown(
            """
- **Large disabilitato** per limiti di RAM.
- **int8** = più leggero/veloce; **float32** = più qualità (usa più memoria).
- Il **Prompt iniziale** aiuta con nomi propri/termini tecnici.
"""
        )


# ------------------------- OPZIONI PRINCIPALI -------------------------

with_ts = st.checkbox(
    "Includi timestamp nei file di output (SRT/VTT)",
    value=False,
    help="Disattivato di default. Abilitalo solo se ti servono i sottotitoli.",
)

auto_improve = st.checkbox(
    "Migliora e formatta automaticamente al termine",
    value=True,
    help="Applica una pulizia testuale leggera (spazi, frasi spezzate, paragrafi).",
)

if auto_improve:
    colI, colJ = st.columns([1, 1])
    with colI:
        preset = st.selectbox(
            "Stile di miglioramento",
            options=["Pulito neutro", "Paragrafi leggibili", "Appunti sintetici"],
            index=0,
        )
    with colJ:
        custom_rules = st.text_input(
            "Regole aggiuntive (facoltative)",
            value="",
            help="Brevi note personali (es. 'mantieni elenchi puntati se presenti').",
        )
else:
    preset = "Pulito neutro"
    custom_rules = ""


# ------------------------- UPLOAD -------------------------

st.subheader("Carica un file audio/video")
uploaded = st.file_uploader(
    "Drag and drop file here",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

start_btn = st.button("Avvia trascrizione", type="primary", disabled=(uploaded is None))
status_box = st.container()


# ------------------------- ELABORAZIONE -------------------------

if start_btn and uploaded is not None:
    t0 = time.time()

    with status_box:
        st.write("### Inizio elaborazione…")
        status = st.status("Preparazione…", expanded=True)

        # 1) File temporaneo
        status.update(label="Preparazione file…", state="running")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        duration = probe_duration(tmp_path)
        if duration > 0:
            st.info(f"**File:** {uploaded.name} — **{file_size_mb:.1f} MB** — **Durata:** {human_time(duration)}")
        else:
            st.info(f"**File:** {uploaded.name} — **{file_size_mb:.1f} MB** — **Durata:** sconosciuta")

        # 2) Modello
        status.update(label=f"Carico il modello `{model_size}` (CPU {compute_type})…", state="running")
        model = load_model(model_size, compute_type=compute_type)
        status.write("Modello pronto.")

        # 3) Trascrizione + progress
        status.update(label="Trascrivo…", state="running")
        progress = st.progress(0)
        seg_list: List[Tuple[float, float, str]] = []

        transcribe_kwargs = dict(
            language=None if language == "auto" else language,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            initial_prompt=initial_prompt or None,
        )

        total = max(duration, 1.0)
        try:
            segments, info = model.transcribe(tmp_path, **transcribe_kwargs)
            for seg in segments:
                seg_list.append((seg.start, seg.end, seg.text.strip()))
                p = min(1.0, float(seg.end) / total) if total > 0 else 0.0
                progress.progress(int(p * 100))
            status.write(f"Lingua rilevata: **{info.language}** — confidenza **{info.language_probability:.2f}**")
        except Exception as e:
            status.update(label="Errore durante la trascrizione.", state="error")
            st.error(f"Si è verificato un errore: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            st.stop()

        raw_text = "\n".join(t for _, _, t in seg_list).strip()

        # 4) Rifinitura
        status.update(label="Rifinitura testo…", state="running")
        final_text = improve_text(raw_text, preset=preset, custom_rules=custom_rules) if auto_improve else raw_text

        # 5) Export
        txt_bytes = final_text.encode("utf-8")
        srt_bytes = to_srt(seg_list).encode("utf-8") if with_ts else None
        vtt_bytes = to_vtt(seg_list).encode("utf-8") if with_ts else None

        elapsed = time.time() - t0
        rtf = (elapsed / duration) if duration > 0 else float("nan")
        status.update(label="Completato ✅", state="complete")
        st.success(
            f"Elaborazione completata in **{human_time(elapsed)}**"
            + (f" — RTF ~ **{rtf:.2f}x**" if duration > 0 else "")
        )

        st.subheader("Download")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Scarica testo (.txt)",
                data=txt_bytes,
                file_name=os.path.splitext(uploaded.name)[0] + ".txt",
                mime="text/plain",
                use_container_width=True,
            )
        if with_ts:
            with col2:
                st.download_button(
                    "Scarica sottotitoli (.srt)",
                    data=srt_bytes,
                    file_name=os.path.splitext(uploaded.name)[0] + ".srt",
                    mime="application/x-subrip",
                    use_container_width=True,
                )
            with col3:
                st.download_button(
                    "Scarica sottotitoli (.vtt)",
                    data=vtt_bytes,
                    file_name=os.path.splitext(uploaded.name)[0] + ".vtt",
                    mime="text/vtt",
                    use_container_width=True,
                )

        st.subheader("Anteprima testo")
        st.text_area("Testo finale", value=final_text, height=320, label_visibility="collapsed")

        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ------------------------- SUGGERIMENTI PROMPT -------------------------

with st.expander("Suggerimenti rapidi per il prompt iniziale"):
    st.markdown(
        """
- *"L'audio è in **italiano**. Mantieni una punteggiatura naturale."*
- *"Nomi propri presenti: Paolo Ricca, … (aiuta a scriverli correttamente)."*
- *"Se compaiono termini tecnici teologici, non tradurli."*
- *"Se ci sono sigle (es. **UE**, **ONU**), mantienile in maiuscolo."*
"""
    )
