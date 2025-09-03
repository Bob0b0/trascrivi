# trascrizione.py
# UI Streamlit per trascrivere file audio/video con faster-whisper
# di Roberto M. — timestamp off di default + miglioramento automatico

import os
import io
import time
import math
import tempfile
from datetime import timedelta
from typing import List, Tuple

import streamlit as st
from faster_whisper import WhisperModel

# ffmpeg-python serve solo per leggere la durata
# (su Streamlit Cloud è già presente ffprobe tramite apt)
try:
    import ffmpeg  # type: ignore
except Exception:
    ffmpeg = None


# ------------------------- CONFIGURAZIONE PAGINA -------------------------

st.set_page_config(
    page_title="Trascrizione audio by Roberto M.",
    page_icon="📝",
    layout="wide",
)

TITLE = "Trascrizione audio by Roberto M."
st.title(TITLE)

# ------------------------- UTILS -------------------------


def human_time(seconds: float) -> str:
    seconds = max(0, float(seconds))
    return str(timedelta(seconds=int(seconds)))


def probe_duration(file_path: str) -> float:
    """Rileva la durata del file in secondi usando ffprobe; fallback a 0."""
    if ffmpeg is None:
        return 0.0
    try:
        meta = ffmpeg.probe(file_path)
        # Cerca la prima stream con duration valida
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
    """Pulizia leggera: spaziature, righe spezzate, maiuscole a inizio frase."""
    if not text:
        return ""

    # Normalizza spazi
    import re

    t = text.replace(" \n", "\n").replace("\n ", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    # Unisce righe troppo corte spezzate
    t = re.sub(r"\n{2,}", "\n\n", t)
    lines = [ln.strip() for ln in t.splitlines()]
    # Ricostruisci paragrafi (righe non vuote attaccate)
    paras: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if ln:
            buf.append(ln)
        else:
            if buf:
                paras.append(" ".join(buf))
                buf = []
            paras.append("")  # paragrafo vuoto
    if buf:
        paras.append(" ".join(buf))

    t = "\n\n".join(paras)

    # Aggiusta spazi prima della punteggiatura
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)

    # A capo dopo frasi molto lunghe? No, lasciamo a paragrafi.
    # Maiuscola ad inizio paragrafo
    def cap_first(s: str) -> str:
        for i, ch in enumerate(s):
            if ch.isalpha():
                return s[:i] + ch.upper() + s[i + 1 :]
        return s

    t = "\n\n".join(cap_first(p) if p else "" for p in t.split("\n\n"))
    return t.strip()


def improve_text(text: str, preset: str, custom_rules: str) -> str:
    """
    Migliora/formatta senza LLM: applica tidy + micro-regole in base al preset.
    Nota: non inventa contenuti, solo cosmetica.
    """
    t = tidy_text(text)

    # Preset semplici
    if preset == "Pulito neutro":
        return t

    import re

    if preset == "Paragrafi leggibili":
        # Cerca di separare dopo punti seguiti da spazio + maiuscola
        t = re.sub(r"([.!?])\s+(?=[A-ZÀ-ÖØ-Þ])", r"\1\n\n", t)
        return tidy_text(t)

    if preset == "Appunti sintetici":
        # Trasforma frasi in bullet se sufficientemente corte
        sentences = re.split(r"(?<=[.!?])\s+", t)
        bullets = []
        for s in sentences:
            s = s.strip("-• ").strip()
            if not s:
                continue
            if len(s) <= 180:  # euristica
                bullets.append(f"• {s}")
            else:
                bullets.append(s)
        return "\n".join(bullets).strip()

    # Applica eventuali regole testuali: è solo un promemoria per l'utente,
    # qui facciamo una rifinitura minima (non è un LLM).
    if custom_rules.strip():
        # Alcuni pattern comuni
        t = re.sub(r"\s{3,}", "  ", t)
    return t


def to_srt(segments: List[Tuple[float, float, str]]) -> str:
    """Converte segmenti in SRT."""
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
        lines.append("")  # riga vuota
    return "\n".join(lines).strip() + "\n"


def to_vtt(segments: List[Tuple[float, float, str]]) -> str:
    """Converte segmenti in WebVTT."""
    def ts(x: float) -> str:
        ms = int((x - int(x)) * 1000)
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        # WebVTT consente HH:MM:SS.mmm (con punto)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for (start, end, txt) in segments:
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(txt.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, compute_type: str = "int8") -> WhisperModel:
    # device='cpu' for Streamlit Cloud; compute_type int8 per RAM ridotta
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


# ------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.subheader("Legenda & note")
    st.markdown(
        """
- **Modelli disponibili**: `tiny`, `base`, `small`, `medium` (il **large è disabilitato** perché richiede troppa RAM sulla piattaforma).
- **Timestamp**: disattivati di default. Abilitali solo se ti servono i file **SRT/VTT**.
- **Migliora e formatta automaticamente**: applica pulizia e formattazione base, senza inventare contenuti.
- **Prompt iniziale (opzionale)**: frase che aiuta il modello (es. *"Nomi propri: Paolo Ricca; lingua italiana"*).
- I tempi dipendono dalla lunghezza dell’audio e dal modello scelto.
"""
    )
    st.caption("Suggerimento: per file lunghi usa `base` o `small` per un buon compromesso.")


# ------------------------- OPZIONI AVANZATE -------------------------

with st.expander("Opzioni avanzate (riassunto)"):
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        model_size = st.selectbox(
            "Seleziona modello",
            options=["tiny", "base", "small", "medium"],  # large escluso
            index=1,
            help="Modelli più grandi = maggiore qualità ma più memoria/tempo.",
        )
    with colB:
        language = st.selectbox(
            "Lingua dell'audio",
            options=["auto", "it", "en", "fr", "de", "es", "pt", "ro"],
            index=0,
            help="Se non sei sicuro, lascia **auto**.",
        )
    with colC:
        initial_prompt = st.text_input(
            "Prompt iniziale (opzionale)",
            value="",
            help="Esempio: 'L'audio è in italiano; nomi propri: Paolo Ricca; usa punteggiatura naturale.'",
        )

# Opzioni principali (semplificate)
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
    "Drag and drop file qui",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

start_btn = st.button("Avvia trascrizione", type="primary", disabled=(uploaded is None))

# Area di stato
status_box = st.container()

# ------------------------- ELABORAZIONE -------------------------

if start_btn and uploaded is not None:
    t0 = time.time()

    with status_box:
        st.write("### Inizio elaborazione…")
        status = st.status("Preparazione…", expanded=True)

        # Step 1 – prepara file temporaneo
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

        # Step 2 – carica modello (cache_resource evita ricarichi)
        status.update(label=f"Carico il modello `{model_size}`…", state="running")
        model = load_model(model_size, compute_type="int8")
        status.write("Modello pronto.")

        # Step 3 – trascrizione con barra di avanzamento
        status.update(label="Trascrivo…", state="running")
        progress = st.progress(0)
        seg_list: List[Tuple[float, float, str]] = []

        # Parametri whisper
        transcribe_kwargs = dict(
            language=None if language == "auto" else language,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            initial_prompt=initial_prompt or None,
        )

        # Generatore segmenti: aggiorniamo progress in base al tempo "end" del segmento
        total = max(duration, 1.0)
        try:
            segments, info = model.transcribe(tmp_path, **transcribe_kwargs)
            for seg in segments:
                seg_list.append((seg.start, seg.end, seg.text.strip()))
                # update progress
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

        # Step 4 – testo grezzo
        raw_text = "\n".join(t for _, _, t in seg_list).strip()

        # Step 5 – migliora automaticamente (se abilitato)
        status.update(label="Rifinitura testo…", state="running")
        final_text = improve_text(raw_text, preset=preset, custom_rules=custom_rules) if auto_improve else raw_text

        # Step 6 – prepara export
        txt_bytes = final_text.encode("utf-8")
        srt_bytes = to_srt(seg_list).encode("utf-8") if with_ts else None
        vtt_bytes = to_vtt(seg_list).encode("utf-8") if with_ts else None

        # Elapsed
        elapsed = time.time() - t0
        rtf = (elapsed / duration) if duration > 0 else float("nan")
        status.update(label="Completato ✅", state="complete")
        st.success(
            f"Elaborazione completata in **{human_time(elapsed)}**"
            + (f" — RTF ~ **{rtf:.2f}x**" if duration > 0 else "")
        )

        # ------------------------- DOWNLOAD -------------------------
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

        # Mostra anteprima testo
        st.subheader("Anteprima testo")
        st.text_area("Testo finale", value=final_text, height=320, label_visibility="collapsed")

        # Cleanup file temporaneo
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ------------------------- SUGGERIMENTI PROMPT (FOOTER) -------------------------

with st.expander("Suggerimenti rapidi per il prompt iniziale"):
    st.markdown(
        """
- *"L'audio è in **italiano**. Mantieni una punteggiatura naturale."*
- *"Nomi propri presenti: Paolo Ricca, ... (aiuta a scriverli correttamente)"*
- *"Se compaiono termini tecnici teologici, non tradurli in inglese."*
- *"Se ci sono sigle (es. **UE**, **ONU**), mantienile in maiuscolo."*
"""
    )
