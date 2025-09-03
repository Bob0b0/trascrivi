# trascrizione.py
# Streamlit app – Trascrizione audio/video con Faster-Whisper
# by Roberto M. (semplificata: qualità sempre attiva, timestamp opzionale)

from __future__ import annotations

import io
import os
import re
import textwrap
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import ffmpeg  # usa ffprobe sotto il cofano
import streamlit as st
from faster_whisper import WhisperModel


# ============================ Utilità di formattazione ============================

def human_time(seconds: float) -> str:
    """Ritorna tempo HH:MM:SS.mmm da secondi (float)."""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def srt_time(seconds: float) -> str:
    """Tempo SRT -> HH:MM:SS,mmm"""
    return human_time(seconds).replace(".", ",")


def get_media_duration(path: str) -> Optional[float]:
    """Prova a leggere la durata via ffprobe; se fallisce, None."""
    try:
        probe = ffmpeg.probe(path)
        # Cerca prima la durata del primo stream con duration
        for stream in probe.get("streams", []):
            if "duration" in stream:
                return float(stream["duration"])
        # Altrimenti dal formato
        fmt = probe.get("format", {})
        if "duration" in fmt:
            return float(fmt["duration"])
    except Exception:
        pass
    return None


def tidy_text(txt: str) -> str:
    """
    Pulizia leggera e deterministica:
    - spaziature e punteggiatura,
    - rimozione ripetizioni immediate (2-3 volte),
    - normalizzazione maiuscole a inizio frase.
    Non inventa contenuti.
    """
    if not txt:
        return ""

    # spazi multipli -> singolo; normalizza spazi prima/ dopo punteggiatura
    t = re.sub(r"\s+", " ", txt)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"\s+([)\]])", r"\1", t)

    # ripetizioni immediate di 1-3 parole (case-insensitive)
    def _dedup(m):
        return m.group(1)

    t = re.sub(r"\b([A-Za-zÀ-ÖØ-öø-ÿ0-9']{2,})\b(?:\s+\1\b){1,2}", _dedup, t, flags=re.IGNORECASE)

    # capitalizza inizio frase semplice (dopo .?!)
    sentences = re.split(r"([.?!])\s*", t)
    rebuilt = []
    buf = ""
    for i, chunk in enumerate(sentences):
        if i % 2 == 0:  # testo
            chunk = chunk.strip()
            if not chunk:
                continue
            if not buf:  # inizio frase
                chunk = chunk[:1].upper() + chunk[1:]
            buf += chunk
        else:  # segno .?!
            buf += chunk + " "
            rebuilt.append(buf.strip())
            buf = ""
    if buf:
        rebuilt.append(buf.strip())

    t = " ".join(rebuilt).strip()
    return t


def wrap_text(txt: str, width: int = 90) -> str:
    """Impaginazione semplice per lettura: a capo morbido ~width col."""
    if not txt:
        return ""
    # Mantieni eventuali doppi a capo come paragrafi
    paragraphs = re.split(r"\n\s*\n", txt.strip())
    wrapped = [textwrap.fill(p.strip(), width=width, break_long_words=False, replace_whitespace=False) for p in paragraphs if p.strip()]
    return "\n\n".join(wrapped)


@dataclass
class Segment:
    start: float
    end: float
    text: str


def build_srt(segments: List[Segment]) -> str:
    """Costruisce SRT dai segmenti."""
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{srt_time(s.start)} --> {srt_time(s.end)}")
        lines.append(s.text.strip())
        lines.append("")  # riga vuota
    return "\n".join(lines).strip() + "\n"


def build_vtt(segments: List[Segment]) -> str:
    """Costruisce VTT dai segmenti."""
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{human_time(s.start)} --> {human_time(s.end)}")
        lines.append(s.text.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def make_ai_prompt(lang_hint: str = "italiano") -> str:
    """Prompt suggerito per post-processing con un LLM esterno (per copia/incolla)."""
    return textwrap.dedent(f"""
    Migliora il seguente testo trascritto da audio, **senza inventare contenuti**.

    Obiettivi:
    - correggi errori ortografici e di punteggiatura;
    - elimina ripetizioni e tic verbali;
    - modernizza forme lessicali desuete mantenendo lo stile dell'autore;
    - aggiusta la sintassi per scorrevolezza, preservando il senso;
    - mantieni i nomi propri esattamente come appaiono;
    - restituisci testo ben formattato con capoversi regolari.

    Lingua di uscita: {lang_hint}.
    Restituisci SOLO il testo finale.

    Testo da migliorare:
    ---
    {{INCOLLA QUI IL .TXT PULITO}}
    ---
    """).strip()


# ============================ Cache modello ============================

@st.cache_resource(show_spinner=False)
def load_model_cached(model_size: str, compute_type: str) -> WhisperModel:
    """
    Carica (e cache) il modello. Forziamo local_files_only=False per consentire
    il download la prima volta. Impostiamo anche una cartella cache stabile.
    """
    download_root = os.path.expanduser("~/.cache/faster_whisper")
    os.makedirs(download_root, exist_ok=True)
    return WhisperModel(
        model_size,
        compute_type=compute_type,
        local_files_only=False,           # <-- fix per l'errore visto
        download_root=download_root,
    )


# ============================ Trascrizione con Faster-Whisper ============================

def run_transcription(
    file_path: str,
    model_size: str,
    compute_type: str,
    language: str,
    want_timestamps: bool,
    status_area
) -> Tuple[str, str, Optional[str], Optional[str], List[Segment], float, str]:
    """
    Esegue la trascrizione e restituisce:
      raw_text, tidy, srt, vtt, segments, duration, detected_language
    """
    with status_area.status("Carico il modello…"):
        model = load_model_cached(model_size, compute_type)

    duration = get_media_duration(file_path) or 0.0

    with status_area.status("Trascrivo…"):
        segments_gen, info = model.transcribe(
            file_path,
            language=None if language == "auto" else language,
            task="transcribe",
            vad_filter=True,
            beam_size=5,
        )

        detected_lang = info.language or language
        segments: List[Segment] = []
        raw_parts: List[str] = []

        prog = st.progress(0.0) if duration > 0 else None
        last_end = 0.0

        for seg in segments_gen:
            s = Segment(start=seg.start or 0.0, end=seg.end or 0.0, text=seg.text or "")
            segments.append(s)
            raw_parts.append(s.text.strip())

            if prog and duration > 0:
                last_end = max(last_end, s.end or last_end)
                prog.progress(min(1.0, last_end / max(duration, 1e-6)))

        if prog:
            prog.empty()

    raw_text = " ".join(raw_parts).strip()
    tidy = tidy_text(raw_text)
    pretty = wrap_text(tidy, width=90)

    srt_text = build_srt(segments) if want_timestamps else None
    vtt_text = build_vtt(segments) if want_timestamps else None

    return raw_text, pretty, srt_text, vtt_text, segments, duration, detected_lang or "auto"


# ================================== UI ==================================

st.set_page_config(
    page_title="Trascrizione audio by Roberto M.",
    page_icon="📝",
    layout="wide",
)

# --- Sidebar: impostazioni "tecniche" ---
st.sidebar.header("Impostazioni")

model_size = st.sidebar.selectbox(
    "Modello Whisper",
    options=["tiny", "base", "small", "medium"],  # large disattivato (limite RAM)
    index=1,
    help="Scegli un modello. 'base' è un buon compromesso tra qualità e velocità. (large disattivato su questa piattaforma)"
)

compute_type = st.sidebar.selectbox(
    "Calcolo (CPU)",
    options=["int8 (consigliato)", "float16", "float32"],
    index=0,
    help="int8 è il più leggero. float16/float32 possono migliorare leggermente la qualità ma richiedono più RAM."
)
compute_map = {
    "int8 (consigliato)": "int8",
    "float16": "float16",
    "float32": "float32",
}
compute_type = compute_map[compute_type]

language = st.sidebar.selectbox(
    "Lingua dell'audio",
    options=["auto", "it", "en", "fr", "de", "es", "pt"],
    index=0,
    help="Se non sei sicuro, lascia 'auto'."
)

# Guida rapida (pulsante in sidebar)
show_readme = st.sidebar.checkbox("Mostra guida rapida", value=False)
if show_readme:
    with st.expander("Guida rapida (come usare l’app)", expanded=True):
        st.markdown(
            """
**3 passi semplici**
1. Carica un file audio/video.
2. Premi **Avvia trascrizione**.
3. Scarica il testo pulito (TXT), la versione impaginata per lettura e—se richiesto—SRT/VTT con timestamp.

**Cosa fa la “Qualità automatica”**
- Corregge punteggiatura, ripetizioni e piccoli refusi.
- Non inventa contenuti.
- Genera anche una versione impaginata (~90 colonne) per una lettura più comoda.

**Suggerimento**
- I modelli disponibili sono *tiny, base, small, medium*.  
  *large* è disattivato perché non compatibile con la RAM della piattaforma.
            """
        )

# --- Titolo pagina ---
st.title("Trascrizione audio by Roberto M.")

# --- Opzioni avanzate: SOLO timestamp ---
with st.expander("Opzioni avanzate", expanded=False):
    want_ts = st.checkbox(
        "Includi timestamp (SRT/VTT)",
        value=False,
        help="Per la maggioranza degli utenti non serve. Abilitalo se ti servono SRT/VTT."
    )

# Flag qualità: sempre ON (come richiesto)
AUTO_FIX = True     # revisione/correzione deterministica
PRETTY_TXT = True   # impaginazione a capo morbido

# --- Uploader ---
uploaded = st.file_uploader(
    "Carica un file audio/video",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
    help="Limite 200MB per file",
)

# Istruzioni sintetiche che spariscono all’avvio
guide_box = st.empty()
if not st.session_state.get("processing", False) and "results" not in st.session_state:
    with guide_box.container():
        st.markdown(
            """
**Come funziona (3 passi):**
1) Carica il file.  
2) Clicca **Avvia trascrizione**.  
3) Scarica gli output finali.

**Nota:** la revisione automatica corregge punteggiatura, ripetizioni e piccoli refusi; il testo *non viene inventato*.
            """
        )

start = st.button("Avvia trascrizione", disabled=uploaded is None)

# Se abbiamo già dei risultati in sessione, mostriamoli sempre
def show_results():
    res = st.session_state["results"]
    file_label = f"**File:** {res.get('filename','(sconosciuto)')}"
    st.subheader("Risultati")
    st.caption(file_label)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Testo pulito (pronto per AI / editing)**")
        st.text_area("Anteprima", value=res["tidy"], height=280, label_visibility="collapsed")
        st.download_button(
            "Scarica TXT pulito",
            data=res["tidy"].encode("utf-8"),
            file_name=(res.get("stem","trascrizione") + "_pulito.txt"),
            mime="text/plain",
        )

    with col2:
        st.markdown("**Versione impaginata per lettura (~90 col.)**")
        st.text_area("Anteprima leggibile", value=res["pretty"], height=280, label_visibility="collapsed")
        st.download_button(
            "Scarica TXT leggibile",
            data=res["pretty"].encode("utf-8"),
            file_name=(res.get("stem","trascrizione") + "_leggibile.txt"),
            mime="text/plain",
        )

    if res.get("srt"):
        st.markdown("**Sottotitoli (SRT/VTT)**")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Scarica SRT",
                data=res["srt"].encode("utf-8"),
                file_name=(res.get("stem","trascrizione") + ".srt"),
                mime="application/x-subrip",
            )
        with c2:
            st.download_button(
                "Scarica VTT",
                data=res["vtt"].encode("utf-8"),
                file_name=(res.get("stem","trascrizione") + ".vtt"),
                mime="text/vtt",
            )

    # Prompt AI sempre disponibile finché i risultati restano in sessione
    st.markdown("---")
    st.markdown("### Prompt suggerito per migliorare ulteriormente con un'AI")
    prompt_text = make_ai_prompt(lang_hint=res.get("lang","italiano"))
    st.code(prompt_text, language="markdown")
    st.info("Copia il prompt, **incolla il TXT pulito** e utilizza il tuo modello AI preferito. Non vengono eseguite chiamate esterne da questa app.")

if "results" in st.session_state:
    show_results()

# --- Avvio elaborazione ---
if start and uploaded is not None:
    guide_box.empty()
    st.session_state["processing"] = True

    # Salva su file temporaneo
    stem, _ = os.path.splitext(uploaded.name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Area di stato progressivo
    st.subheader("Elaborazione")
    status_area = st.container()

    try:
        raw, pretty, srt_text, vtt_text, segments, duration, det_lang = run_transcription(
            file_path=tmp_path,
            model_size=model_size,
            compute_type=compute_type,
            language=language,
            want_timestamps=want_ts,
            status_area=status_area
        )

        results = {
            "filename": uploaded.name,
            "stem": stem,
            "raw": raw,
            "tidy": tidy_text(raw) if AUTO_FIX else raw,
            "pretty": pretty if PRETTY_TXT else tidy_text(raw),
            "srt": srt_text,
            "vtt": vtt_text,
            "lang": det_lang if det_lang != "auto" else "italiano",
            "duration": duration,
        }
        st.session_state["results"] = results

        # Mostra riassunto
        dur_s = results.get("duration") or 0.0
        mm = int(dur_s // 60)
        ss = int(dur_s % 60)
        st.success(f"Completato! Durata audio rilevata: {mm:02d}:{ss:02d}. Lingua: {results.get('lang','it')}.")

        show_results()

    except Exception as e:
        st.error("Si è verificato un errore durante l'elaborazione.")
        st.exception(e)
        st.info("Se vedi un errore tipo 'LocalEntryNotFoundError', significa che il modello non era in cache e il download è stato bloccato. "
                "Riprova più tardi o verifica che l'accesso a Internet dell'app sia consentito.")
    finally:
        # Pulisci file temporaneo
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        st.session_state["processing"] = False
