# trascrizione.py
# ------------------------------------------------------------
# Trascrizione audio/video con faster-whisper + Streamlit
# UI minimale: revisione & formattazione automatiche sempre attive
# README integrato (sidebar) con flag per Popover o pagina intera
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import io
import time
import textwrap
import tempfile
from typing import List, Tuple, Optional

import streamlit as st
import numpy as np

# Audio utils
import ffmpeg  # apt-get + ffmpeg-python nel requirements
from faster_whisper import WhisperModel

# =========================
# Configurazioni globali
# =========================

APP_TITLE = "Trascrizione audio by Roberto M."
# Modalità di visualizzazione README nella sidebar:
#  - "popover"  → piccola finestra affiancata al pulsante (fallback Expander)
#  - "fullpage" → vista a pagina intera (toggle nella pagina principale)
READ_ME_UI = "popover"  # oppure: "fullpage"

# Modelli disponibili (large disabilitato per limiti RAM su Streamlit Cloud)
WHISPER_MODELS = ["tiny", "base", "small", "medium"]  # niente "large"

# Tipi di calcolo consigliati: int8 → più leggero; float16 → più qualità (se GPU)
COMPUTE_TYPES = {
    "int8 (consigliato)": "int8",
    "float16 (GPU)": "float16",
}

# =========================
# Utility
# =========================

def load_readme_text() -> str:
    """Carica il README.md dal repository, con fallback in chiaro."""
    fallback = """\
# Trascrizione audio by Roberto M.

Questa app trascrive file audio/video usando **faster-whisper**.

**Come si usa**
1. Carica un file audio o video (max 200MB su Streamlit Cloud).
2. Clicca **Avvia trascrizione**.
3. Scarica il testo *pulito e formattato* (.txt). Opzionalmente esporta SRT/VTT.
4. (Facoltativo) Usa il **prompt per l’AI** per un’ulteriore rifinitura editoriale.

**Suggerimenti**
- Lascia *auto* per la lingua se non sei sicuro.
- Per file lunghi, usa modello **base** o **small** con calcolo **int8**.
- Il modello **large** è disabilitato in questa istanza per motivi di RAM.

Buon lavoro!"""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback


def seconds_to_hms(seconds: float) -> str:
    seconds = int(seconds or 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def probe_duration(path: str) -> Optional[float]:
    """Ottieni la durata con ffmpeg.probe (in secondi)."""
    try:
        info = ffmpeg.probe(path)
        dur = float(info["format"]["duration"])
        return dur
    except Exception:
        return None


def tidy_transcript(text: str) -> str:
    """Pulizia leggera: spazi, ripetizioni semplici, punteggiatura basica."""
    if not text:
        return ""
    # Spazi multipli
    text = re.sub(r"[ \t]+", " ", text)
    # Spazi prima della punteggiatura
    text = re.sub(r" \,", ",", text)
    text = re.sub(r" \.", ".", text)
    text = re.sub(r" \;", ";", text)
    text = re.sub(r" \:", ":", text)
    # Doppia punteggiatura
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"([!?]){2,}", r"\1", text)
    # Ripetizioni immediate tipo "che che", "io io"
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    # Trim
    text = text.strip()
    return text


def pretty_wrap(text: str, width: int = 90) -> str:
    """Formatta a capo morbido per una lettura comoda (~90 col)."""
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width))
    return "\n".join(lines)


def segments_to_srt(segments: List) -> str:
    def _ts(t: float) -> str:
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _ts(seg.start)
        end = _ts(seg.end)
        text = seg.text.strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks).strip()


def segments_to_vtt(segments: List) -> str:
    def _ts(t: float) -> str:
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    blocks = ["WEBVTT\n"]
    for seg in segments:
        start = _ts(seg.start)
        end = _ts(seg.end)
        text = seg.text.strip()
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks).strip()


def build_ai_prompt(text: str, lang: str | None) -> str:
    """Suggerimento di prompt per migliorare ulteriormente con un LLM."""
    lingua = lang or "auto"
    return textwrap.dedent(f"""\
    SEI UN EDITOR PROFESSIONISTA.
    Migliora il testo seguente senza alterare i contenuti: correggi refusi,
    punteggiatura, ripetizioni, sintassi, lessico datato; applica una 
    formattazione leggibile con paragrafi e titoli minimi se opportuno.
    Mantieni lo stile orale, ma rendilo scorrevole. Lingua: {lingua}.

    TESTO DA MIGLIORARE (tra i delimitatori):
    ---
    {text}
    ---
    Output richiesto: testo finale pulito, pronto per la pubblicazione.
    """)


def ensure_session_state():
    if "results" not in st.session_state:
        st.session_state.results = None
    if "show_readme" not in st.session_state:
        st.session_state.show_readme = False


# =========================
# UI – README nella sidebar
# =========================

def sidebar_readme_button():
    md = load_readme_text()

    st.sidebar.markdown("### ❓ Guida / README")

    if READ_ME_UI == "popover":
        # popover è disponibile nelle versioni recenti; fallback a expander
        try:
            with st.sidebar.popover("Apri guida rapida"):
                st.markdown(md)
        except Exception:
            with st.sidebar.expander("Apri guida rapida"):
                st.markdown(md)
    else:
        # fullpage: toggle che mostra il README nella pagina principale
        if st.sidebar.button("Mostra README a pagina intera", use_container_width=True):
            st.session_state.show_readme = True


# =========================
# Logica di trascrizione
# =========================

def run_transcription(
    path: str,
    model_name: str,
    compute_type: str,
    language: str | None,
    export_subs: bool,
) -> Tuple[str, str, Optional[str], Optional[str], List, Optional[float], Optional[str]]:
    """
    Esegue la trascrizione e restituisce:
    raw_text, pretty_text, srt, vtt, segments, duration, detected_language
    """
    # 1) Carica modello
    # local_files_only=False → evita l'errore di snapshot mancante
    model = WhisperModel(
        model_name,
        compute_type=compute_type,
        local_files_only=False,
    )

    # 2) Trascrivi
    # Revisione & pulizia applicate in seguito (sempre attive)
    segments, info = model.transcribe(
        path,
        vad_filter=True,
        language=None if (language in [None, "", "auto"]) else language,
        beam_size=5,
        initial_prompt=None,
    )

    segment_list = list(segments)  # materializza il generatore
    detected_lang = getattr(info, "language", None)

    # 3) Riassembla testo grezzo
    raw = "".join([s.text for s in segment_list]).strip()
    raw = tidy_transcript(raw)

    # 4) Formattazione leggibile
    pretty = pretty_wrap(raw, width=90)

    # 5) Sottotitoli opzionali
    srt_text = segments_to_srt(segment_list) if export_subs else None
    vtt_text = segments_to_vtt(segment_list) if export_subs else None

    # 6) Durata (da ffmpeg) – se manca usa info.duration
    duration = probe_duration(path)
    if not duration:
        try:
            duration = float(getattr(info, "duration", None) or 0.0)
        except Exception:
            duration = None

    return raw, pretty, srt_text, vtt_text, segment_list, duration, detected_lang


# =========================
# Pagina principale
# =========================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_session_state()

    # Sidebar – README
    sidebar_readme_button()

    # Modalità README a pagina intera
    if READ_ME_UI == "fullpage" and st.session_state.show_readme:
        st.title("📘 README / Guida")
        st.markdown(load_readme_text())
        st.button("Chiudi", on_click=lambda: st.session_state.update(show_readme=False))
        return

    # Titolo
    st.title(APP_TITLE)

    # Impostazioni (minimali) in sidebar
    st.sidebar.markdown("### Impostazioni")

    model_name = st.sidebar.selectbox(
        "Modello Whisper",
        options=WHISPER_MODELS,
        index=1,  # default "base"
        help="Usa tiny/base per file brevi, small/medium per file più lunghi.",
    )

    compute_key = st.sidebar.selectbox(
        "Calcolo (CPU/GPU)",
        options=list(COMPUTE_TYPES.keys()),
        index=0,
        help="int8 è leggero (consigliato su Cloud). float16 se hai GPU.",
    )
    compute_type = COMPUTE_TYPES[compute_key]

    language = st.sidebar.selectbox(
        "Lingua dell'audio",
        options=["auto", "it", "en", "fr", "de", "es", "pt", "ro", "pl", "nl"],
        index=0,
        help="Lascia 'auto' se non sei sicuro.",
    )
    language = None if language == "auto" else language

    with st.sidebar.expander("Opzioni avanzate", expanded=False):
        export_subs = st.checkbox(
            "Esporta anche SRT/VTT (timestamp)",
            value=False,
            help="Disattivo di default. Abilitalo solo se ti servono i sottotitoli.",
        )

    # Upload
    st.subheader("Carica un file audio/video")
    up = st.file_uploader(
        "Drag and drop oppure clicca su 'Browse files'",
        type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
        label_visibility="collapsed",
    )

    # Istruzioni sintetiche (spariscono all’avvio o quando ci sono risultati)
    if not st.session_state.results:
        with st.container(border=True):
            st.markdown(
                """
**Come funziona (in breve)**  
1) Carica il file → 2) Avvia trascrizione → 3) Scarica il **Testo pulito**.  
Al termine vedrai anche un **prompt per l’AI** per un’ulteriore rifinitura editoriale.

**Note**
- Pulizia & formattazione sono **sempre attive**.  
- SRT/VTT sono **opzionali** (abilita nelle *Opzioni avanzate*).  
- Per file lunghi: modello **base/small** + **int8**.
                """
            )

    # Avvio trascrizione
    start_btn_disabled = up is None
    start = st.button("Avvia trascrizione", type="primary", disabled=start_btn_disabled)

    if start and up is not None:
        # Salva su file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{up.name}") as tmp:
            tmp.write(up.read())
            temp_path = tmp.name

        # Status/Progress
        try:
            status = st.status("Inizio elaborazione…", expanded=True)
            status.write("① Carico il modello…")
        except Exception:
            status = None
            st.write("Inizio elaborazione…")
            prog = st.progress(0)

        t0 = time.time()
        try:
            # Step 1: (status già scritto)
            if status is None:
                prog.progress(10)

            # Step 2: Trascrizione
            if status:
                status.write("② Trascrivo…")
            raw, pretty, srt_text, vtt_text, segments, duration, det_lang = run_transcription(
                temp_path, model_name, compute_type, language, export_subs
            )
            if status is None:
                prog.progress(80)

            # Step 3: Fine
            elapsed = time.time() - t0
            if status:
                status.update(label="Elaborazione completata ✅", state="complete")
            else:
                prog.progress(100)
                st.success("Elaborazione completata ✅")

            # Salva risultati in sessione (persistono anche dopo i download)
            st.session_state.results = {
                "raw": raw,
                "pretty": pretty,
                "srt": srt_text,
                "vtt": vtt_text,
                "segments": segments,
                "duration": duration,
                "det_lang": det_lang,
                "elapsed": elapsed,
                "filename": up.name,
            }

        except Exception as e:
            if status:
                status.update(label="Errore durante l'elaborazione", state="error")
                status.write(str(e))
            else:
                st.error(f"Errore: {e}")
        finally:
            # Pulisci il file temporaneo
            try:
                os.remove(temp_path)
            except Exception:
                pass

    # Se ci sono risultati, mostriamoli
    res = st.session_state.results
    if res:
        st.divider()
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown("### ⏺️ Riepilogo")
            dur = seconds_to_hms(res["duration"]) if res["duration"] else "—"
            st.write(f"**File:** {res['filename']}")
            st.write(f"**Durata audio:** {dur}")
            st.write(f"**Modello:** `{model_name}`  |  **Calcolo:** `{compute_type}`")
            st.write(f"**Lingua rilevata:** `{res['det_lang'] or 'n/d'}`")
            st.write(f"**Tempo elaborazione:** {seconds_to_hms(res['elapsed'])}")

            st.markdown("### ⬇️ Download")
            st.download_button(
                "Scarica testo pulito (.txt)",
                data=res["pretty"].encode("utf-8"),
                file_name=f"{os.path.splitext(res['filename'])[0]}_pulito.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.download_button(
                "Scarica testo grezzo (.txt)",
                data=res["raw"].encode("utf-8"),
                file_name=f"{os.path.splitext(res['filename'])[0]}_grezzo.txt",
                mime="text/plain",
                use_container_width=True,
            )

            if res["srt"]:
                st.download_button(
                    "Scarica sottotitoli SRT",
                    data=res["srt"].encode("utf-8"),
                    file_name=f"{os.path.splitext(res['filename'])[0]}.srt",
                    mime="text/plain",
                    use_container_width=True,
                )
            if res["vtt"]:
                st.download_button(
                    "Scarica sottotitoli VTT",
                    data=res["vtt"].encode("utf-8"),
                    file_name=f"{os.path.splitext(res['filename'])[0]}.vtt",
                    mime="text/vtt",
                    use_container_width=True,
                )

        with right:
            st.markdown("### 📄 Testo pulito (formattato)")
            st.text_area(
                "Anteprima (solo lettura)",
                value=res["pretty"],
                height=400,
                label_visibility="collapsed",
            )

        # Prompt per LLM
        st.divider()
        st.markdown("### 💡 Prompt per miglioramento con AI")
        prompt_text = build_ai_prompt(res["pretty"], res["det_lang"])
        st.code(prompt_text, language="markdown")

        st.download_button(
            "Scarica prompt (.txt)",
            data=prompt_text.encode("utf-8"),
            file_name=f"{os.path.splitext(res['filename'])[0]}_prompt_AI.txt",
            mime="text/plain",
        )

        st.caption(
            "Suggerimento: apri il tuo LLM preferito (es. ChatGPT) e incolla il prompt "
            "insieme al **testo pulito** scaricato sopra."
        )

    # Footer piccolo
    st.write("")
    st.caption("© Roberto M. — powered by faster-whisper + Streamlit")

# Entrypoint
if __name__ == "__main__":
    main()

