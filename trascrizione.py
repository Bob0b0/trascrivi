# trascrivi.py
# App Streamlit per trascrivere audio con faster-whisper
# ✎ Modifica qui il nome autore mostrato in alto a destra:
AUTHOR_NAME = "Roberto M."

import os
import io
import math
import time
import json
import queue
import shutil
import string
import tempfile
import subprocess
from datetime import datetime
import threading

import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg


# ----------------------------- Utility ----------------------------- #

def _human_time(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _sanitize_filename(name: str) -> str:
    valid = f"-_.() {string.ascii_letters}{string.digits}"
    clean = "".join(c if c in valid else "_" for c in name).strip(" ._")
    return clean or "file"

def _ensure_session_dir():
    if "session_dir" not in st.session_state:
        st.session_state.session_dir = tempfile.mkdtemp(prefix="trascrivi_")
    st.session_state.setdefault("saved_files", [])
    st.session_state.setdefault("transcript_text", "")
    st.session_state.setdefault("ai_prompt_text", "")

def _save_text_session(basename: str, text: str) -> str:
    _ensure_session_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_filename(basename.rsplit(".", 1)[0])
    path = os.path.join(st.session_state.session_dir, f"{ts}_{base}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    label = os.path.basename(path)
    st.session_state.saved_files.append({"path": path, "label": label})
    return path

def _check_ffprobe() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def _probe_duration_seconds(path: str) -> float:
    if not _check_ffprobe():
        raise RuntimeError("ffprobe non trovato. Installa ffmpeg/ffprobe.")
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    info = json.loads(p.stdout.decode("utf-8"))
    return float(info["format"]["duration"])

def _export_wav_segment(src_path: str, start_s: float, dur_s: float, dst_path: str):
    (
        ffmpeg.input(src_path, ss=max(0, start_s), t=max(0.1, dur_s))
              .output(dst_path, format="wav", ac=1, ar=16000, acodec="pcm_s16le")
              .overwrite_output()
              .run(quiet=True)
    )

def _plan_chunks(duration_s: float, target_min: float = 10.5, max_min: float = 12.0, overlap_s: float = 10.0):
    n = max(1, round(duration_s / (target_min * 60.0)))
    while (duration_s / n) > (max_min * 60.0):
        n += 1
    chunk_len = duration_s / n
    chunks = []
    for i in range(n):
        start = i * chunk_len
        if i > 0:
            start = max(0.0, start - overlap_s)
        end = min(duration_s, (i + 1) * chunk_len)
        dur = max(0.1, end - start)
        chunks.append((start, dur))
    return chunks, n, (chunk_len / 60.0)

def _build_ai_prompt(transcript_text: str) -> str:
    transcript_text = transcript_text.strip()
    return (
        "SEI UN EDITOR PROFESSIONISTA.\n\n"
        "Migliora il testo seguente senza alterare i contenuti: correggi refusi,\n"
        "punteggiatura, ripetizioni, sintassi, lessico datato; applica una\n"
        "formattazione leggibile con paragrafi e titoli minimi se opportuno.\n"
        "Mantieni lo stile orale, ma rendilo scorrevole. Lingua: it.\n"
        "<< TESTO DA INSERIRE >>\n"
        "Output richiesto: testo finale pulito, pronto per la pubblicazione.\n"
        "§§§§§§\n"
        "questo era il testo (nel caso l'avessi perso)\n\n"
        "TESTO DA MIGLIORARE (tra i delimitatori):\n"
        "§§§§§§\n"
        f"{transcript_text}\n"
        "§§§§§§\n"
    )


# ----------------------------- UI helpers ----------------------------- #

def _header(title="Trascrivi — Whisper (faster-whisper)"):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(f"## {title}")
    with col2:
        st.markdown(f"<div style='text-align:right;color:#666;'>Autore: <b>{AUTHOR_NAME}</b></div>", unsafe_allow_html=True)

def _reset_session_state():
    st.session_state["saved_files"] = []
    st.session_state["transcript_text"] = ""
    st.session_state["ai_prompt_text"] = ""
    st.toast("Sessione reimpostata. I file già scaricati in locale restano.", icon="✅")

def _sidebar_settings():
    st.sidebar.header("Impostazioni")

    model_size = st.sidebar.selectbox(
        "Modello Whisper",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        key="model_size_sel",
        help="Modello di trascrizione."
    )
    language = st.sidebar.text_input("Lingua", value="it", key="lang_sel",
                                     help="Codice lingua ISO (es. it, en, fr).")

    st.sidebar.caption("Blocchi: automatico 10–12 minuti (overlap 10s).")

    st.sidebar.subheader("File salvati (sessione)")
    if st.session_state.get("saved_files"):
        for i, item in enumerate(st.session_state.saved_files):
            with open(item["path"], "rb") as f:
                st.sidebar.download_button(
                    "Scarica",
                    data=f.read(),
                    file_name=item["label"],
                    key=f"dl_{item['label']}",
                    help=item["label"]
                )
            st.sidebar.caption(item["label"])
    else:
        st.sidebar.caption("Nessun file salvato in questa sessione.")

    st.sidebar.button("🔄 Reimposta sessione\n*(non cancella file locali già scaricati)*",
                      on_click=_reset_session_state)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Prompt per AI (post-produzione)")
    if st.session_state.get("ai_prompt_text"):
        st.sidebar.text_area("Prompt completo (copia & incolla ovunque)",
                             value=st.session_state["ai_prompt_text"], height=260, key="ai_prompt_box")
        prompt_name = f"prompt_AI_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.sidebar.download_button("Scarica prompt",
                                   data=st.session_state["ai_prompt_text"].encode("utf-8"),
                                   file_name=prompt_name,
                                   key="dl_prompt")
    else:
        st.sidebar.caption("Il prompt verrà generato qui dopo la trascrizione.")

    return model_size, language


# ----------------------------- App ----------------------------- #

def main():
    st.set_page_config(page_title="Trascrivi — Whisper", page_icon="🎙️", layout="wide")
    _ensure_session_dir()

    # Sidebar (mostrata sempre, e aggiornata ad ogni rerun)
    model_size, language = _sidebar_settings()

    _header()

    # Messaggio post-rerun quando è stato salvato un file
    if st.session_state.get("just_saved_label"):
        st.success(f"File salvato: **{st.session_state.just_saved_label}** (vedi sidebar).")
        st.session_state.pop("just_saved_label", None)

    with st.expander("📌 Guida rapida", expanded=False):
        st.markdown(
            """
**1. Carica un file audio.** Formati: `mp3, wav, m4a, aac, flac, ogg, webm, wma`.

**2. Avvia la trascrizione.** Se l'audio è lungo, verrà suddiviso in blocchi **~10–12 min** con **overlap 10s**.

**3. Scarica il .txt.** I file restano disponibili finché **la sessione resta aperta**.

**4. Prompt AI.** Dopo la trascrizione, nella **sidebar** trovi un **prompt precompilato** col testo già inserito.
"""
        )

    with st.expander("📖 Apri README.md"):
        readme_path = os.path.join(os.getcwd(), "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("README.md non trovato nel repository.")

    st.markdown("### 1) Carica il file audio")
    uploaded = st.file_uploader(
        "Drag and drop / Browse",
        type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "wma", "webm"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

    audio_path = None
    duration_s = None
    if uploaded is not None:
        audio_path = os.path.join(st.session_state.session_dir, _sanitize_filename(uploaded.name))
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())
        try:
            duration_s = _probe_duration_seconds(audio_path)
            chunks, n_chunks, avg_min = _plan_chunks(duration_s)
            st.success(
                f"L'audio durerà **{round(duration_s/60.0, 1)} min**. "
                f"Verrà elaborato in **{n_chunks} blocchi** da circa **{avg_min:.1f} min** (overlap 10s)."
            )
        except Exception as e:
            st.error(f"Impossibile leggere i metadata (ffprobe): {e}")
            return

    st.markdown("### 2) Avvia la trascrizione")
    start_btn = st.button("🔴 Avvia", type="primary", disabled=(uploaded is None))

    # Mostra eventualmente l'ultima trascrizione anche dopo un rerun
    if st.session_state.get("transcript_text"):
        st.markdown("### Ultima trascrizione")
        st.text_area("Trascrizione completa (.txt)",
                     value=st.session_state.transcript_text, height=260, key="latest_txt")

    if start_btn and uploaded is not None:
        _run_transcription(audio_path, uploaded.name, duration_s, model_size, language)


def _run_transcription(audio_path: str, display_name: str, duration_s: float,
                       model_size: str, language: str):
    st.info(f"Modello: **{model_size}** · Lingua: **{language}** · File: **{display_name}**")

    chunks, n_chunks, avg_min = _plan_chunks(duration_s)

    st.write("### 3) Transcodifica e trascrizione")
    st.caption("ETA = stima residua per l'intero job; si aggiorna mentre elabora i blocchi.")
    progress = st.progress(0.0)
    status = st.empty()
    job_start = time.time()

    audio_done_s = 0.0
    proc_do_

