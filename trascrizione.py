# trascrivi.py — versione stabile
# Autore mostrato in alto a destra:
AUTHOR_NAME = "Roberto M."  # <- cambia qui se serve

import os
import io
import math
import time
import json
import shutil
import string
import tempfile
import subprocess
from datetime import datetime

import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg  # ffmpeg-python

# -------------------------- util --------------------------

def human_time(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def sanitize_filename(name: str) -> str:
    valid = f"-_.() {string.ascii_letters}{string.digits}"
    clean = "".join(c if c in valid else "_" for c in name).strip(" ._")
    return clean or "file"

def ensure_session_dir():
    if "session_dir" not in st.session_state:
        st.session_state.session_dir = tempfile.mkdtemp(prefix="trascrivi_")
    st.session_state.setdefault("saved_files", [])
    st.session_state.setdefault("transcript_text", "")
    st.session_state.setdefault("ai_prompt_text", "")

def save_text_session(basename: str, text: str) -> str:
    ensure_session_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = sanitize_filename(basename.rsplit(".", 1)[0])
    path = os.path.join(st.session_state.session_dir, f"{ts}_{base}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    st.session_state.saved_files.append({"path": path, "label": os.path.basename(path)})
    return path

def check_ffprobe() -> bool:
    try:
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def probe_duration_seconds(path: str) -> float:
    if not check_ffprobe():
        raise RuntimeError("ffprobe non trovato. Installa ffmpeg/ffprobe nel sistema.")
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    info = json.loads(p.stdout.decode("utf-8"))
    return float(info["format"]["duration"])

def export_wav_segment(src_path: str, start_s: float, dur_s: float, dst_path: str):
    (
        ffmpeg.input(src_path, ss=max(0, start_s), t=max(0.1, dur_s))
              .output(dst_path, format="wav", ac=1, ar=16000, acodec="pcm_s16le")
              .overwrite_output()
              .run(quiet=True)
    )

def plan_chunks(duration_s: float, target_min: float = 10.5, max_min: float = 12.0, overlap_s: float = 10.0):
    """Restituisce lista [(start, dur)], numero blocchi, media minuti/blocco."""
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

def build_ai_prompt(transcript_text: str) -> str:
    t = transcript_text.strip()
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
        f"{t}\n"
        "§§§§§§\n"
    )

# -------------------------- UI helpers --------------------------

def header(title="Trascrivi — Whisper (faster-whisper)"):
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.markdown(f"## {title}")
    with c2:
        st.markdown(
            f"<div style='text-align:right;color:#666;'>Autore: <b>{AUTHOR_NAME}</b></div>",
            unsafe_allow_html=True
        )

def reset_session_state():
    st.session_state["saved_files"] = []
    st.session_state["transcript_text"] = ""
    st.session_state["ai_prompt_text"] = ""
    st.toast("Sessione reimpostata. I file già scaricati in locale restano.", icon="✅")

def sidebar(model_default="medium", lang_default="it"):
    st.sidebar.header("Impostazioni")
    model_size = st.sidebar.selectbox("Modello Whisper",
                                      ["tiny", "base", "small", "medium", "large-v3"],
                                      index=["tiny","base","small","medium","large-v3"].index(model_default))
    language = st.sidebar.text_input("Lingua", value=lang_default)
    st.sidebar.caption("Blocchi: automatico 10–12 minuti, overlap 10s.")

    st.sidebar.subheader("File salvati (sessione)")
    if st.session_state.get("saved_files"):
        for item in st.session_state.saved_files:
            with open(item["path"], "rb") as f:
                st.sidebar.download_button("Scarica", f.read(), file_name=item["label"], key=f"dl_{item['label']}")
            st.sidebar.caption(item["label"])
    else:
        st.sidebar.caption("Nessun file salvato in questa sessione.")

    st.sidebar.button("🔄 Reimposta sessione\n*(non cancella file locali già scaricati)*",
                      on_click=reset_session_state)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Prompt per AI (post-produzione)")
    if st.session_state.get("ai_prompt_text"):
        st.sidebar.text_area("Prompt completo (copia & incolla)", value=st.session_state["ai_prompt_text"],
                             height=260, key="ai_prompt_box")
        st.sidebar.download_button("Scarica prompt",
                                   data=st.session_state["ai_prompt_text"].encode("utf-8"),
                                   file_name=f"prompt_AI_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    else:
        st.sidebar.caption("Il prompt apparirà qui dopo la trascrizione.")

    return model_size, language

# -------------------------- app --------------------------

def main():
    st.set_page_config(page_title="Trascrivi — Whisper", page_icon="🎙️", layout="wide")
    ensure_session_dir()

    # Sidebar (lettura impostazioni)
    model_size, language = sidebar()

    header()

    with st.expander("📌 Guida rapida", expanded=False):
        st.markdown(
            """
**1. Carica un file audio.** Formati: `mp3, wav, m4a, aac, flac, ogg, webm, wma`.

**2. Avvia la trascrizione.** Se l'audio è lungo, viene suddiviso in blocchi **~10–12 min** (overlap 10s).

**3. Scarica il .txt.** I file restano disponibili finché **la sessione del browser resta aperta**.

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
    uploaded = st.file_uploader("Drag and drop / Browse",
                                type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "wma", "webm"],
                                accept_multiple_files=False, label_visibility="collapsed")

    audio_path = None
    duration_s = None
    if uploaded is not None:
        audio_path = os.path.join(st.session_state.session_dir, sanitize_filename(uploaded.name))
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())
        try:
            duration_s = probe_duration_seconds(audio_path)
            chunks, n_chunks, avg_min = plan_chunks(duration_s)
            st.success(
                f"L'audio dura **{round(duration_s/60.0,1)} min**. "
                f"Sarà elaborato in **{n_chunks} blocchi** da circa **{avg_min:.1f} min** (overlap 10s)."
            )
        except Exception as e:
            st.error(f"Impossibile leggere i metadata (ffprobe): {e}")
            return

    st.markdown("### 2) Avvia la trascrizione")
    start_btn = st.button("🔴 Avvia", type="primary", disabled=(uploaded is None))

    # Se già esiste una trascrizione in sessione, mostro
    if st.session_state.get("transcript_text"):
        st.markdown("### Ultima trascrizione")
        st.text_area("Trascrizione completa (.txt)",
                     value=st.session_state["transcript_text"], height=260, key="latest_txt")

    if start_btn and uploaded is not None:
        run_transcription(audio_path, uploaded.name, duration_s, model_size, language)

def run_transcription(audio_path: str, display_name: str, duration_s: float,
                      model_size: str, language: str):
    st.info(f"Modello: **{model_size}** · Lingua: **{language}** · File: **{display_name}**")

    chunks, n_chunks, _ = plan_chunks(duration_s)

    st.write("### 3) Transcodifica e trascrizione")
    st.caption("La barra mostra l'avanzamento a blocchi; ETA è la stima residua per l'intero job.")
    progress = st.progress(0.0)
    status = st.empty()
    job_start = time.time()

    # Stima basata sui blocchi già fatti
    audio_done_s = 0.0
    proc_done_s = 0.0
    fallback_ratio = 1.8  # finché non ho misure reali

    # Modello
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    parts = []
    for i, (start_s, dur_s) in enumerate(chunks, start=1):
        # Stima ETA PRIMA del blocco
        ratio = (proc_done_s / audio_done_s) if audio_done_s > 0 else fallback_ratio
        remaining_audio = max(0.0, duration_s - audio_done_s)
        eta_before = max(0.0, ratio * remaining_audio)
        elapsed = time.time() - job_start
        status.markdown(
            f"**Blocco {i}/{n_chunks}** — Trascorso: **{human_time(elapsed)}** · "
            f"ETA residua: **{human_time(eta_before)}**"
        )
        progress.progress((i - 1) / n_chunks)

        # Esporta segmento e trascrivi
        seg_path = os.path.join(st.session_state.session_dir, f"chunk_{i:02d}.wav")
        export_wav_segment(audio_path, start_s, dur_s, seg_path)

        t0 = time.time()
        seg_iter, _info = model.transcribe(seg_path, language=language, vad_filter=True, beam_size=5)
        seg_text = "".join(s.text for s in seg_iter)
        dt = time.time() - t0

        parts.append(seg_text)
        audio_done_s += dur_s
        proc_done_s += dt

        # Stima ETA DOPO il blocco
        ratio = proc_done_s / max(1e-6, audio_done_s)
        remaining_audio = max(0.0, duration_s - audio_done_s)
        eta_after = ratio * remaining_audio
        elapsed = time.time() - job_start
        status.markdown(
            f"**Blocco {i}/{n_chunks} completato** — Trascorso: **{human_time(elapsed)}** · "
            f"ETA residua: **{human_time(eta_after)}**"
        )
        progress.progress(i / n_chunks)

    full_text = "".join(parts).strip()
    st.session_state["transcript_text"] = full_text

    st.success("Trascrizione completata.")
    st.text_area("Trascrizione completa (.txt)", value=full_text, height=260, key="final_txt")

    # Salva .txt in sessione
    saved_path = save_text_session(display_name, full_text)
    st.info(f"File salvato nella sessione: **{os.path.basename(saved_path)}** (vedi sidebar).")

    # Prompt AI per la sidebar
    st.session_state["ai_prompt_text"] = build_ai_prompt(full_text)


if __name__ == "__main__":
    main()
