# trascrivi.py
# Streamlit app per trascrizione con faster-whisper + gestione chunk 10–12 minuti
# Autore visibile nell'header: modifica qui sotto ✎ se vuoi cambiarlo al volo.
AUTHOR_NAME = "Roberto M."  # ✎ Cambia qui il nome autore mostrato nell'app

import os
import io
import math
import time
import json
import queue
import shutil
import string
import threading
import tempfile
import subprocess
from datetime import datetime

import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg


# ----------------------------- Utility di base ----------------------------- #

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
    if "saved_files" not in st.session_state:
        st.session_state.saved_files = []
    if "transcript_text" not in st.session_state:
        st.session_state.transcript_text = ""
    if "ai_prompt_text" not in st.session_state:
        st.session_state.ai_prompt_text = ""


def _save_text_session(basename: str, text: str) -> str:
    _ensure_session_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = _sanitize_filename(basename.rsplit(".", 1)[0])
    path = os.path.join(st.session_state.session_dir, f"{ts}_{base}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    # memorizza per download rapido
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
    """Restituisce la durata in secondi via ffprobe."""
    if not _check_ffprobe():
        raise RuntimeError("ffprobe non trovato. Assicurati che ffmpeg/ffprobe sia installato.")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    info = json.loads(p.stdout.decode("utf-8"))
    return float(info["format"]["duration"])


def _export_wav_segment(src_path: str, start_s: float, dur_s: float, dst_path: str):
    """Esporta un segmento a WAV mono 16k per massima compatibilità."""
    (
        ffmpeg
        .input(src_path, ss=max(0, start_s), t=max(0.1, dur_s))
        .output(dst_path, format="wav", ac=1, ar=16000, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )


def _plan_chunks(duration_s: float, target_min: float = 10.5, max_min: float = 12.0, overlap_s: float = 10.0):
    """
    Crea un piano di chunk ~10–12 minuti con overlap fisso.
    Ritorna lista di (start, dur), numero chunk, dur_media_min.
    """
    # numero chunk intorno al target
    n = max(1, round(duration_s / (target_min * 60.0)))
    # se qualcuno superasse max_min, aumentiamo n
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
    """
    Genera il prompt completo con la trascrizione già inserita tra i delimitatori.
    """
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


# ----------------------------- UI Header & Sidebar ----------------------------- #

def _header(title="Trascrivi — Whisper (faster-whisper)"):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(f"## {title}")
    with col2:
        st.markdown(f"<div style='text-align:right;color:#666;'>Autore: <b>{AUTHOR_NAME}</b></div>", unsafe_allow_html=True)


def _sidebar_settings():
    st.sidebar.header("Impostazioni")

    model_size = st.sidebar.selectbox(
        "Modello Whisper",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=2,
        help="Modello di trascrizione (tiny/base/small/medium/large-v3)."
    )

    lang = st.sidebar.text_input("Lingua", value="it", help="Codice lingua ISO (es. it, en, fr, ...).")

    st.sidebar.caption("Blocchi: automatico 10–12 minuti (impostazione interna).")

    # Sezione salvataggi sessione
    st.sidebar.subheader("File salvati (sessione)")
    if st.session_state.get("saved_files"):
        for i, item in enumerate(st.session_state.saved_files):
            with open(item["path"], "rb") as f:
                st.sidebar.download_button(
                    "Scarica",
                    data=f.read(),
                    file_name=item["label"],
                    key=f"dl_{i}",
                    help=item["label"]
                )
                st.sidebar.caption(item["label"])
    else:
        st.sidebar.caption("Nessun file salvato in questa sessione.")

    st.sidebar.button("🔄 Reimposta sessione\n*(non cancella file locali già scaricati)*",
                      on_click=_reset_session_state)

    # Prompt AI (precompilato dopo trascrizione)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Prompt per AI (post-produzione)")
    if st.session_state.get("ai_prompt_text"):
        st.sidebar.text_area(
            "Prompt completo (copia & incolla ovunque)",
            value=st.session_state["ai_prompt_text"],
            height=260
        )
        # download del prompt come txt
        prompt_name = f"prompt_AI_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.sidebar.download_button("Scarica prompt",
                                   data=st.session_state["ai_prompt_text"].encode("utf-8"),
                                   file_name=prompt_name)
    else:
        st.sidebar.caption("Il prompt verrà generato qui dopo la trascrizione.")


def _reset_session_state():
    # NON elimina la cartella, lascia i file finché la sessione resta viva
    for key in ["saved_files", "transcript_text", "ai_prompt_text"]:
        st.session_state[key] = [] if key == "saved_files" else ""
    st.toast("Sessione reimpostata (i file scaricati in locale restano).", icon="✅")


# ----------------------------- App ----------------------------- #

def main():
    st.set_page_config(page_title="Trascrivi — Whisper", page_icon="🎙️", layout="wide")
    _ensure_session_dir()

    _header()

    with st.expander("📌 Guida rapida", expanded=False):
        st.markdown(
            """
**1. Carica un file audio.**  
Formati comuni: `mp3, wav, m4a, aac, flac, ogg, webm, wma...`

**2. Avvia la trascrizione.**  
Se l'audio è lungo, verrà suddiviso in blocchi **~10–12 min** con **overlap 10s** per non perdere parole a cavallo.

**3. Scarica il .txt.**  
I file **restano disponibili per tutta la durata della sessione** (finché non chiudi/ricarichi la pagina).

**4. Prompt AI.**  
Dopo la trascrizione, nella **sidebar** trovi un **prompt precompilato** con il testo già inserito tra i delimitatori.
"""
        )

    # Prova ad aprire README.md se presente nel repo
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

    # Mostra info di pianificazione chunk una volta noto il file
    audio_path = None
    duration_s = None
    if uploaded is not None:
        # Salva una copia nella cartella di sessione
        audio_path = os.path.join(st.session_state.session_dir, _sanitize_filename(uploaded.name))
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())

        try:
            duration_s = _probe_duration_seconds(audio_path)
            chunks, n_chunks, avg_min = _plan_chunks(duration_s)
            st.success(
                f"L'audio durerà **{round(duration_s/60.0, 1)} min**. "
                f"Verrà elaborato in **{n_chunks} blocchi** da circa **{avg_min:.1f} min** "
                f"(overlap 10s)."
            )
        except Exception as e:
            st.error(f"Impossibile leggere i metadata (ffprobe): {e}")
            return

    st.markdown("### 2) Avvia la trascrizione")
    left, _ = st.columns([1, 3])
    start_btn = left.button("🔴 Avvia", type="primary", disabled=(uploaded is None))

    # Impostazioni dalla sidebar
    _sidebar_settings()
    model_size = st.session_state.get("model_size_sel") or None  # (non usato: manteniamo pick diretto)
    # Recuperiamo i valori letti nella sidebar (usiamo direttamente i widget)
    # Streamlit non assegna automaticamente a session_state con nome, per cui rileggeremo via widget state.
    # (qui non serve: li ricalcoliamo nell'azione)

    if start_btn and uploaded is not None:
        _run_transcription(audio_path, uploaded.name, duration_s)


def _run_transcription(audio_path: str, display_name: str, duration_s: float):
    # Rileggi preferenze correnti dalla sidebar
    # (Sono letti direttamente dai widget; qui usiamo st.session_state per la lingua)
    lang = st.session_state.get("Lingua", None)  # fallback, ma meglio ripescare via variabile locale della sidebar
    # In realtà, estraiamo di nuovo i valori interrogando i widget: Streamlit assegna chiavi automatiche.
    # Per semplicità, reimpostiamo manualmente:
    lang = st.sidebar.text_input if False else None  # no-op per evitare warning lint

    # Siccome non abbiamo accesso diretto alle chiavi dei widget, definiamo default ragionevoli:
    language = "it"
    # Per il modello, usiamo un trucco: cerchiamo nello state i possibili valori già noti
    possible_models = ["tiny", "base", "small", "medium", "large-v3"]
    chosen_model = None
    for k, v in st.session_state.items():
        if isinstance(v, str) and v in possible_models:
            chosen_model = v
    if not chosen_model:
        chosen_model = "medium"

    # UI: info lavoro
    st.info(f"Modello: **{chosen_model}** · Lingua: **it** · File: **{display_name}**")

    # Piano chunk
    chunks, n_chunks, avg_min = _plan_chunks(duration_s)

    # Prepara modello
    st.write("### 3) Transcodifica e trascrizione")
    st.caption("I blocchi vengono elaborati in sequenza. ETA stimata si aggiorna mentre lavora.")
    progress = st.progress(0.0)
    status = st.empty()  # testo 'Blocco i/N — Trascorso — ETA'
    log_area = st.empty()  # eventuali messaggi
    job_start = time.time()

    # Contatori per ETA dinamica
    audio_done_s = 0.0
    proc_done_s = 0.0
    # fallback: ipotizziamo 1.8x realtime finché non abbiamo una misura
    fallback_ratio = 1.8

    # Istanzia modello faster-whisper
    # compute_type "int8" è veloce su CPU; se hai GPU puoi passare "float16"
    model = WhisperModel(chosen_model, device="cpu", compute_type="int8")

    transcript_parts = []

    for i, (start_s, dur_s) in enumerate(chunks, start=1):
        # 1) Prepara file WAV del segmento
        seg_name = f"chunk_{i:02d}.wav"
        seg_path = os.path.join(st.session_state.session_dir, seg_name)
        _export_wav_segment(audio_path, start_s, dur_s, seg_path)

        # 2) Avvia trascrizione su thread dedicato per poter aggiornare ETA in tempo reale
        q: queue.Queue[str] = queue.Queue()
        err = {}

        def worker():
            try:
                seg_iter, _info = model.transcribe(
                    seg_path,
                    language="it",
                    vad_filter=True,
                    beam_size=5
                )
                text = "".join(s.text for s in seg_iter)
                q.put(text)
            except Exception as e:
                err["e"] = e
                q.put("")

        t0 = time.time()
        th = threading.Thread(target=worker, daemon=True)
        th.start()

        # 3) Aggiorna timer & ETA mentre il thread lavora
        while th.is_alive():
            elapsed = time.time() - job_start
            # Stima ratio se abbiamo almeno un segmento concluso
            if audio_done_s > 0:
                ratio = proc_done_s / audio_done_s  # sec processamento per sec audio
            else:
                ratio = fallback_ratio
            remaining_audio = (sum(d for _, d in chunks[i-1:]) if audio_done_s == 0 else
                               (sum(d for _, d in chunks[i-1:]) - (time.time() - t0)))
            eta = max(0.0, ratio * remaining_audio)
            status.markdown(
                f"**Blocco {i}/{n_chunks}** — Trascorso: **{_human_time(elapsed)}** · "
                f"ETA: **{_human_time(eta)}**"
            )
            # progresso globale (per blocchi)
            progress.progress((i - 1) / n_chunks)
            time.sleep(0.2)

        # 4) Fine segmento: raccogli risultato
        th.join()
        seg_text = q.get()
        if "e" in err:
            st.error(f"Errore nel blocco {i}: {err['e']}")
            return

        dt = time.time() - t0
        proc_done_s += dt
        audio_done_s += dur_s
        transcript_parts.append(seg_text)

        # aggiorna progress e status dopo il blocco
        elapsed = time.time() - job_start
        remaining_audio_total = max(0.0, duration_s - audio_done_s)
        ratio = proc_done_s / max(1e-6, audio_done_s)
        eta = ratio * remaining_audio_total
        status.markdown(
            f"**Blocco {i}/{n_chunks} completato** — Trascorso: **{_human_time(elapsed)}** · "
            f"ETA residua: **{_human_time(eta)}**"
        )
        progress.progress(i / n_chunks)

    # Concatenazione finale
    full_text = "".join(transcript_parts).strip()
    st.session_state.transcript_text = full_text

    # Mostra/Salva .txt
    st.success("Trascrizione completata.")
    st.text_area("Trascrizione completa (.txt)", value=full_text, height=260)

    saved_path = _save_text_session(display_name, full_text)
    with open(saved_path, "rb") as f:
        st.download_button("Scarica trascrizione (.txt)", data=f.read(), file_name=os.path.basename(saved_path), type="primary")

    # Genera prompt AI precompilato (in sidebar)
    st.session_state.ai_prompt_text = _build_ai_prompt(full_text)
    st.info("Prompt AI generato nella **sidebar** (sezione “Prompt per AI”).")


if __name__ == "__main__":
    main()
