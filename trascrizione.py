# trascrivi.py
# Streamlit app per trascrizione audio con faster-whisper
# Funzioni chiave:
# - Suddivisione automatica in blocchi 10–12 minuti
# - Barra di avanzamento per ogni blocco (tempo totale → residuo stimato)
# - Salvataggio incrementale e persistenza .txt per tutta la sessione
# - Guida rapida + lettura README.md
# - Prompt di post-produzione SOLO in sidebar (persistente in sessione)

from __future__ import annotations

import os
import re
import time
import json
import threading
import tempfile
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg


# ============ CONFIGURAZIONE APP ============
st.set_page_config(page_title="Trascrivi audio", page_icon="🎧", layout="wide")

# [MODIFICA] Autore visibile in app (cerca questa riga per cambiarlo rapidamente)
AUTHOR_DISPLAY_NAME = "Roberto M."   # << Cambia qui il nome autore

APP_TITLE = "Trascrivi — Whisper (faster-whisper)"
DEFAULT_LANGUAGE = "it"
DEFAULT_MODEL = "tiny"  # tiny/base/small/medium/large-v3 ecc.
# Overlap tra blocchi per non troncare frasi
CHUNK_OVERLAP_SECONDS = 10.0

# Cartella per-sessione e indice file salvati
SESSION_DIR_KEY = "workdir"
TRANSCRIPTS_INDEX_KEY = "saved_transcripts"
LAST_PROMPT_KEY = "last_prompt"   # prompt completo in sessione


# ============ UTILS ============
def _ensure_session_dir() -> str:
    if SESSION_DIR_KEY not in st.session_state:
        sess_dir = tempfile.mkdtemp(prefix="trascrivi_")
        st.session_state[SESSION_DIR_KEY] = sess_dir
    return st.session_state[SESSION_DIR_KEY]


def _safe_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE)
    return name[:120]


def _write_bytes(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _append_text(path: str, text: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _probe_duration_seconds(path: str) -> float:
    """Rileva durata audio in secondi usando ffprobe."""
    meta = ffmpeg.probe(path)
    dur = None
    for s in meta.get("streams", []):
        if s.get("codec_type") == "audio" and "duration" in s:
            dur = float(s["duration"])
            break
    if dur is None:
        dur = float(meta["format"]["duration"])
    return dur


def _compute_chunk_plan(total_sec: float) -> List[Tuple[float, float]]:
    """
    Regola: durata_blocco = clamp(total/8, 10min, 12min).
    Ritorna lista di (start, durata) in secondi, con overlap fisso.
    """
    min_chunk = 10 * 60.0
    max_chunk = 12 * 60.0
    target = max(min_chunk, min(max_chunk, total_sec / 8.0))
    step = target - CHUNK_OVERLAP_SECONDS
    plan = []
    start = 0.0
    while start < total_sec:
        end = min(start + target, total_sec)
        dur = max(0.0, end - start)
        if dur > 0.5:
            plan.append((start, dur))
        if end >= total_sec:
            break
        start = start + step
    return plan


def _slice_to_wav(src_path: str, dst_path: str, start_sec: float, dur_sec: float):
    """Crea un WAV mono 16 kHz della finestra richiesta."""
    (
        ffmpeg
        .input(src_path, ss=start_sec, t=dur_sec)
        .output(dst_path, ac=1, ar=16000, format="wav", y="-y")
        .overwrite_output()
        .run(quiet=True)
    )


def _estimate_eta_text(total_s: float, elapsed_s: float) -> str:
    remain = max(0.0, total_s - elapsed_s)

    def fmt(s: float) -> str:
        m, s = divmod(int(round(s)), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    return f"Totale stimato: {fmt(total_s)} • Residuo: {fmt(remain)}"


def _collect_segments_text(segments) -> str:
    parts = []
    for seg in segments:
        parts.append(seg.text)
    return "".join(parts)


def _make_prompt_with_text(transcript: str) -> str:
    return (
        "SEI UN EDITOR PROFESSIONISTA.\n\n"
        "Migliora il testo seguente senza alterare i contenuti: correggi refusi,\n"
        "punteggiatura, ripetizioni, sintassi, lessico datato; applica una\n"
        "formattazione leggibile con paragrafi e titoli minimi se opportuno.\n"
        "Mantieni lo stile orale, ma rendilo scorrevole. Lingua: it.\n"
        "<< TESTO DA INSERIRE >>\n"
        "Output richiesto: testo finale pulito, pronto per la pubblicazione.\n"
        "§§§§§§\n"
        "TESTO DA MIGLIORARE (tra i delimitatori):\n"
        "[---INIZIO TESTO---]\n"
        f"{transcript}\n"
        "[---FINE TESTO---]\n"
    )


def _init_transcripts_index():
    if TRANSCRIPTS_INDEX_KEY not in st.session_state:
        st.session_state[TRANSCRIPTS_INDEX_KEY] = {}  # {label: path}


# ============ HEADER ============
col_title, col_author = st.columns([1, 0.25])
with col_title:
    st.title(APP_TITLE)
with col_author:
    st.markdown(
        f"<div style='text-align:right; opacity:0.9;'>Autore: <b>{AUTHOR_DISPLAY_NAME}</b></div>",
        unsafe_allow_html=True,
    )

with st.expander("📌 Guida rapida", expanded=True):
    st.markdown(
        """
- Carica un file audio e avvia la trascrizione.
- L'audio viene diviso **automaticamente** in blocchi da **10–12 min** con una piccola sovrapposizione.
- Per **ogni blocco** vedi una barra di avanzamento con **stima del tempo residuo**.
- Il `.txt` viene **salvato dopo ogni blocco** e rimane disponibile finché **la sessione resta aperta**.
- Il **prompt per AI** comparirà **in sidebar** con dentro **la trascrizione** già inserita.
        """
    )

# README (se presente)
readme_text = None
try:
    if os.path.exists("README.md"):
        readme_text = _read_text("README.md")
except Exception:
    readme_text = None

with st.expander("📖 Apri README.md", expanded=False):
    if readme_text:
        st.markdown(readme_text)
    else:
        st.info("Nessun README.md trovato nel repository.")

st.divider()


# ============ SIDEBAR ============
with st.sidebar:
    st.subheader("Impostazioni")
    model_size = st.selectbox(
        "Modello Whisper",
        options=["tiny", "base", "small", "medium", "large-v3"],
        index=0,
        help="Usa 'tiny' per test veloci; modelli maggiori = più qualità, più lenti."
    )
    language = st.text_input("Lingua", value=DEFAULT_LANGUAGE, help="Codice lingua (es. it, en, fr).")
    st.caption("Blocchi: automatico 10–12 minuti (impostazione interna).")

    st.markdown("---")
    st.markdown("### File salvati (sessione)")
    _init_transcripts_index()
    if st.session_state[TRANSCRIPTS_INDEX_KEY]:
        for label, path in list(st.session_state[TRANSCRIPTS_INDEX_KEY].items()):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = f.read()
                st.download_button(
                    label=f"⬇️ Scarica {label}",
                    data=data,
                    file_name=os.path.basename(path),
                    mime="text/plain",
                    key=f"dl_{label}"
                )
            except Exception:
                st.caption(f"(Non riesco a leggere {label})")
    else:
        st.caption("Nessun file salvato in questa sessione.")

    # Prompt per AI (persistente in sessione)
    st.markdown("---")
    st.markdown("### Prompt per AI")
    prompt_sidebar_val = st.session_state.get(LAST_PROMPT_KEY, "")
    if prompt_sidebar_val:
        st.text_area(
            "Prompt completo (con testo inserito)",
            value=prompt_sidebar_val,
            height=260,
            key="prompt_sidebar",
        )
        st.download_button(
            "⬇️ Scarica prompt_AI.txt",
            data=prompt_sidebar_val,
            file_name="prompt_AI.txt",
            mime="text/plain",
            key="dl_prompt",
        )
    else:
        st.caption("Il prompt apparirà qui al termine (o in caso di salvataggi parziali).")

    st.markdown("---")
    if st.button("🔁 Reimposta sessione (non cancella file locali già scaricati)"):
        for k in [TRANSCRIPTS_INDEX_KEY, SESSION_DIR_KEY, LAST_PROMPT_KEY]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()


# ============ UPLOAD ============
st.subheader("1) Carica il file audio")
uploaded = st.file_uploader(
    "Formati comuni supportati (mp3, wav, m4a, webm, ...)",
    type=["mp3", "wav", "m4a", "aac", "flac", "ogg", "webm"],
    accept_multiple_files=False,
)

workdir = _ensure_session_dir()
_init_transcripts_index()

if uploaded is not None:
    # Salva l'upload nella cartella di sessione
    base = _safe_filename(uploaded.name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_audio_path = os.path.join(workdir, f"{ts}_{base}")
    _write_bytes(src_audio_path, uploaded.getvalue())

    # Info durata
    try:
        total_sec = _probe_duration_seconds(src_audio_path)
    except Exception as e:
        st.error(f"Impossibile leggere la durata dell'audio: {e}")
        st.stop()

    chunks = _compute_chunk_plan(total_sec)
    n_chunks = len(chunks)
    avg_chunk_min = sum(d for _, d in chunks) / n_chunks / 60.0 if n_chunks else 0
    st.success(
        f"L'audio durerà **{total_sec/60:.1f} min**. "
        f"Verrà elaborato in **{n_chunks} blocchi** da circa **{avg_chunk_min:.1f} min** (overlap {int(CHUNK_OVERLAP_SECONDS)}s)."
    )

    st.subheader("2) Avvia la trascrizione")
    start_btn = st.button("🚀 Avvia", type="primary")

    if start_btn:
        # Preparazione modello
        st.info("Inizializzo il modello, attendi…")
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as e:
            st.error(f"Errore inizializzazione modello: {e}")
            st.stop()

        # File di output (persistente nella sessione)
        out_txt_name = os.path.splitext(os.path.basename(src_audio_path))[0] + ".txt"
        out_txt_path = os.path.join(workdir, out_txt_name)

        # Se esiste già, ruota per non sovrascrivere
        rot = 1
        final_out = out_txt_path
        while os.path.exists(final_out):
            final_out = os.path.join(workdir, f"{os.path.splitext(out_txt_name)[0]}_{rot}.txt")
            rot += 1
        out_txt_path = final_out

        st.session_state[TRANSCRIPTS_INDEX_KEY][os.path.basename(out_txt_path)] = out_txt_path

        overall_box = st.container()
        overall_bar = overall_box.progress(0.0, text="Avanzamento complessivo…")
        overall_eta = overall_box.empty()

        rolling_rtf_inv: List[float] = []
        accumulated_text = []

        for i, (start_sec, dur_sec) in enumerate(chunks, start=1):
            st.markdown(f"#### Blocco {i}/{n_chunks}")
            chunk_box = st.container()
            chunk_bar = chunk_box.progress(0.0)
            chunk_eta = chunk_box.empty()
            chunk_log = chunk_box.empty()

            # Prepara WAV del blocco
            chunk_wav = os.path.join(workdir, f"chunk_{i:02d}.wav")
            try:
                _slice_to_wav(src_audio_path, chunk_wav, start_sec, dur_sec)
            except Exception as e:
                st.error(f"Errore nel pre-processing del blocco {i}: {e}")
                break

            # Stima tempo blocco
            if rolling_rtf_inv:
                avg_inv = sum(rolling_rtf_inv) / len(rolling_rtf_inv)
                expected_total_s = max(3.0, dur_sec * avg_inv)
            else:
                expected_total_s = max(3.0, dur_sec * 1.7)

            # Transcribe in thread
            done_evt = threading.Event()
            result_holder = {"text": "", "elapsed": 0.0, "error": None}

            def _worker():
                t0 = time.time()
                try:
                    segments, info = model.transcribe(
                        chunk_wav,
                        language=language,
                        beam_size=1,
                        vad_filter=True,
                    )
                    txt = _collect_segments_text(segments)
                    result_holder["text"] = txt
                except Exception as ex:
                    result_holder["error"] = ex
                finally:
                    result_holder["elapsed"] = time.time() - t0
                    done_evt.set()

            th = threading.Thread(target=_worker, daemon=True)
            th.start()

            # Aggiorna barra
            t_loop0 = time.time()
            while not done_evt.is_set():
                elapsed = time.time() - t_loop0
                frac = min(0.95, elapsed / expected_total_s)
                chunk_bar.progress(frac)
                chunk_eta.markdown(_estimate_eta_text(expected_total_s, elapsed))
                time.sleep(0.2)

            # Concluso il blocco
            if result_holder["error"] is not None:
                st.error(f"Errore nel blocco {i}: {result_holder['error']}")
                if accumulated_text:
                    try:
                        _append_text(out_txt_path, "".join(accumulated_text))
                        # aggiorna prompt parziale in sidebar
                        try:
                            partial = _read_text(out_txt_path)
                            st.session_state[LAST_PROMPT_KEY] = _make_prompt_with_text(partial)
                        except Exception:
                            pass
                    except Exception:
                        pass
                break

            chunk_bar.progress(1.0)
            chunk_eta.markdown(_estimate_eta_text(result_holder["elapsed"], result_holder["elapsed"]))

            # Aggiorna statistiche e salvataggio incrementale
            elapsed = result_holder["elapsed"]
            if dur_sec > 0.2:
                rolling_rtf_inv.append(elapsed / dur_sec)

            accumulated_text.append(result_holder["text"])
            try:
                _append_text(out_txt_path, result_holder["text"])
                # aggiorna prompt parziale in sidebar dopo ogni blocco
                try:
                    partial = _read_text(out_txt_path)
                    st.session_state[LAST_PROMPT_KEY] = _make_prompt_with_text(partial)
                except Exception:
                    pass
            except Exception as e:
                chunk_log.warning(f"Avviso: impossibile salvare incrementale (blocco {i}): {e}")

            # Barra complessiva / ETA job
            overall_frac = i / n_chunks
            if rolling_rtf_inv:
                avg_inv = sum(rolling_rtf_inv) / len(rolling_rtf_inv)
                remaining_audio_s = sum(d for _, d in chunks[i:])  # i è 1-based
                remaining_calc_s = remaining_audio_s * avg_inv
            else:
                remaining_calc_s = 0.0
            overall_bar.progress(overall_frac, text=f"Avanzamento complessivo: {int(overall_frac*100)}%")
            overall_eta.markdown(f"⏱️ Stima tempo residuo job: **{int(remaining_calc_s//60)} min {int(remaining_calc_s%60)} s**")

        # Fine loop blocchi
        if os.path.exists(out_txt_path):
            st.success("✅ Trascrizione completata (o salvata parzialmente in caso di errori).")
            final_text = _read_text(out_txt_path)

            st.subheader("3) Scarica la trascrizione (.txt)")
            st.download_button(
                label="⬇️ Scarica trascrizione",
                data=final_text,
                file_name=os.path.basename(out_txt_path),
                mime="text/plain",
                type="primary",
            )

            # Prepara prompt definitivo SOLO per la sidebar
            st.session_state[LAST_PROMPT_KEY] = _make_prompt_with_text(final_text)
        else:
            st.error("La trascrizione non è stata generata.")
else:
    st.info("Carica un file audio per iniziare.")
