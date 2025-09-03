import os
import io
import math
import json
import time
import hashlib
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
import ffmpeg
from faster_whisper import WhisperModel


# =========================
# CONFIG / COSTANTI
# =========================
APP_TITLE = "Trascrivi – Robusta"
APP_AUTHOR = "Robusta"
TARGET_CHUNK_MIN = 6          # ~6 minuti a pezzo, deciso automaticamente
OVERLAP_SEC = 3               # overlap interno (non esposto all'utente)
MAX_PARTS = 32                # limite di sicurezza per file molto lunghi
MAX_RETRIES = 2               # tentativi per pezzo in errore
MODEL_DEFAULT = "tiny"        # default per test rapidi
COMPUTE_TYPE = "int8"         # veloce su CPU
CHUNK_INTERNAL_SEC = 20       # segmentazione interna di faster-whisper per ridurre RAM


# =========================
# UTIL
# =========================
def human_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h}:{m:02}:{s:02}"
    return f"{m}:{s:02}"


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def ensure_wav_mono16k(src_path: str, dst_path: str) -> None:
    (
        ffmpeg
        .input(src_path)
        .output(dst_path, ac=1, ar=16000, format="wav", y=None)
        .overwrite_output()
        .run(quiet=True)
    )


def probe_duration_seconds(path: str) -> float:
    try:
        info = ffmpeg.probe(path)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


def build_plan(duration_s: float) -> List[Tuple[int, int]]:
    """
    Restituisce lista di (start_sec, dur_sec) con overlap gestito internamente.
    Distribuzione uniforme -> meno rischi sugli ultimi pezzi.
    """
    if duration_s <= 0:
        return [(0, 0)]
    target = TARGET_CHUNK_MIN * 60
    n_parts = max(1, min(MAX_PARTS, math.ceil(duration_s / target)))
    base = duration_s / n_parts

    plan = []
    for i in range(n_parts):
        start = int(i * base)
        if i > 0:
            start = max(0, start - OVERLAP_SEC)
        end = duration_s if i == n_parts - 1 else int((i + 1) * base + OVERLAP_SEC)
        end = max(end, start + 1)
        plan.append((start, int(end - start)))
    return plan


@st.cache_resource(show_spinner=False)
def load_model(model_size: str):
    # Nota: num_workers=1 per evitare picchi RAM; cpu_threads = os.cpu_count()
    return WhisperModel(
        model_size,
        device="auto",
        compute_type=COMPUTE_TYPE,
        cpu_threads=os.cpu_count() or 4,
        num_workers=1,
    )


def transcribe_chunk(model, wav_path: str) -> str:
    """
    Una chiamata “stretta”, con parametri conservativi per stabilità e RAM.
    """
    segments, _ = model.transcribe(
        wav_path,
        beam_size=1,
        vad_filter=True,
        word_timestamps=False,
        condition_on_previous_text=False,
        chunk_length=CHUNK_INTERNAL_SEC,
        temperature=0.0,
        compression_ratio_threshold=2.6,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    out = []
    for seg in segments:
        out.append(seg.text)
    return "".join(out).strip()


def write_text(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_editor_prompt(transcript_text: str) -> str:
    return (
        "SEI UN EDITOR PROFESSIONISTA.\n\n"
        "Migliora il testo seguente senza alterare i contenuti: correggi refusi,\n"
        "punteggiatura, ripetizioni, sintassi, lessico datato; applica una\n"
        "formattazione leggibile con paragrafi e titoli minimi se opportuno.\n"
        "Mantieni lo stile orale, ma rendilo scorrevole. Lingua: it.\n"
        "<< TESTO DA INSERIRE >>\n"
        f"[incolla qui il TESTO DA MIGLIORARE]\n\n{transcript_text}\n\n"
        "Output richiesto: testo finale pulito, pronto per la pubblicazione.\n"
        "§§§§§§\n"
        "TESTO DA MIGLIORARE (tra i delimitatori):\n"
    )


# =========================
# UI
# =========================
st.set_page_config(APP_TITLE, layout="wide")
st.title("Trascrizione audio")
st.caption(f"Autore: **{APP_AUTHOR}**")

with st.sidebar:
    st.header("Impostazioni")
    model_size = st.selectbox(
        "Modello",
        ["tiny", "base", "small", "medium", "large-v3"],
        index=["tiny", "base", "small", "medium", "large-v3"].index(MODEL_DEFAULT),
        help="Scegli un modello. Tiny/Base più veloci; Large più accurato ma lento."
    )
    lang = st.text_input(
        "Lingua (opzionale)",
        value="it",
        help="Codice lingua ISO (es. it, en). Lascia vuoto per auto-detectarla."
    )
    st.markdown("---")
    st.write("Le opzioni tecniche di frazionamento sono automatiche per evitare errori.")

uploaded = st.file_uploader("Carica un file audio", type=["mp3", "wav", "m4a", "aac", "flac", "ogg"])

# Stato
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

def log(msg: str):
    st.session_state.log_lines.append(msg)

if uploaded:
    # Salva input su disco
    work_dir = Path(tempfile.gettempdir()) / "trascrivi"
    work_dir.mkdir(parents=True, exist_ok=True)
    src_path = str(work_dir / uploaded.name)
    with open(src_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Normalizza in wav/mono/16k per stabilità
    norm_path = str(work_dir / (Path(uploaded.name).stem + "_16k.wav"))
    try:
        ensure_wav_mono16k(src_path, norm_path)
    except Exception as e:
        st.error(f"Impossibile preparare l'audio (ffmpeg): {e}")
        st.stop()

    duration = probe_duration_seconds(norm_path)
    st.info(f"Durata: **{human_dur(duration)}**")

    # Checkpoint directory per riprese/partial
    token = f"{Path(uploaded.name).name}-{file_sha1(norm_path)}-{model_size}"
    ckpt_dir = work_dir / f"ckpt_{token}"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    # Piano di slicing
    plan = build_plan(duration)
    n_parts = len(plan)
    st.write(f"Pezzi previsti: **{n_parts}** (auto)")

    # Pulsante azione
    colA, colB = st.columns([1, 1])
    go = colA.button("Trascrivi")
    clear_ckpt = colB.button("Riparti da zero")

    if clear_ckpt:
        for p in ckpt_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        st.success("Checkpoint cancellati.")

    if go:
        model = load_model(model_size)

        # Ripresa: individua pezzi già ok
        done = {p.stem for p in ckpt_dir.glob("chunk_*.txt")}
        progress = st.progress(0, text="In elaborazione…")
        status = st.empty()

        merged_texts: List[str] = []
        failures = []

        for idx, (ss, dd) in enumerate(plan, start=1):
            progress.progress((idx - 1) / n_parts, text=f"Preparazione pezzo {idx}/{n_parts}")

            chunk_tag = f"chunk_{idx:03d}"
            txt_path = ckpt_dir / f"{chunk_tag}.txt"
            wav_path = ckpt_dir / f"{chunk_tag}.wav"

            # Se già fatto, carica
            if txt_path.exists():
                merged_texts.append(read_text(str(txt_path)))
                status.write(f"✓ {chunk_tag} già pronto")
                continue

            # Estrai audio pezzo
            try:
                (
                    ffmpeg
                    .input(norm_path, ss=ss, t=dd)
                    .output(str(wav_path), ac=1, ar=16000, format="wav", y=None)
                    .overwrite_output()
                    .run(quiet=True)
                )
            except Exception as e:
                failures.append(idx)
                log(f"[estrazione] pezzo {idx}: {e}")
                status.error(f"Errore estrazione pezzo {idx} – si prosegue")
                continue

            # Trascrivi con retry
            ok = False
            err_msg = ""
            for attempt in range(1, MAX_RETRIES + 2):
                try:
                    progress.progress((idx - 1) / n_parts, text=f"Trascrizione pezzo {idx}/{n_parts} (tentativo {attempt})")
                    text_out = transcribe_chunk(model, str(wav_path))
                    write_text(str(txt_path), text_out)
                    merged_texts.append(text_out)
                    ok = True
                    status.write(f"✓ pezzo {idx} trascritto")
                    break
                except Exception as e:
                    err_msg = str(e)
                    time.sleep(0.5)  # backoff minimo

            if not ok:
                failures.append(idx)
                log(f"[trascrizione] pezzo {idx} fallito: {err_msg}")
                status.error(f"Errore nel pezzo {idx} – proseguo con i successivi")

            # Pulizia wav del pezzo per risparmiare spazio
            try:
                if wav_path.exists():
                    wav_path.unlink()
            except Exception:
                pass

            progress.progress(idx / n_parts, text=f"Avanzamento {idx}/{n_parts}")

        # Merge finale
        final_text = "\n".join(t for t in merged_texts if t.strip())
        out_txt_path = work_dir / f"{Path(uploaded.name).stem}.trascrizione.txt"
        write_text(str(out_txt_path), final_text)

        st.success("Trascrizione completata (con gestione errori per singolo pezzo).")
        if failures:
            st.warning(f"Pezzi non riusciti: {failures}. Puoi rieseguire: riprenderà solo quelli mancanti.")

        st.download_button(
            "Scarica trascrizione (.txt)",
            data=final_text.encode("utf-8"),
            file_name=out_txt_path.name,
            mime="text/plain",
        )

        # Prompt post-produzione
        st.markdown("### Prompt per post-produzione (copia/incolla in qualunque AI)")
        prompt_text = build_editor_prompt(final_text)
        st.text_area("Prompt completo", value=prompt_text, height=240)
        st.download_button(
            "Scarica prompt (.txt)",
            data=prompt_text.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}.prompt_editor.txt",
            mime="text/plain",
        )

        # Log visibile
        if st.session_state.log_lines:
            st.markdown("### Log")
            st.code("\n".join(st.session_state.log_lines), language="text")

