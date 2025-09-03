# trascrizione.py
# App Streamlit per trascrivere audio con Faster-Whisper
# - Precheck e conversione a WAV mono 16 kHz
# - Frazionamento automatico per audio lunghi (no settaggi ansiosi)
# - Progress bar per ogni fase
# - Export .txt
# - Prompt di post-produzione auto-compilato con la trascrizione

from __future__ import annotations
import os
import io
import math
import time
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
import ffmpeg  # ffmpeg-python
import numpy as np
from faster_whisper import WhisperModel


# =========================
# Configurazione pagina
# =========================
st.set_page_config(
    page_title="Trascrivi - Audio → Testo",
    page_icon="🗣️",
    layout="wide",
)

APP_NAME = "Trascrivi"
APP_TAGLINE = "Trascrizione audio robusta, con frazionamento automatico"
APP_AUTHOR = "Autore: <inserisci il tuo nome>"  # ← Facoltativo, puoi personalizzare


# =========================
# Costanti operative
# =========================
SUPPORTED_AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma", ".mkv", ".mp4", ".webm")
TARGET_SR = 16000
TARGET_CH = 1
# Se l'audio dura più di questa soglia, si attiva il frazionamento automatico
SPLIT_THRESHOLD_SEC = 30 * 60  # 30 minuti
# Suddividiamo in ~N chunk (adattivo). Ogni chunk resterà comunque tra MIN e MAX.
TARGET_CHUNKS = 12
MIN_CHUNK_SEC = 5 * 60   # 5 minuti
MAX_CHUNK_SEC = 15 * 60  # 15 minuti
CHUNK_OVERLAP_SEC = 2    # overlap interno per non troncare parole

# Impostazioni modello (minimali, niente ansie)
DEFAULT_MODEL = "tiny"  # "tiny", "base", "small", "medium", "large-v3"
DEFAULT_COMPUTE_TYPE = "int8"  # CPU veloce. Opzioni: "int8", "int8_float16", "float16", "float32"...


# =========================
# Utility audio
# =========================
def probe_media(path: Path) -> dict:
    """Ritorna i metadati ffprobe. Lancia eccezione se fallisce."""
    return ffmpeg.probe(str(path))


def get_duration_seconds(probe: dict) -> float:
    """Estrae la durata in secondi dal probe (format.duration)."""
    try:
        return float(probe["format"]["duration"])
    except Exception:
        return 0.0


def convert_to_wav_mono_16k(src: Path, dst: Path) -> None:
    """Converte qualunque audio a WAV mono 16 kHz."""
    (
        ffmpeg
        .input(str(src))
        .output(
            str(dst),
            ac=TARGET_CH,
            ar=TARGET_SR,
            f="wav",
            # codec PCM 16 bit (scelta solida e interoperabile)
            acodec="pcm_s16le",
            loglevel="error",
        )
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def compute_chunk_length(duration_sec: float) -> int:
    """Calcola una lunghezza chunk 'comoda' per avere ~TARGET_CHUNKS segmenti."""
    if duration_sec <= 0:
        return MAX_CHUNK_SEC
    raw = math.ceil(duration_sec / TARGET_CHUNKS)
    return int(min(MAX_CHUNK_SEC, max(MIN_CHUNK_SEC, raw)))


def split_wav_by_time(src_wav: Path, out_dir: Path, chunk_sec: int, overlap_sec: int = CHUNK_OVERLAP_SEC) -> List[Path]:
    """Crea chunk WAV consecutivi con piccolo overlap agli estremi (solo in mezzo)."""
    info = probe_media(src_wav)
    total = get_duration_seconds(info)
    if total <= 0:
        return [src_wav]

    chunks: List[Path] = []
    num = math.ceil(total / chunk_sec)

    for i in range(num):
        start = i * chunk_sec
        # Applica overlap agli intermezzi
        extra_left = overlap_sec if i > 0 else 0
        extra_right = overlap_sec if i < num - 1 else 0
        ss = max(0, start - extra_left)
        effective_dur = min(chunk_sec + extra_left + extra_right, max(0.1, total - ss))

        out = out_dir / f"chunk_{i:03d}.wav"
        (
            ffmpeg
            .input(str(src_wav), ss=ss)
            .output(
                str(out),
                t=effective_dur,
                acodec="copy",  # WAV PCM → copia senza ricompressione
                loglevel="error",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        chunks.append(out)
    return chunks


# =========================
# Prompt post-produzione
# =========================
def render_post_production_prompt(transcript_text: str) -> str:
    # Template richiesto, con inserimento automatico del testo
    return f"""SEI UN EDITOR PROFESSIONISTA.

Migliora il testo seguente senza alterare i contenuti: correggi refusi,
punteggiatura, ripetizioni, sintassi, lessico datato; applica una
formattazione leggibile con paragrafi e titoli minimi se opportuno.
Mantieni lo stile orale, ma rendilo scorrevole. Lingua: it.

<< TESTO DA INSERIRE >>
[TESTO DA INSERIRE QUI SOTTO]
{transcript_text}

Output richiesto: testo finale pulito, pronto per la pubblicazione.
§§§§§§
questo era il testo (nel caso l'avessi perso)

TESTO DA MIGLIORARE (tra i delimitatori):
§§§§§§
{transcript_text}
§§§§§§
"""


def show_post_production_prompt_ui(txt_path: Optional[Path] = None):
    st.markdown("### ✍️ Prompt per post-produzione (copia e incolla nella tua AI)")
    transcript_text = st.session_state.get("transcript_text", "").strip()
    if not transcript_text and txt_path and txt_path.is_file():
        transcript_text = txt_path.read_text(encoding="utf-8").strip()

    if not transcript_text:
        st.info("Nessuna trascrizione disponibile. Esegui una trascrizione o seleziona un file .txt generato dall'app.")
        return

    prompt_text = render_post_production_prompt(transcript_text)

    st.text_area(
        "Prompt precompilato",
        value=prompt_text,
        height=420,
        key="post_production_prompt_area",
    )

    st.download_button(
        "Scarica prompt (.txt)",
        data=prompt_text.encode("utf-8"),
        file_name="prompt_postproduzione.txt",
        mime="text/plain",
    )


# =========================
# UI - Sidebar
# =========================
st.sidebar.title("🗣️ Trascrivi")
st.sidebar.caption(APP_TAGLINE)

uploaded = st.sidebar.file_uploader("Carica un file audio/video", type=[e[1:] for e in SUPPORTED_AUDIO_EXT])

model_size = st.sidebar.selectbox(
    "Modello",
    ["tiny", "base", "small", "medium", "large-v3"],
    index=["tiny", "base", "small", "medium", "large-v3"].index(DEFAULT_MODEL),
    help="Modelli più piccoli = più veloci (ma meno accurati).",
)

language_opt = st.sidebar.selectbox(
    "Lingua",
    ["Auto", "Italiano", "Inglese"],
    index=0,
    help="Se lasci 'Auto', rilevamento automatico.",
)

run_btn = st.sidebar.button("▶️ Avvia trascrizione", type="primary")

with st.sidebar.expander("ℹ️ Info & limiti"):
    st.markdown(
        """
- Formati accettati: **mp3, wav, m4a, aac, ogg, flac, wma, mkv, mp4, webm**.
- I file lunghi vengono **frazionati automaticamente** per evitare errori di memoria/timeout.
- Audio multi-ora: la velocità dipende molto dalla CPU e dal modello scelto.
- Output: **.txt** con la trascrizione completa e **prompt** per post-produzione.
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption(APP_AUTHOR)


# =========================
# UI - Main
# =========================
st.title("Trascrizione audio → testo")
st.write("Carica un file e avvia. L’app si occupa di conversione, verifica e frazionamento se serve.")

with st.expander("📘 Istruzioni rapide"):
    st.markdown(
        """
1) Carica il file e premi **Avvia trascrizione**  
2) Attendi: vedrai **barre di avanzamento** per ogni fase  
3) Scarica il **.txt** o usa il **Prompt per post-produzione** per rifinire il testo
        """
    )


# =========================
# Funzione principale
# =========================
def trascrivi_file(file_bytes: bytes, filename: str, model_name: str, language_ui: str) -> Tuple[str, Path]:
    """
    Esegue:
    - Salvataggio temporaneo
    - Precheck + conversione WAV 16k mono
    - Eventuale frazionamento
    - Trascrizione chunk per chunk con progress bar
    - Salvataggio .txt finale
    Ritorna: (trascrizione, path_txt)
    """
    language_code: Optional[str] = None
    if language_ui == "Italiano":
        language_code = "it"
    elif language_ui == "Inglese":
        language_code = "en"

    # 1) Salva il file originale su disco
    tmp_root = Path(tempfile.mkdtemp(prefix="trascrivi_"))
    src_path = tmp_root / filename
    src_path.write_bytes(file_bytes)

    # 2) Precheck
    st.subheader("Verifica file")
    pre_prog = st.progress(0, text="Analisi del file…")
    try:
        pr = probe_media(src_path)
        duration_sec = get_duration_seconds(pr)
        a_streams = [s for s in pr.get("streams", []) if s.get("codec_type") == "audio"]
        audio_codec = a_streams[0].get("codec_name") if a_streams else "sconosciuto"
        pre_prog.progress(25, text="Analisi del file… OK")
    except Exception as e:
        pre_prog.empty()
        st.error(f"Impossibile leggere i metadati del file. Dettagli: {e}")
        raise

    # 3) Conversione a WAV mono 16 kHz
    st.write(f"- Durata: **{duration_sec/60:.1f} min** — Codec: **{audio_codec}**")
    wav_path = tmp_root / f"{src_path.stem}_16kmono.wav"
    pre_prog.progress(50, text="Conversione a WAV 16 kHz mono…")
    try:
        convert_to_wav_mono_16k(src_path, wav_path)
        pre_prog.progress(100, text="Conversione completata")
        time.sleep(0.2)
        pre_prog.empty()
    except Exception as e:
        pre_prog.empty()
        st.error(f"Errore nella conversione a WAV: {e}")
        raise

    # 4) Frazionamento automatico (se necessario)
    chunks: List[Path]
    split_needed = duration_sec >= SPLIT_THRESHOLD_SEC
    chunk_len = compute_chunk_length(duration_sec) if split_needed else int(duration_sec) or MAX_CHUNK_SEC
    if split_needed:
        st.subheader("Frazionamento automatico")
        st.write(f"Audio lungo → verrà suddiviso in chunk da ~**{chunk_len//60} min** (overlap {CHUNK_OVERLAP_SEC}s).")
        split_prog = st.progress(0, text="Creo i chunk…")
        chunk_dir = tmp_root / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks = split_wav_by_time(wav_path, chunk_dir, chunk_len, CHUNK_OVERLAP_SEC)
        for i in range(len(chunks)):
            split_prog.progress(int((i + 1) / len(chunks) * 100))
        split_prog.empty()
    else:
        chunks = [wav_path]

    st.subheader("Trascrizione")
    st.write(f"Modello: **{model_name}** — Linguaggio: **{language_ui}**")

    # 5) Carica modello Faster-Whisper
    load_prog = st.progress(0, text="Carico il modello…")
    model = WhisperModel(model_name, device="cpu", compute_type=DEFAULT_COMPUTE_TYPE)
    load_prog.progress(100, text="Modello pronto")
    time.sleep(0.1)
    load_prog.empty()

    # 6) Trascrivi chunk per chunk
    total_chunks = len(chunks)
    trans_prog = st.progress(0, text="Trascrizione in corso…")
    collected_text: List[str] = []

    for idx, ch in enumerate(chunks, start=1):
        trans_prog.progress(int((idx - 1) / total_chunks * 100), text=f"Chunk {idx}/{total_chunks}…")

        segments, info = model.transcribe(
            str(ch),
            language=language_code,  # None = auto
            vad_filter=True,
            beam_size=1,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        piece = "".join(seg.text for seg in segments).strip()
        collected_text.append(piece if piece else "")

    trans_prog.progress(100, text="Trascrizione completata")
    time.sleep(0.2)
    trans_prog.empty()

    # 7) Merge & salvataggio testo
    full_text = "\n\n".join([p for p in collected_text if p]).strip()
    if not full_text:
        st.warning("La trascrizione è vuota. Potrebbe esserci un problema con l’audio.")
    out_txt = tmp_root / f"{src_path.stem}_trascrizione.txt"
    out_txt.write_text(full_text, encoding="utf-8")

    # Memorizza in sessione per il prompt post-produzione
    st.session_state["transcript_text"] = full_text
    st.session_state["transcript_txt_path"] = str(out_txt)

    return full_text, out_txt


# =========================
# Run
# =========================
if run_btn:
    if not uploaded:
        st.error("Carica prima un file.")
    else:
        # Esecuzione principale
        try:
            with st.spinner("Elaborazione…"):
                text, txt_path = trascrivi_file(
                    uploaded.read(),
                    uploaded.name,
                    model_size,
                    language_opt,
                )
        except Exception as e:
            st.exception(e)
            st.stop()

        st.success("Operazione completata.")

        # Download trascrizione
        st.subheader("Risultato")
        st.download_button(
            "⬇️ Scarica trascrizione (.txt)",
            data=text.encode("utf-8"),
            file_name=Path(uploaded.name).stem + "_trascrizione.txt",
            mime="text/plain",
        )

        st.text_area("Anteprima testo", value=text, height=300)

        # Prompt post-produzione (auto-compilato)
        st.markdown("---")
        show_post_production_prompt_ui(Path(st.session_state.get("transcript_txt_path", "")))
else:
    # Se abbiamo già una trascrizione in sessione (turni precedenti), mostra diretto il prompt
    if "transcript_text" in st.session_state and st.session_state["transcript_text"]:
        st.info("Trascrizione già disponibile in questa sessione.")
        st.text_area("Anteprima testo", value=st.session_state["transcript_text"], height=240)
        st.markdown("---")
        show_post_production_prompt_ui(Path(st.session_state.get("transcript_txt_path", "")))

