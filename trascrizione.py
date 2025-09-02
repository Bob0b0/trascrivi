import io
import os
import time
import tempfile
from typing import List, Tuple

import streamlit as st
from faster_whisper import WhisperModel


# ----------------------------- Util -------------------------------------------

def tidy_text(text: str) -> str:
    """Pulisce e formatta il testo:
    - spazi multipli
    - spazi prima della punteggiatura
    - aggiunge spazi dopo , ;
    - rimuove ripetizioni semplici di parola
    - spezza in frasi su ., !, ?, … e capitalizza
    """
    import re

    if not text:
        return ""

    t = text.replace("\r", " ").strip()

    # comprime spazi/tabs
    t = re.sub(r"[ \t]+", " ", t)

    # niente spazio prima di punteggiatura
    t = re.sub(r"\s+([,;:.!?…])", r"\1", t)

    # spazio dopo virgola e punto e virgola se manca
    t = re.sub(r"([,;])([^\s])", r"\1 \2", t)

    # rimozione ripetizioni immediate di una parola (molto semplice)
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.IGNORECASE)

    # frasi su . ! ? …
    sentences = re.split(r"(?<=[.!?…])\s+", t)
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]

    return "\n\n".join(sentences).strip()


def sec_to_srt_time(seconds: float) -> str:
    """Converte secondi float in timecode SRT HH:MM:SS,mmm"""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_text_and_srt(segments) -> Tuple[str, str]:
    """Ritorna (testo_grezzo, srt) a partire dai segmenti di faster-whisper"""
    pieces: List[str] = []
    srt_lines: List[str] = []
    idx = 1

    for seg in segments:
        # testo “grezzo” senza timecode
        pieces.append(seg.text.strip())

        # srt
        start = sec_to_srt_time(seg.start)
        end = sec_to_srt_time(seg.end)
        srt_lines.append(
            f"{idx}\n{start} --> {end}\n{seg.text.strip()}\n"
        )
        idx += 1

    raw_text = " ".join(pieces).strip()
    srt_text = "\n".join(srt_lines).strip()
    return raw_text, srt_text


def save_bytes_for_download(content: str, suffix: str) -> str:
    """Scrive contenuto testuale in un file temporaneo e ritorna il path"""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with io.open(tf.name, "w", encoding="utf-8") as f:
        f.write(content)
    return tf.name


# ---------------------------- UI / App ----------------------------------------

st.set_page_config(
    page_title="Trascrizione audio by Roberto M.",
    layout="centered",
)

st.title("Trascrizione audio by Roberto M.")

st.write(
    "Carica un file audio/video, scegli modello e opzioni, quindi avvia la trascrizione. "
    "Durante l'elaborazione vedrai una barra di avanzamento con la stima del tempo residuo."
)

with st.expander("Opzioni avanzate", expanded=False):
    model_name = st.selectbox(
        "Modello Whisper",
        options=["tiny", "base", "small", "medium", "large-v3"],
        index=1,  # default: base
        help=(
            "Più il modello è grande, più la qualità tende a salire ma aumenta il tempo di calcolo. "
            "Su CPU è consigliato 'base' o 'small'."
        ),
    )
    lang = st.selectbox(
        "Lingua",
        options=["auto", "it", "en", "fr", "de", "es", "pt"],
        index=0,
        help="Se non sei sicuro, lascia **auto**.",
    )
    beam_size = st.slider(
        "Beam size (ricerca frasi migliori)",
        1, 10, 5, help="Valori più alti migliorano un po' l'accuratezza ma rallentano."
    )

# Preferenze rapide
show_timestamps = st.checkbox(
    "Includi timestamp nei file di output (SRT)",
    value=False,
    help="Disattivato di default. Abilitalo solo se vuoi anche i sottotitoli .srt.",
)
auto_improve = st.checkbox(
    "Migliora e formatta automaticamente al termine",
    value=True,
    help="Se attivo, l'app pulirà e formatterà il testo subito dopo la trascrizione.",
)

st.subheader("Carica un file audio/video")
uploaded = st.file_uploader(
    "Drag and drop file here",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

start_btn = st.button("Avvia trascrizione", disabled=(uploaded is None))

if uploaded and start_btn:
    # Salva caricamento su disco (faster-whisper richiede un path)
    with st.status("Preparazione file…", expanded=False) as status:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}")
        tmp_in.write(uploaded.read())
        tmp_in.flush()
        tmp_in.close()
        status.update(label="File pronto", state="complete")

    # Caricamento modello
    t0 = time.time()
    with st.status(
        f"Caricamento modello **{model_name}** (compute_type: int8)…",
        expanded=False
    ) as status:
        try:
            model = WhisperModel(model_name, compute_type="int8")  # CPU friendly
        except Exception as e:
            status.update(label="Errore nel caricamento del modello", state="error")
            st.error(f"Impossibile caricare il modello: {e}")
            os.unlink(tmp_in.name)
            st.stop()

        elapsed = time.time() - t0
        status.update(
            label=f"Modello {model_name} pronto in {elapsed:0.2f}s",
            state="complete"
        )

    # Trascrizione
    st.subheader("Elaborazione")
    progress = st.progress(0)
    info_placeholder = st.empty()

    try:
        info_placeholder.info("Avvio trascrizione…")
        seg_gen, info = model.transcribe(
            tmp_in.name,
            language=None if lang == "auto" else lang,
            beam_size=beam_size,
            vad_filter=True,
        )

        # Colleziona segmenti aggiornando progress (best-effort)
        segments = []
        last_update = time.time()
        for s in seg_gen:
            segments.append(s)
            # throttle UI updates
            if (time.time() - last_update) > 0.2 and info.duration:
                # stima grezza progresso: fine segmento / durata
                p = min(1.0, s.end / max(1e-6, info.duration))
                progress.progress(int(p * 100))
                info_placeholder.info(
                    f"Elaborazione… ~{int(p*100)}%  •  "
                    f"segmenti: {len(segments)}  •  durata: {int(info.duration)}s"
                )
                last_update = time.time()

        progress.progress(100)
        info_placeholder.success("Completato")

    except Exception as e:
        st.error(f"Errore durante la trascrizione: {e}")
        os.unlink(tmp_in.name)
        st.stop()
    finally:
        # libera il file di input
        try:
            os.unlink(tmp_in.name)
        except Exception:
            pass

    # Costruisci output (testo grezzo e SRT)
    raw_text, srt_text = segments_to_text_and_srt(segments)

    # Miglioramento automatico se richiesto
    final_text = tidy_text(raw_text) if auto_improve else raw_text

    # Mostra testo
    st.subheader("Trascrizione")
    st.caption(
        "Il testo mostrato qui sotto è già **pulito e formattato**"
        " (se l'opzione era attiva)."
    )
    st.text_area("Testo", value=final_text, height=350, label_visibility="collapsed")

    # Download
    st.subheader("Download")

    txt_path = save_bytes_for_download(final_text, ".txt")
    st.download_button(
        "Scarica TXT",
        data=open(txt_path, "rb").read(),
        file_name=os.path.splitext(uploaded.name)[0] + ".txt",
        mime="text/plain",
    )

    if show_timestamps:
        srt_path = save_bytes_for_download(srt_text, ".srt")
        st.download_button(
            "Scarica SRT (sottotitoli)",
            data=open(srt_path, "rb").read(),
            file_name=os.path.splitext(uploaded.name)[0] + ".srt",
            mime="application/x-subrip",
        )

    # Info finali
    st.info(
