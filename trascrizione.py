# trascrizione.py
# App Streamlit robusta per trascrizione audio con Faster-Whisper
# - Precheck deterministico
# - Conversione sicura a WAV 16 kHz mono
# - Scelta modello automatica
# - Messaggi d'errore chiari e download TXT/SRT

import os
import io
import json
import math
import shutil
import tempfile
import subprocess
from datetime import timedelta

import streamlit as st
from faster_whisper import WhisperModel

# ------------------------ Configurazione ------------------------

APP_TITLE = "Trascrizione audio (robusta)"
ALLOWED_EXTS = ["mp3", "m4a", "aac", "wav", "flac", "ogg", "opus", "wma"]  # ffmpeg-friendly

# Limiti (sovrascrivibili via env)
MAX_DURATION_S = int(os.getenv("MAX_DURATION_S", "3600"))   # max 60 min
MAX_SIZE_MB    = int(os.getenv("MAX_SIZE_MB", "200"))       # max 200 MB

# Modello/compute (override opzionale via env)
ENV_MODEL_NAME   = os.getenv("WHISPER_MODEL", "auto")       # auto | tiny | base | small | medium | large-v3 | ecc.
ENV_COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")        # int8 (CPU) di default

# ------------------------ Utilità di sistema ------------------------

def which_or_error(bin_name: str):
    path = shutil.which(bin_name)
    if not path:
        raise RuntimeError(f"'{bin_name}' non trovato nell'ambiente. Impossibile procedere.")
    return path

def available_mem_gb() -> float:
    # Stima memoria disponibile senza dipendenze extra
    try:
        pagesize = os.sysconf("SC_PAGE_SIZE")
        av_pages = os.sysconf("SC_AVPHYS_PAGES")
        return (pagesize * av_pages) / (1024 ** 3)
    except Exception:
        return 4.0  # fallback prudente

def human_size(bytes_val: int) -> str:
    if bytes_val is None:
        return "N/D"
    units = ["B","KB","MB","GB","TB"]
    size = float(bytes_val)
    for u in units:
        if size < 1024.0:
            return f"{size:.1f} {u}"
        size /= 1024.0
    return f"{size:.1f} PB"

def run_subprocess(cmd: list, check=False, capture=True, text=True):
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=text
    )

def ffprobe_json(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,bit_rate,size",
        "-show_streams",
        "-of", "json",
        path
    ]
    res = run_subprocess(cmd)
    if res.returncode != 0 or not res.stdout.strip():
        raise ValueError("Impossibile leggere il file (ffprobe). File corrotto o formato non supportato.")
    return json.loads(res.stdout)

def pick_first_audio_stream(meta: dict):
    for s in meta.get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    return None

def s_to_srt_ts(t: float) -> str:
    if t is None:
        t = 0.0
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000))
    h = ms // 3600000
    ms = ms % 3600000
    m = ms // 60000
    ms = ms % 60000
    s = ms // 1000
    ms = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_srt(segments) -> str:
    # segments: iter/seq di oggetti con .start .end .text
    lines = []
    for i, seg in enumerate(segments, 1):
        start = s_to_srt_ts(seg["start"])
        end   = s_to_srt_ts(seg["end"])
        text  = (seg["text"] or "").strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text if text else "")
        lines.append("")  # separatore blocchi
    return "\n".join(lines).strip() + "\n"

# ------------------------ Precheck & Conversione ------------------------

def preflight_and_prepare(src_path: str):
    which_or_error("ffmpeg")
    which_or_error("ffprobe")

    # 1) dimensione
    size_bytes = os.path.getsize(src_path)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise ValueError(f"File troppo grande ({size_mb:.1f} MB). Limite: {MAX_SIZE_MB} MB.")

    # 2) lettura metadata + selezione stream audio
    meta = ffprobe_json(src_path)
    fmt = meta.get("format", {}) or {}
    duration = float(fmt.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError("Durata non rilevabile: file corrotto o non audio.")
    if duration > MAX_DURATION_S:
        raise ValueError(f"Audio troppo lungo ({duration/60:.0f} min). Limite: {MAX_DURATION_S/60:.0f} min.")

    astream = pick_first_audio_stream(meta)
    if not astream:
        raise ValueError("Nessuno stream audio rilevato (file solo video/metadata o danneggiato).")

    # 3) test di decodifica rapido (5s): intercetta corruzioni/codec problematici
    test = run_subprocess(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-xerror",
         "-t", "5", "-i", src_path, "-f", "null", "-"]
    )
    if test.returncode != 0:
        msg = test.stderr.strip() or "Formato/codec non decodificabile."
        raise ValueError(f"Decodifica fallita: {msg}")

    # 4) conversione sicura → WAV PCM 16 kHz mono (rimuove metadati, video, sottotitoli, ecc.)
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    conv = run_subprocess(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y",
         "-i", src_path,
         "-map", "0:a:0",             # solo primo audio
         "-map_metadata", "-1",       # nessun metadata
         "-vn", "-sn", "-dn",         # no video/subs/data
         "-ac", "1",                  # mono
         "-ar", "16000",              # 16 kHz
         "-acodec", "pcm_s16le",      # PCM 16bit
         dst
        ]
    )
    if conv.returncode != 0:
        raise ValueError(f"Conversione audio fallita: {conv.stderr.strip()}")

    # Report utile per UI
    report = {
        "original": {
            "codec": astream.get("codec_name") or "n/d",
            "sample_rate": astream.get("sample_rate") or "n/d",
            "channels": astream.get("channels") or "n/d",
            "bit_rate": fmt.get("bit_rate") or astream.get("bit_rate"),
            "duration_s": duration,
            "size": human_size(int(fmt.get("size") or size_bytes)),
        },
        "safe_wav": {
            "path": dst,
            "target": "WAV PCM s16le, mono, 16 kHz",
        }
    }
    return dst, report

# ------------------------ Modello ------------------------

def select_model() -> dict:
    # Se l'utente ha fissato un modello, rispettiamolo
    if ENV_MODEL_NAME and ENV_MODEL_NAME.lower() != "auto":
        return {"name": ENV_MODEL_NAME, "compute_type": ENV_COMPUTE_TYPE, "chunk_length": 20}

    mem_gb = available_mem_gb()
    if mem_gb < 2.0:
        return {"name": "tiny",  "compute_type": "int8", "chunk_length": 10}
    elif mem_gb < 4.0:
        return {"name": "base",  "compute_type": "int8", "chunk_length": 15}
    else:
        # small è un buon compromesso su CPU; su GPU cambiare compute_type via env
        return {"name": "small", "compute_type": ENV_COMPUTE_TYPE, "chunk_length": 20}

@st.cache_resource(show_spinner=False)
def get_model_cached(name: str, compute_type: str):
    # Nota: download del modello la prima volta
    return WhisperModel(
        name,
        compute_type=compute_type,   # es. "int8" su CPU
        num_workers=1,
        cpu_threads=max(2, (os.cpu_count() or 2) // 2),
    )

def transcribe_wav(path: str, language_opt: str | None, cfg: dict):
    model = get_model_cached(cfg["name"], cfg["compute_type"])

    # Costruiamo il generatore e accumuliamo i segmenti per output TXT/SRT
    segments_iter, info = model.transcribe(
        path,
        vad_filter=True,
        chunk_length=cfg["chunk_length"],
        beam_size=1,
        temperature=0.0,
        language=None if language_opt in (None, "auto") else language_opt,
        task="transcribe",
    )

    collected = []
    for seg in segments_iter:
        collected.append({
            "start": float(seg.start or 0.0),
            "end": float(seg.end or 0.0),
            "text": seg.text.strip()
        })

    return collected, info  # info.language, info.duration, ecc.

# ------------------------ UI ------------------------

st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)

with st.sidebar:
    st.subheader("Impostazioni")
    st.markdown("I file vengono **verificati** e **convertiti** prima della trascrizione, per evitare crash.")
    st.caption(f"Limiti correnti — Durata: {MAX_DURATION_S//60} min · Dimensione: {MAX_SIZE_MB} MB")
    language_opt = st.selectbox(
        "Lingua forzata (opzionale)",
        options=["auto","it","en","fr","de","es","pt","nl","sv","pl","ru","ja","zh"],
        index=0,
        help="Lascia 'auto' per rilevamento automatico."
    )
    st.divider()
    st.caption("Modello/compute:")
    cfg_preview = select_model()
    st.code(f"model={cfg_preview['name']} · compute={cfg_preview['compute_type']} · chunk={cfg_preview['chunk_length']}s", language="bash")
    st.caption("Override via env: WHISPER_MODEL, COMPUTE_TYPE")

st.markdown("Carica un **file audio**. L'app esegue i **controlli preliminari** e solo se passa li trascrive.")

uploaded = st.file_uploader("Seleziona un file", type=ALLOWED_EXTS, accept_multiple_files=False, label_visibility="visible")

if uploaded is not None:
    # Salvataggio in tmp
    suffix = "." + (uploaded.name.split(".")[-1].lower() if "." in uploaded.name else "bin")
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    with open(tmp_in, "wb") as f:
        f.write(uploaded.getbuffer())

    safe_wav = None
    try:
        # -------------------- PRECHECK --------------------
        with st.status("Controlli preliminari in corso…", expanded=True) as status:
            st.write("• Verifica presenza ffmpeg/ffprobe")
            which_or_error("ffmpeg"); which_or_error("ffprobe")
            st.write("• Lettura metadata e controlli di durata/dimensione")
            safe_wav, report = preflight_and_prepare(tmp_in)
            st.success("Precheck superato.")
            status.update(label="Precheck completato", state="complete")

        st.success("File accettato. Conversione sicura effettuata (WAV mono 16 kHz).")
        with st.expander("Dettagli file"):
            st.json(report)

        # -------------------- TRASCRIZIONE --------------------
        cfg = select_model()
        with st.status(f"Caricamento modello '{cfg['name']}'…", expanded=False) as s:
            _ = get_model_cached(cfg["name"], cfg["compute_type"])
            s.update(label=f"Modello pronto: {cfg['name']} ({cfg['compute_type']})", state="complete")

        with st.status("Trascrizione in corso…", expanded=True) as status:
            st.write("• Avvio decodifica con VAD")
            segs, info = transcribe_wav(safe_wav, language_opt, cfg)
            st.write(f"• Rilevata lingua: **{info.language or 'auto'}**")
            st.write(f"• Segmenti: **{len(segs)}**")
            status.update(label="Trascrizione completata", state="complete")

        # -------------------- OUTPUT --------------------
        text_out = "\n".join(s["text"] for s in segs).strip()
        srt_out = build_srt(segs)

        st.subheader("Testo")
        st.text_area("Trascrizione", value=text_out, height=240, label_visibility="collapsed")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Scarica TXT",
                data=text_out.encode("utf-8"),
                file_name=os.path.splitext(uploaded.name)[0] + ".txt",
                mime="text/plain"
            )
        with col2:
            st.download_button(
                "⬇️ Scarica SRT",
                data=srt_out.encode("utf-8"),
                file_name=os.path.splitext(uploaded.name)[0] + ".srt",
                mime="application/x-subrip"
            )

    except Exception as e:
        st.error(f"Impossibile processare il file: {e}")
        st.stop()
    finally:
        # Pulizia file temporanei
        try:
            if safe_wav and os.path.exists(safe_wav):
                os.unlink(safe_wav)
        except Exception:
            pass
        try:
            if tmp_in and os.path.exists(tmp_in):
                os.unlink(tmp_in)
        except Exception:
            pass

else:
    st.info("Attendi selezione file. Formati consigliati: MP3, M4A/AAC, WAV, FLAC, OGG/Opus.")


