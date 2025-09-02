import streamlit as st
from faster_whisper import WhisperModel
import tempfile, os, datetime, time
import ffmpeg

# ---------------------- Helpers ----------------------
def fmt_hms(seconds: float) -> str:
    if seconds is None or seconds == float("inf"):
        return "--:--"
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def fmt_timecode(ts):
    td = datetime.timedelta(seconds=float(ts))
    return str(td)[:12].replace(".", ",").zfill(12)

def to_srt(segments):
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(f"{i}\n{fmt_timecode(s.start)} --> {fmt_timecode(s.end)}\n{s.text.strip()}\n")
    return "\n".join(lines)

def media_duration_seconds(path: str) -> float:
    try:
        info = ffmpeg.probe(path)
        return float(info.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0

def convert_to_wav_with_progress(src_path: str, progress):
    """Converte a WAV 16 kHz mono mostrando una barra e una stima dell'ETA."""
    import time as _t
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out.close()
    total_media = media_duration_seconds(src_path)  # secondi del file di origine
    t0 = _t.time()
    proc = (
        ffmpeg
        .input(src_path)
        .output(out.name, ac=1, ar=16000)
        .global_args("-progress", "pipe:1", "-nostats")
        .overwrite_output()
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    processed_sec = 0.0
    while True:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:  # processo finito
                break
            continue
        try:
            s = line.decode("utf-8", errors="ignore").strip()
        except Exception:
            s = ""
        if s.startswith("out_time_ms="):
            try:
                processed_sec = int(s.split("=")[1]) / 1_000_000.0
            except Exception:
                processed_sec = processed_sec
            pct_media = int(min(100, (processed_sec / max(1e-6, total_media)) * 100))
            elapsed = max(1e-6, _t.time() - t0)
            rate = processed_sec / elapsed  # secondi di audio elaborati al secondo
            remaining = (max(0.0, total_media - processed_sec) / max(1e-6, rate))
            progress.progress(pct_media, text=f"Conversione… {pct_media}%  • ETA ~ {fmt_hms(remaining)}")
        elif s.startswith("progress=") and s.split("=")[1] == "end":
            progress.progress(100, text="Conversione completata ✅")
            break
    proc.wait()
    return out.name

# ---------------------- UI ----------------------
st.set_page_config(page_title="Trascrizione audio by Roberto M.", page_icon="🎙️")
st.title("🎙️ Trascrizione audio by Roberto M.")

uploaded_file = st.file_uploader(
    "📤 Carica un file audio/video (MP3, WAV, M4A, AAC, OGG, FLAC, MP4, MOV, MKV, WEBM, AVI)",
    type=["mp3","wav","m4a","aac","ogg","flac","mp4","mov","mkv","webm","avi"],
)

colA, colB, colC, colD = st.columns([1,1,1,1])
with colA:
    model_size = st.selectbox("⚙️ Modello", ["tiny", "base", "small", "medium"], index=2)
with colB:
    language = st.selectbox("🌍 Lingua", ["auto", "it", "en", "es", "fr", "de"], index=0)
with colC:
    compute = st.selectbox("🧮 Precisione (CPU)", ["int8", "int8_float16", "float32"], index=0)
with colD:
    vad = st.checkbox("🧹 VAD", value=True, help="Rimuove silenzi/rumori tra frasi")

if uploaded_file:
    # Salva upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        src_path = tmp.name

    # 1) Conversione con ETA
    st.subheader("1) Conversione con ffmpeg")
    conv_bar = st.progress(0, text="Preparazione…")
    try:
        wav_path = convert_to_wav_with_progress(src_path, conv_bar)
    except Exception as e:
        conv_bar.empty()
        st.error(f"Errore nella conversione: {e}")
        st.stop()

    # 2) Trascrizione con ETA
    st.subheader("2) Trascrizione")
    trans_bar = st.progress(0, text="Carico modello…")
    try:
        t0 = time.time()
        model = WhisperModel(model_size, device="cpu", compute_type=compute)
        segments_gen, info = model.transcribe(
            wav_path,
            language=None if language == "auto" else language,
            vad_filter=vad,
        )
        total_dur = max(1e-6, float(info.duration or 0.0))  # secondi dell'audio
        segments, last_end = [], 0.0

        for seg in segments_gen:
            segments.append(seg)
            last_end = getattr(seg, "end", last_end)
            frac = min(1.0, last_end / total_dur)
            pct = int(frac * 100)
            elapsed = time.time() - t0
            rate = max(1e-6, frac / max(1e-6, elapsed))  # frazione per secondo
            remaining = (1.0 - frac) / rate
            trans_bar.progress(pct, text=f"Trascrizione… {pct}%  • ETA ~ {fmt_hms(remaining)}")

        trans_bar.progress(100, text="Trascrizione completata ✅")

        text = "".join(s.text for s in segments).strip()
        srt_text = to_srt(segments)

        # Durata in minuti e in mm:ss
        dur_min = (info.duration or 0.0) / 60.0
        dur_hms = fmt_hms(info.duration or 0.0)
        st.success(f"Fatto! Durata: {dur_hms}  ({dur_min:.1f} min) – Segmenti: {len(segments)}")
        st.text_area("📝 Testo", text, height=300)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Scarica .txt", text, file_name="trascrizione.txt")
        with c2:
            st.download_button("⬇️ Scarica .srt", srt_text, file_name="trascrizione.srt", mime="application/x-subrip")

        with st.expander("📚 Dettaglio segmenti"):
            for s in segments:
                st.write(f"[{s.start:.2f} → {s.end:.2f}] {s.text.strip()}")

    except Exception as e:
        trans_bar.empty()
        st.error(f"Errore nella trascrizione: {e}")
    finally:
        try: os.unlink(src_path)
        except Exception: pass
        try: os.unlink(wav_path)
        except Exception: pass

else:
    st.caption("Carica un file per iniziare. Conversione e trascrizione mostrano ETA; la durata finale è in minuti e in formato mm:ss.")
