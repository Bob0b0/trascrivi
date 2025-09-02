import streamlit as st
from faster_whisper import WhisperModel
import tempfile, os, datetime
import ffmpeg

st.set_page_config(page_title="Trascrizione audio", page_icon="🎙️")
st.title("🎧 Trascrizione audio con avanzamento (faster-whisper)")

# ---------------- Utilities ----------------
def fmt_time(ts):
    td = datetime.timedelta(seconds=float(ts))
    return str(td)[:12].replace(".", ",").zfill(12)

def to_srt(segments):
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(f"{i}\n{fmt_time(s.start)} --> {fmt_time(s.end)}\n{s.text.strip()}\n")
    return "\n".join(lines)

def media_duration_seconds(path: str) -> float:
    """Legge la durata con ffprobe (in secondi)."""
    try:
        info = ffmpeg.probe(path)
        return float(info.get("format", {}).get("duration", 0.0))
    except Exception:
        return 0.0

def convert_to_wav_with_progress(src_path: str, progress):
    """Converte a WAV mono 16kHz mostrando l'avanzamento reale via ffmpeg."""
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_out.close()
    total = media_duration_seconds(src_path)
    # Avvia ffmpeg con stream di progresso su stdout
    process = (
        ffmpeg
        .input(src_path)
        .output(tmp_out.name, ac=1, ar=16000)
        .global_args("-progress", "pipe:1", "-nostats")
        .overwrite_output()
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    # Loop su linee di progresso: cerchiamo "out_time_ms="
    while True:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            continue
        try:
            line = line.decode("utf-8", errors="ignore").strip()
        except Exception:
            line = ""
        if line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=")[1])
                if total > 0:
                    pct = min(100, int((ms / (total * 1_000_000)) * 100))
                    progress.progress(pct, text=f"Conversione… {pct}%")
            except Exception:
                pass
        elif line.startswith("progress=") and line.split("=")[1] == "end":
            progress.progress(100, text="Conversione completata ✅")
            break

    process.wait()
    return tmp_out.name

# ---------------- UI ----------------
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

    # Conversione con avanzamento reale
    st.subheader("1) Conversione con ffmpeg")
    conv_progress = st.progress(0, text="Preparazione…")
    try:
        wav_path = convert_to_wav_with_progress(src_path, conv_progress)
    except Exception as e:
        conv_progress.empty()
        st.error(f"Errore nella conversione con ffmpeg: {e}")
        st.stop()

    # Trascrizione con avanzamento (in base al tempo dei segmenti)
    st.subheader("2) Trascrizione")
    trans_progress = st.progress(0, text="Carico modello…")
    try:
        from time import time
        model = WhisperModel(model_size, device="cpu", compute_type=compute)
        segments_gen, info = model.transcribe(
            wav_path,
            language=None if language == "auto" else language,
            vad_filter=vad,
        )
        total = max(1e-6, float(info.duration or 0.0))
        segments, last_end = [], 0.0
        for seg in segments_gen:
            segments.append(seg)
            last_end = getattr(seg, "end", last_end)
            pct = min(100, int((last_end / total) * 100))
            trans_progress.progress(pct, text=f"Trascrizione… {pct}%")
        trans_progress.progress(100, text="Trascrizione completata ✅")

        text = "".join(s.text for s in segments).strip()
        srt_text = to_srt(segments)

        st.success(f"Fatto! Durata: {info.duration:.1f}s – Segmenti: {len(segments)}")
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
        trans_progress.empty()
        st.error(f"Errore nella trascrizione: {e}")
    finally:
        # Pulizia file temporanei
        try: os.unlink(src_path)
        except Exception: pass
        try: os.unlink(wav_path)
        except Exception: pass

else:
    st.caption("Carica un file per iniziare. Durante conversione e trascrizione vedrai l'avanzamento in tempo reale.")
