import streamlit as st
from faster_whisper import WhisperModel
import tempfile, os, datetime
import ffmpeg

st.set_page_config(page_title="Trascrizione audio", page_icon="🎙️")
st.title("🎧 Trascrizione audio con faster-whisper (CPU)")

# Upload
uploaded_file = st.file_uploader(
    "📤 Carica un file audio/video (MP3, WAV, M4A, MP4...)",
    type=["mp3","wav","m4a","aac","ogg","flac","mp4","mov","mkv","webm","avi"],
)

# Modello e opzioni
model_size = st.selectbox("⚙️ Modello", ["tiny", "base", "small", "medium"], index=2)
language = st.selectbox("🌍 Lingua", ["auto", "it", "en", "es", "fr", "de"], index=0)
compute = st.selectbox("🧮 Precisione (CPU)", ["int8", "int8_float16", "float32"], index=0)
vad = st.checkbox("🧹 Applica VAD (rimuove silenzi)", value=True)

def convert_to_wav(src_path):
    """Converte qualunque formato in WAV mono 16kHz temporaneo"""
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_out.close()
    (
        ffmpeg
        .input(src_path)
        .output(tmp_out.name, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return tmp_out.name

def to_srt(segments):
    def fmt(ts):
        td = datetime.timedelta(seconds=float(ts))
        # 00:00:00,000
        return str(td)[:12].replace(".", ",").zfill(12)
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(f"{i}\n{fmt(s.start)} --> {fmt(s.end)}\n{s.text.strip()}\n")
    return "\n".join(lines)

if uploaded_file:
    # salva l'upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        src_path = tmp.name

    st.info("🔧 Converto il file con ffmpeg…")
    try:
        wav_path = convert_to_wav(src_path)
    except Exception as e:
        st.error(f"Errore nella conversione con ffmpeg: {e}")
        st.stop()

    st.info("🤖 Carico il modello e trascrivo…")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type=compute)
        segments, info = model.transcribe(
            wav_path,
            language=None if language == "auto" else language,
            vad_filter=vad,
        )
        segments = list(segments)
        text = "".join(s.text for s in segments).strip()
        srt_text = to_srt(segments)

        st.success(f"✅ Trascrizione completata! Durata: {info.duration:.1f}s")
        st.text_area("📝 Testo trascritto", text, height=300)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Scarica .txt", text, file_name="trascrizione.txt")
        with col2:
            st.download_button("⬇️ Scarica .srt", srt_text, file_name="trascrizione.srt", mime="application/x-subrip")

        with st.expander("📚 Segmenti"):
            for s in segments:
                st.write(f"[{s.start:.2f} → {s.end:.2f}] {s.text.strip()}")

    except Exception as e:
        st.error(f"Errore nella trascrizione: {e}")

    # pulizia
    try:
        os.unlink(src_path)
        os.unlink(wav_path)
    except Exception:
        pass
else:
    st.warning("📎 Carica un file audio/video per iniziare.")
