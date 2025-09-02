# app_locale.py - v1.0 - Trascrizione Whisper offline
import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="Trascrizione audio offline", page_icon="🎙️")
st.title("🎧 Trascrizione audio offline con Whisper")

# Caricamento file audio
uploaded_file = st.file_uploader("📤 Carica un file audio (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

# Selezione del modello
model_size = st.selectbox("⚙️ Scegli il modello Whisper", ["tiny", "base", "small", "medium", "large"], index=1)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("⏳ Caricamento modello, attendere...")
    model = whisper.load_model(model_size)

    st.info("🔍 Trascrizione in corso...")
    result = model.transcribe(tmp_path)

    st.success("✅ Trascrizione completata!")
    st.text_area("📝 Testo trascritto", result["text"], height=300)
    st.download_button("💾 Scarica la trascrizione", result["text"], file_name="trascrizione.txt")

    os.remove(tmp_path)
else:
    st.warning("📎 Carica un file audio per iniziare.")
