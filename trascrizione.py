# trascrizione.py - v2.0 - Trascrizione Whisper offline con feedback visivo
import streamlit as st
import whisper
import tempfile
import os
import threading
import time

st.set_page_config(page_title="Trascrizione audio offline", page_icon="🎙️")
st.title("🎧 Trascrizione audio offline con Whisper")

# Caricamento file audio
uploaded_file = st.file_uploader("📤 Carica un file audio (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])

# Selezione del modello
model_size = st.selectbox("⚙️ Scegli il modello Whisper", ["tiny", "base", "small", "medium", "large"], index=1)

# Info sui modelli
with st.expander("ℹ️ Velocità stimata dei modelli (su CPU)"):
    st.markdown("""
| Modello | Velocità | Precisione |
|---------|----------|------------|
| tiny    | ⚡ Veloce (1-2 min) | ★★☆☆☆ |
| base    | 🕐 Media (3-8 min)  | ★★★☆☆ |
| small   | 🕐 Lenta (8-20 min) | ★★★★☆ |
| medium  | 🐢 Molto lenta      | ★★★★★ |
| large   | 🐢 Lentissima       | ★★★★★ |
    """)

if uploaded_file:
    # Salva file temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)

    # --- FASE 1: Caricamento modello ---
    with st.spinner(f"⚙️ Caricamento modello **{model_size}**..."):
        model = whisper.load_model(model_size)
    st.success(f"✅ Modello **{model_size}** caricato.")

    # --- FASE 2: Trascrizione con timer live ---
    st.markdown("### 🔍 Trascrizione in corso...")

    col1, col2 = st.columns([3, 1])
    with col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    with col2:
        timer_display = st.empty()

    # Messaggi rotativi per far capire che sta lavorando
    messages = [
        "Analisi del segnale audio...",
        "Riconoscimento parlato in corso...",
        "Elaborazione segmenti audio...",
        "Decodifica testo...",
        "Whisper sta lavorando, attendere...",
        "Quasi pronto... (i file grandi richiedono tempo)",
    ]

    result_container = {}
    error_container = {}

    def run_transcription():
        try:
            result_container["result"] = model.transcribe(tmp_path)
        except Exception as e:
            error_container["error"] = str(e)

    # Avvia trascrizione in thread separato
    thread = threading.Thread(target=run_transcription)
    thread.start()

    start_time = time.time()
    msg_index = 0
    pulse = 0
    pulse_chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]

    while thread.is_alive():
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        # Aggiorna messaggio ogni 4 secondi
        msg_index = int(elapsed / 4) % len(messages)

        # Barra di progresso simulata (sale fino a 95%, poi si ferma)
        simulated_progress = min(0.95, elapsed / (file_size_mb * 8))
        progress_bar.progress(simulated_progress)

        status_text.markdown(f"{pulse_chars[pulse % len(pulse_chars)]} *{messages[msg_index]}*")
        timer_display.markdown(f"⏱️ **{elapsed_str}**")

        pulse += 1
        time.sleep(0.2)

    # Trascrizione completata
    elapsed_total = time.time() - start_time
    elapsed_str = f"{int(elapsed_total // 60):02d}:{int(elapsed_total % 60):02d}"

    if "error" in error_container:
        progress_bar.empty()
        status_text.empty()
        timer_display.empty()
        st.error(f"❌ Errore durante la trascrizione: {error_container['error']}")
    else:
        progress_bar.progress(1.0)
        status_text.markdown("✅ Trascrizione completata!")
        timer_display.markdown(f"⏱️ **{elapsed_str}**")

        result = result_container["result"]

        st.success(f"🎉 Completato in **{elapsed_str}** — {len(result['text'].split())} parole trascritte.")
        st.text_area("📝 Testo trascritto", result["text"], height=300)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "💾 Scarica .txt",
                result["text"],
                file_name="trascrizione.txt",
                mime="text/plain"
            )
        with col_b:
            # Export con timestamp se disponibili
            if "segments" in result and result["segments"]:
                srt_lines = []
                for i, seg in enumerate(result["segments"], 1):
                    def fmt_time(t):
                        h = int(t // 3600)
                        m = int((t % 3600) // 60)
                        s = int(t % 60)
                        ms = int((t - int(t)) * 1000)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                    srt_lines.append(str(i))
                    srt_lines.append(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}")
                    srt_lines.append(seg["text"].strip())
                    srt_lines.append("")
                srt_content = "\n".join(srt_lines)
                st.download_button(
                    "🎬 Scarica .srt (sottotitoli)",
                    srt_content,
                    file_name="trascrizione.srt",
                    mime="text/plain"
                )

    os.remove(tmp_path)

else:
    st.warning("📎 Carica un file audio per iniziare.")
