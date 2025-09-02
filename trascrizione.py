
import os
import io
import re
import json
import time
from datetime import timedelta

import streamlit as st
from faster_whisper import WhisperModel
import ffmpeg
import requests

APP_TITLE = "Trascrizione audio by Roberto M."

# ---------------------------
# Utilities
# ---------------------------
def human_time(seconds: float) -> str:
    if seconds is None:
        return "N/D"
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def probe_duration(file_path: str) -> float | None:
    try:
        info = ffmpeg.probe(file_path)
        return float(info["format"]["duration"])
    except Exception:
        return None

def format_timestamp_srt(t: float) -> str:
    # 00:00:00,000
    ms = int(round((t - int(t)) * 1000))
    td = timedelta(seconds=int(t))
    base = str(td)
    if td.days > 0:
        # timedelta with days prints like "1 day, HH:MM:SS"
        base = base.split(", ")[-1]
    if len(base) == 7:  # M:SS
        base = "0:" + base
    if len(base) == 4:  # SSS?
        base = "00:0" + base
    if len(base) == 5:  # M:SS but missing hour
        base = "0:" + base
    # Ensure HH:MM:SS
    parts = base.split(":")
    if len(parts) == 2:
        base = "00:" + base
    return f"{base},{ms:03d}"

def clean_text_basic(text: str) -> str:
    """Lightweight local improvement: dedup words, fix spaces/punct, capitalize sentence starts, add line breaks."""
    if not text or not text.strip():
        return text

    # Normalize whitespace
    t = re.sub(r"\s+", " ", text).strip()

    # Remove repeated words (up to 3 repetitions)
    t = re.sub(r"\b(\w+)(\s+\1){1,3}\b", r"\1", t, flags=re.IGNORECASE)

    # Fix spacing around punctuation
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?])(?!\s)", r"\1 ", t)

    # Collapse multiple punctuation
    t = re.sub(r"([.!?]){2,}", r".", t)
    t = re.sub(r",,", r",", t)

    # Split into sentences and capitalize
    sentences = re.split(r"(?<=[.!?])\s+", t)
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]

    # Join with paragraph breaks every 2-3 sentences
    out_lines = []
    for i, s in enumerate(sentences, start=1):
        out_lines.append(s)
        if i % 3 == 0:
            out_lines.append("")  # blank line
    out = "\n".join(out_lines).strip()

    # Ensure reasonable line lengths
    wrapped = []
    for para in out.split("\n"):
        if not para.strip():
            wrapped.append("")
            continue
        line = []
        count = 0
        for word in para.split():
            if count + len(word) + 1 > 110:
                wrapped.append(" ".join(line))
                line = [word]
                count = len(word)
            else:
                line.append(word)
                count += len(word) + 1
        if line:
            wrapped.append(" ".join(line))
    return "\n".join(wrapped).strip()

def improve_with_openai(text: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Optional AI improvement via OpenAI if API key is provided. Returns improved text or raises."""
    prompt = (
        "Sei un editor italiano. Pulisci e migliora il seguente testo di trascrizione: "
        "correggi ortografia e punteggiatura, elimina ripetizioni ed esitazioni, mantieni il significato, "
        "aggiungi capoversi logici (ogni 2–3 frasi) e rendi la prosa scorrevole. Non inventare contenuti."
        "\n\nTesto da migliorare:\n"
    )
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Sei un assistente utile."},
            {"role": "user", "content": prompt + text},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(payload), timeout=180)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

@st.cache_resource(show_spinner=False)
def load_model(model_size: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, compute_type=compute_type)

def make_transcript_text(segments, with_timestamps: bool) -> tuple[str, str | None]:
    """Return plain_text and srt_text (if timestamps)."""
    plain_parts = []
    srt_parts = []
    if with_timestamps:
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            plain_parts.append(text)
            start = float(seg.start or 0.0)
            end = float(seg.end or 0.0)
            srt_parts.append(str(idx))
            srt_parts.append(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}")
            srt_parts.append(text)
            srt_parts.append("")  # blank line
            idx += 1
        return "\n".join(plain_parts).strip(), "\n".join(srt_parts).strip()
    else:
        for seg in segments:
            plain_parts.append(seg.text.strip())
        return "\n".join(plain_parts).strip(), None

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🎙️", layout="centered")
st.title(APP_TITLE)

st.write(
    "Carica un file audio/video, scegli modello e opzioni, quindi avvia la trascrizione. "
    "Durante l'elaborazione vedrai una barra di avanzamento con la stima del tempo residuo."
)

with st.expander("Opzioni avanzate", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        model_size = st.selectbox(
            "Modello Whisper",
            options=["tiny", "base", "small", "medium", "large-v2"],
            index=1,
            help="Modello più piccolo = più veloce; più grande = migliore qualità."
        )
    with col2:
        compute_type = st.selectbox(
            "Precisione calcolo (compute_type)",
            options=["int8", "int8_float16", "float16", "float32"],
            index=0,
            help="Lascia **int8** per impostazione predefinita: è più veloce e spesso sufficiente."
        )

    lang = st.selectbox(
        "Lingua del parlato",
        options=["auto", "it", "en", "fr", "de", "es", "pt"],
        index=0,
        help="Consiglio: se il file NON è in italiano, seleziona la lingua giusta per ridurre gli errori. "
             "Se non sei sicuro, lascia **auto**."
    )

    with_ts = st.checkbox("Includi timestamp nei file di output (SRT/VTT)", value=False,
                          help="Disattivato di default. Abilitalo solo se ti servono i timecode.")
    auto_improve = st.checkbox("Migliora e formatta automaticamente al termine", value=True)

    method = st.radio(
        "Metodo di miglioramento",
        options=["Regole locali (senza AI)", "AI (OpenAI)"],
        index=0,
        help="Con l'AI ottieni una riscrittura migliore (serve una API key). Altrimenti si applica una pulizia locale."
    )
    openai_key = ""
    openai_model = "gpt-4o-mini"
    if method == "AI (OpenAI)":
        openai_key = st.text_input("OpenAI API Key (facoltativa)", type="password")
        openai_model = st.text_input("Modello OpenAI", value="gpt-4o-mini")

uploaded = st.file_uploader(
    "Carica un file audio/video",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
)

# Session state for outputs
state = st.session_state
if "raw_text" not in state: state.raw_text = ""
if "improved_text" not in state: state.improved_text = ""
if "srt_text" not in state: state.srt_text = None
if "media_duration" not in state: state.media_duration = None
if "elapsed" not in state: state.elapsed = 0

start_btn = st.button("▶️ Avvia trascrizione", type="primary", disabled=uploaded is None)

if start_btn and uploaded is not None:
    # Save temp file
    tmp_dir = st.experimental_get_query_params().get("tmp_dir", ["/tmp"])[0]
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, uploaded.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Duration from ffprobe
    duration = probe_duration(tmp_path)
    state.media_duration = duration

    # Status areas
    eta_box = st.info(f"Caricamento modello **{model_size}** (compute_type: **{compute_type}**)…")
    prog = st.progress(0, text="Inizializzazione…")
    t0 = time.time()

    # Load model
    model = load_model(model_size=model_size, compute_type=compute_type)

    # Transcribe
    prog.progress(10, text="Trascrizione in corso…")
    language = None if lang == "auto" else lang

    # We iterate segments to build text and update progress if possible
    segments_iter, info = model.transcribe(
        tmp_path,
        language=language,
        vad_filter=True,
        word_timestamps=False,  # default senza timestamp parola
        beam_size=5,
    )

    segments = []
    processed_time = 0.0
    last_update = time.time()
    for seg in segments_iter:
        segments.append(seg)
        processed_time = float(seg.end or processed_time)
        # Update progress based on media progress if duration known
        if duration and duration > 0:
            pct = min(99, int(processed_time / duration * 100))
            now = time.time()
            if now - last_update > 0.15:
                remaining = max(0.0, (duration - processed_time) / max(1e-6, processed_time) * (now - t0)) if processed_time > 1 else None
                if remaining is not None:
                    eta_box.info(f"⏳ Stima tempo residuo: **{human_time(remaining)}**")
                prog.progress(pct, text=f"Elaborazione… {pct}%")
                last_update = now

    # Build outputs
    plain_text, srt_text = make_transcript_text(segments, with_timestamps=with_ts)
    state.raw_text = plain_text
    state.srt_text = srt_text

    # Auto-improve if requested
    improved = ""
    if auto_improve:
        try:
            if method == "AI (OpenAI)" and openai_key:
                improved = improve_with_openai(plain_text, api_key=openai_key, model=openai_model)
            else:
                improved = clean_text_basic(plain_text)
        except Exception as e:
            st.warning(f"Impossibile applicare il miglioramento automatico: {e}")
            improved = clean_text_basic(plain_text)
    state.improved_text = improved or ""

    # Done
    elapsed = time.time() - t0
    state.elapsed = elapsed
    prog.progress(100, text="Completato!")
    eta_box.success("Trascrizione completata.")

# ---------------------------
# Output area
# ---------------------------
if state.raw_text:
    st.subheader("Esito trascrizione")

    tabs = st.tabs(["Testo grezzo", "Testo migliorato"])

    with tabs[0]:
        st.caption("Trascrizione senza timecode (di default).")
        st.text_area("Output (grezzo)", value=state.raw_text, height=300)
        st.download_button("⬇️ Scarica .txt (grezzo)", data=state.raw_text, file_name="trascrizione_grezza.txt", mime="text/plain")

        if state.srt_text:
            col_srt, col_vtt = st.columns(2)
            with col_srt:
                st.download_button("⬇️ Scarica .srt", data=state.srt_text, file_name="trascrizione.srt", mime="text/plain")
            with col_vtt:
                # Simple VTT from SRT
                vtt = "WEBVTT\n\n" + state.srt_text.replace(",", ".")
                st.download_button("⬇️ Scarica .vtt", data=vtt, file_name="trascrizione.vtt", mime="text/vtt")

        # Manual improve action (no copia/incolla)
        st.markdown("—")
        if st.button("✨ Applica miglioramento ora (senza AI)" if "AI" not in st.session_state.get("method", "x") else "✨ Applica miglioramento ora"):
            try:
                if method == "AI (OpenAI)" and openai_key:
                    state.improved_text = improve_with_openai(state.raw_text, api_key=openai_key, model=openai_model)
                else:
                    state.improved_text = clean_text_basic(state.raw_text)
                st.success("Miglioramento applicato.")
            except Exception as e:
                st.error(f"Errore durante il miglioramento: {e}")

    with tabs[1]:
        improved_display = state.improved_text if state.improved_text else "— (non ancora applicato)"
        st.text_area("Output (migliorato)", value=improved_display, height=300)
        if state.improved_text:
            st.download_button("⬇️ Scarica .txt (migliorato)", data=state.improved_text, file_name="trascrizione_migliorata.txt", mime="text/plain")

    # Info box with durations
    st.info(
        f"**Durata del file audio:** {human_time(state.media_duration)} — "
        f"**Tempo impiegato:** {human_time(state.elapsed)}"
    )
else:
    st.caption("Suggerimento: carica un file, lascia i timecode disattivati (default) e attiva 'Migliora e formatta automaticamente' per ottenere subito un testo pulito.")

