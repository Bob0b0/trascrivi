# trascrizione.py
import os
import re
import time
import math
import tempfile
from datetime import timedelta
import textwrap
from typing import Optional

import streamlit as st
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import ffmpeg

# ---------------------------
# Utilità di formattazione
# ---------------------------
def seconds_to_hms(seconds: float) -> str:
    seconds = max(0, float(seconds or 0.0))
    return str(timedelta(seconds=int(round(seconds))))

def seconds_to_minutes_label(seconds: float, decimals: int = 1) -> str:
    minutes = (seconds or 0.0) / 60.0
    return f"{minutes:.{decimals}f} min"

def safe_probe_duration(path: str) -> float:
    try:
        meta = ffmpeg.probe(path)
        d = float(meta["format"]["duration"])
        return max(0.0, d)
    except Exception:
        return 0.0

# ---------------------------
# Pulizia & rifinitura locale del testo
# ---------------------------
TS_LINE_RE = re.compile(r'^\s*\[\d{2}:\d{2}(?::\d{2})?\]\s*', flags=re.MULTILINE)

def remove_square_bracket_timestamps(text: str) -> str:
    return TS_LINE_RE.sub('', text)

def dedup_adjacent_words(text: str) -> str:
    return re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

def normalize_spacing_punct(text: str) -> str:
    t = text
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r' +([,;:.!?])', r'\1', t)
    t = re.sub(r'([,;:.!?])(?![\s\)\]\}])', r'\1 ', t)
    t = re.sub(r'([(\[{]) +', r'\1', t)
    t = re.sub(r' +([)\]}])', r'\1', t)
    t = re.sub(r'([.!?]){3,}', '…', t)
    t = re.sub(r'\s*\n\s*', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def split_sentences(text: str):
    t = re.sub(r'\s*\n\s*', ' ', text)
    t = re.sub(r'([.!?…])([^\s])', r'\1 \2', t)
    parts = re.split(r'(?<=[.!?…])\s+', t)
    return [p.strip() for p in parts if p.strip()]

def capitalize_sentences(parts):
    out = []
    for s in parts:
        s = s.strip()
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        out.append(s)
    return out

def wrap_lines(text: str, width: int = 100) -> str:
    wrapped = []
    for para in text.split("\n"):
        wrapped.append(textwrap.fill(para, width=width) if para.strip() else "")
    return "\n".join(wrapped)

def refine_text(
    text: str,
    remove_ts: bool = False,
    dedup_words: bool = True,
    fix_spacing: bool = True,
    capitalize: bool = True,
    line_break_each_sentence: bool = True,
    wrap_width: Optional[int] = 100,
) -> str:
    t = text or ""
    if remove_ts:
        t = remove_square_bracket_timestamps(t)
    if dedup_words:
        t = dedup_adjacent_words(t)
    if fix_spacing:
        t = normalize_spacing_punct(t)

    if line_break_each_sentence:
        parts = split_sentences(t)
        if capitalize:
            parts = capitalize_sentences(parts)
        t = "\n".join(parts)
    elif capitalize:
        parts = split_sentences(t)
        t = " ".join(capitalize_sentences(parts))

    if wrap_width and wrap_width > 20:
        t = wrap_lines(t, width=int(wrap_width))
    return t.strip()

# ---------------------------
# Download/caching modello
# ---------------------------
MODEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "faster_whisper")
os.makedirs(MODEL_CACHE, exist_ok=True)

MODEL_MB = {
    "tiny": 77,
    "base": 142,
    "small": 461,
    "medium": 1500,
    "large-v3": 3100,
}

def _repo_candidates(size: str):
    # Niente f-string per evitare problemi di copia: usiamo concatenazione.
    base = size.replace("_", "-")
    repo1 = "Systran/faster-whisper-" + base
    repo2 = "guillaumekln/faster-whisper-" + base
    return [repo1, repo2]

def predownload_model_dir(size: str) -> str:
    last_err = None
    for repo in _repo_candidates(size):
        try:
            local_dir = snapshot_download(
                repo_id=repo,
                cache_dir=MODEL_CACHE,
                resume_download=True,
                local_files_only=False,
            )
            return local_dir
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Download modello '{size}' non riuscito: {last_err}")

@st.cache_resource(show_spinner=False)
def get_model(size: str, compute_type: str) -> 'WhisperModel':
    # Garantisce che i file siano già in cache, poi istanzia il modello
    local_dir = predownload_model_dir(size)
    return WhisperModel(local_dir, device="auto", compute_type=compute_type)

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Trascrizione audio by Roberto M.", page_icon="📝", layout="centered")
st.title("Trascrizione audio by Roberto M.")

st.markdown(
    "Carica un file audio/video, scegli modello e opzioni, quindi avvia la trascrizione. "
    "Durante l’elaborazione vedrai una barra di avanzamento con la **stima del tempo residuo**."
)

# Sidebar — selezioni e guida
st.sidebar.header("⚙️ Impostazioni")

MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v3"]
model_size = st.sidebar.selectbox("Modello", MODEL_OPTIONS, index=1, help="Modelli piccoli = più veloci, meno accurati. Modelli grandi = più accurati, più lenti.")

PRECISION_OPTIONS = ["int8", "int8_float32", "int16", "float16", "float32"]
compute_type = st.sidebar.selectbox("Precisione (compute_type)", PRECISION_OPTIONS, index=0)

LANG_DISPLAY = [
    "Auto (rilevamento)",
    "it — Italiano",
    "en — English",
    "es — Español",
    "fr — Français",
    "de — Deutsch",
    "pt — Português",
]
lang_display = st.sidebar.selectbox("Lingua dell'audio", LANG_DISPLAY, index=0)
LANG_MAP = {
    "it — Italiano": "it",
    "en — English": "en",
    "es — Español": "es",
    "fr — Français": "fr",
    "de — Deutsch": "de",
    "pt — Português": "pt",
}
language = None if lang_display.startswith("Auto") else LANG_MAP.get(lang_display, None)

with st.sidebar.expander("ℹ️ Guida rapida (modelli, lingua, precisione)", expanded=False):
    st.markdown(
        """
**Quale modello scegliere**
- **tiny/base** → **molto veloci**, **meno accurati** → perfetti per bozze e file brevi.
- **small** → **buon compromesso** su CPU normali.
- **medium** → **più accurato**, **più lento**. Consigliato con **GPU**.
- **large-v3** → **massima qualità**, più pesante (RAM/VRAM), ideale per audio lunghi o difficili.

**Download vs velocità**
- Al **primo uso** il modello viene **scaricato** (una tantum). Influisce solo sull’**avvio**.
- La **velocità di trascrizione** dipende da **dimensione del modello** e **hardware** (GPU ≫ CPU).

**Consigli pratici**
- Se **conosci la lingua** dell’audio, **selezionala**; altrimenti lascia **Auto**.
- **Precisione**: lascia **Int8 (default)** su CPU per il miglior rapporto **velocità/qualità**.
        """
    )
st.sidebar.caption("Suggerimento: per file lunghi usa modelli piccoli su CPU, oppure una GPU per i modelli grandi.")

# Opzioni avanzate
with st.expander("Opzioni avanzate", expanded=False):
    word_timestamps = st.checkbox("Parole con timestamp (più lento)", value=False)
    beam_size = st.slider("Beam size", min_value=1, max_value=10, value=5)
    vad_filter = st.checkbox("VAD filter (migliora segmenti di parlato)", value=True)

uploaded = st.file_uploader(
    "Carica un file audio/video",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm"],
    accept_multiple_files=False,
)

start_button = st.button("▶️ Avvia trascrizione", type="primary", disabled=uploaded is None)

# Output placeholders
progress_bar = st.progress(0, text="In attesa del file…")
eta_box = st.empty()
status_box = st.empty()
output_box = st.empty()
metrics_box = st.empty()
error_box = st.empty()

final_text = ""
audio_seconds = 0.0

if start_button:
    if uploaded is None:
        error_box.error("Nessun file selezionato.")
    else:
        # Salva su file temporaneo
        try:
            suffix = os.path.splitext(uploaded.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
        except Exception as e:
            error_box.error(f"Impossibile salvare il file caricato: {e}")
            st.stop()

        total_duration = safe_probe_duration(tmp_path)

        # Carica o pre-scarica il modello con messaggio esplicito
        approx = MODEL_MB.get(model_size, None)
        size_hint = f" (~{approx} MB da scaricare la prima volta)" if approx else ""
        status_box.info(f"Caricamento modello **{model_size}** (compute_type: **{compute_type}**){size_hint}…")

        load_t0 = time.time()
        try:
            with st.spinner("Preparazione del modello…"):
                model = get_model(model_size, compute_type)
        except Exception as e:
            error_box.error(
                "Errore nel caricamento del modello. Prova con **tiny** o **base** e verifica la connessione di rete. "
                f"Dettagli: {e}"
            )
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            st.stop()
        load_elapsed = time.time() - load_t0

        # Avvio trascrizione
        status_box.info("Trascrizione in corso…")
        progress_bar.progress(0, text="Analisi iniziale…")
        eta_box.write("⏳ Stima tempo residuo: —")

        t0 = time.time()
        collected_text = []
        last_end = 0.0

        try:
            segments, info = model.transcribe(
                tmp_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
            )

            for seg in segments:
                if word_timestamps:
                    collected_text.append(seg.text.strip())
                else:
                    collected_text.append(f"[{seconds_to_hms(seg.start)}] {seg.text.strip()}")

                last_end = float(seg.end or 0.0)
                processed = last_end
                denom = total_duration if total_duration > 0 else max(last_end, 1e-6)
                frac = max(0.0, min(0.999, processed / denom))

                elapsed = time.time() - t0
                rtf = (elapsed / processed) if processed > 0.5 else None

                if rtf is not None and total_duration > 0:
                    est_total = rtf * total_duration
                    remaining = max(0.0, est_total - elapsed)
                    eta_box.write(f"⏳ Stima tempo residuo: **{seconds_to_minutes_label(remaining)}**")
                else:
                    eta_box.write("⏳ Stima tempo residuo: —")

                percent = min(99, int(frac * 100))
                progress_bar.progress(percent, text=f"Elaborazione… {percent}%")

            progress_bar.progress(100, text="Completato ✅")
            status_box.success(f"Trascrizione completata. Rilevata lingua: **{getattr(info, 'language', '—')}**")

        except Exception as e:
            error_box.error(f"Errore durante la trascrizione: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        final_text = "\n".join(collected_text).strip()
        if not final_text:
            final_text = "(Nessun parlato rilevato o testo vuoto)"

        output_box.text_area("Testo trascritto", value=final_text, height=350)

        total_elapsed = time.time() - t0
        audio_seconds = total_duration if total_duration > 0 else last_end
        rtf_final = (total_elapsed / audio_seconds) if audio_seconds > 0 else float("nan")

        metrics_box.info(
            f"**Durata del file audio:** {seconds_to_minutes_label(audio_seconds)}  \n"
            f"**Tempo impiegato (trascrizione):** {seconds_to_minutes_label(total_elapsed)}  \n"
            f"**Fattore tempo reale (RTF):** {rtf_final:.2f}×  \n"
            f"**Tempo caricamento modello:** {seconds_to_minutes_label(load_elapsed)}"
        )

        st.download_button("💾 Scarica testo (.txt)", data=final_text, file_name="trascrizione.txt", mime="text/plain")

# ---------------------------
# ✍️ Migliora & formatta testo (locale)
# ---------------------------
st.markdown("---")
st.subheader("✍️ Migliora & formatta testo")
st.caption("Pulizia locale (senza AI): rimuove timestamp, sistema punteggiatura/spazi, va a capo a fine periodo, capitalizza e consente il download del testo rivisto.")

col1, col2, col3 = st.columns(3)
with col1:
    opt_remove_ts = st.checkbox("Rimuovi timestamp [hh:mm:ss]", value=False)
with col2:
    opt_dedup = st.checkbox("Elimina ripetizioni ravvicinate", value=True)
with col3:
    opt_caps = st.checkbox("Capitalizza inizio frase", value=True)

col4, col5 = st.columns(2)
with col4:
    opt_linebreaks = st.checkbox("A capo ogni periodo", value=True)
with col5:
    wrap_w = st.slider("Larghezza di riga (wrapping)", 60, 140, 100)

src_text = st.text_area("Sorgente da rifinire", value="", height=220)
apply_btn = st.button("✨ Applica miglioramenti")

if apply_btn:
    refined = refine_text(
        src_text,
        remove_ts=opt_remove_ts,
        dedup_words=opt_dedup,
        fix_spacing=True,
        capitalize=opt_caps,
        line_break_each_sentence=opt_linebreaks,
        wrap_width=wrap_w,
    )
    st.text_area("Risultato rifinito", value=refined, height=300)
    st.download_button("💾 Scarica testo rivisto (.txt)", data=refined, file_name="trascrizione_rivista.txt", mime="text/plain")

# ---------------------------
# 🤖 Prompt consigliato per revisione AI
# ---------------------------
with st.expander("🤖 Vuoi una revisione 'di stile' con AI? Prompt consigliato (copia e incolla)", expanded=False):
    prompt_ai = """Agisci come un revisore editoriale in italiano.
Obiettivo: correggi refusi, sintassi e punteggiatura; elimina ripetizioni e intercalari; migliora la scorrevolezza senza alterare il contenuto; vai a capo a fine periodo; mantieni il registro neutro-professionale.

Restituisci SOLO il testo rivisto, senza commenti.

Testo da rivedere:
<<<
[INCOPIA QUI IL TESTO DA RIVEDERE]
>>>"""
    st.code(prompt_ai, language="text")
    st.caption("Suggerimento: incolla il testo trascritto o già rifinito nel blocco tra <<< e >>>.")
