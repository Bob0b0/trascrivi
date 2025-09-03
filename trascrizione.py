# -*- coding: utf-8 -*-
# Trascrizione audio (robusta) — Streamlit
# Include: progress bar, istruzioni, autore, split automatico >60', export TXT/SRT/VTT,
#          e POST-PROCESSING AI con prompt utente (OpenAI se OPENAI_API_KEY è presente).

import os, math, json, tempfile, subprocess, shlex, time, textwrap
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
from faster_whisper import WhisperModel

# ====== CONFIG ======
APP_NAME  = "Trascrizione audio (robusta)"
AUTHOR    = "Autore: Il tuo nome"
VERSION   = "v2.2"

MAX_MINUTES_ALLOWED   = 60        # soglia per proporre split
DEFAULT_CHUNK_MINUTES = 20        # split automatico
DEFAULT_OVERLAP_SEC   = 2
COMPUTE_TYPE          = "int8"    # CPU-friendly: "int8" o "int8_float16"
# =====================

st.set_page_config(page_title=APP_NAME, page_icon="🎧", layout="centered")

# ---- Header
col1, col2 = st.columns([1,1])
with col1:
    st.title(APP_NAME)
with col2:
    st.caption(f"{AUTHOR} · {VERSION}")

with st.expander("Istruzioni rapide"):
    st.markdown(
        "- Carica un file audio (mp3, m4a, aac, wav, flac, ogg, opus, wma).\n"
        "- Se supera 60 minuti l’app **propone lo split** in blocchi (copia stream, no ricodifica).\n"
        "- Trascrizione con **faster-whisper**. Al termine scarichi **TXT / SRT / VTT**.\n"
        "- Se vuoi, puoi **post-processare** il testo con un’AI usando **il tuo prompt**.\n"
    )

with st.popover("Guida completa"):
    st.markdown(
        """
        **Pipeline**
        1) Verifica `ffmpeg`/`ffprobe`  
        2) Lettura durata + eventuale **split** (blocchi da N minuti con overlap)  
        3) Trascrizione per blocco (barra di avanzamento per blocco e globale)  
        4) Merge segmenti + export TXT/SRT/VTT  
        5) *(Opzionale)* **Post-processing AI**: applichi un **prompt** alla trascrizione completa.

        > L’AI usa OpenAI solo se troviamo `OPENAI_API_KEY` nell’ambiente.
        """
    )

# ---- Sidebar (essenziale)
st.sidebar.header("Impostazioni")
model_size = st.sidebar.selectbox(
    "Modello Whisper",
    ["tiny", "base", "small", "medium", "large-v3"],
    index=0,
    help="Più grande = migliore ma più lento."
)
language = st.sidebar.selectbox(
    "Lingua (o auto)",
    ["auto", "it", "en", "fr", "de", "es", "pt", "ro", "ru", "ar"],
    index=0
)

with st.sidebar.expander("Opzioni avanzate (di solito non servono)"):
    adv_chunk_min = st.number_input("Minuti per blocco (split, se necessario)",
                                    min_value=5, max_value=60,
                                    value=DEFAULT_CHUNK_MINUTES, step=5)
    adv_overlap = st.number_input("Overlap tra blocchi (secondi)",
                                  min_value=0, max_value=10,
                                  value=DEFAULT_OVERLAP_SEC, step=1)

# ====================== Utility ======================

@st.cache_data(show_spinner=False)
def which(cmd: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["bash", "-lc", f"command -v {shlex.quote(cmd)}"], text=True).strip()
        return out or None
    except Exception:
        return None

def ensure_ffmpeg() -> None:
    if not which("ffmpeg") or not which("ffprobe"):
        st.error("ffmpeg/ffprobe non disponibili nel sistema.")
        st.stop()

def run(cmd: list[str]) -> Tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out.strip(), err.strip()

def get_duration_sec(path: str) -> float:
    code, out, err = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", path
    ])
    if code == 0 and out:
        try:
            return float(out)
        except Exception:
            pass
    return 0.0

def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h: return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def split_audio(
    input_path: str, chunk_minutes: int, overlap_sec: int, out_dir: Path, progress=None
) -> list[Path]:
    duration = get_duration_sec(input_path)
    chunk_sec = chunk_minutes * 60
    if duration <= chunk_sec:
        return [Path(input_path)]

    out_paths: list[Path] = []
    start = 0
    idx = 0
    total_chunks = math.ceil(duration / chunk_sec)
    while start < duration - 1:  # margine
        end = min(start + chunk_sec, duration)
        out_path = out_dir / f"part_{idx:03d}{Path(input_path).suffix}"
        # -ss prima dell'input + -to durate -> copia stream
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start}",
            "-i", input_path,
            "-to", f"{end - start}",
            "-c", "copy",
            "-y", str(out_path)
        ]
        code, _, err = run(cmd)
        if code != 0:
            raise RuntimeError(f"Split fallito al chunk {idx}: {err}")
        out_paths.append(out_path)
        idx += 1
        start = end - overlap_sec  # overlap
        if progress:
            progress.progress(min(idx / total_chunks, 1.0))
    return out_paths

# ----- Trascrizione -----

def transcribe_file(
    audio_path: str,
    model_size: str,
    language: str = "auto",
    initial_prompt: Optional[str] = None,
) -> Tuple[list[dict], str]:
    """
    Ritorna (segments, full_text)
    segments: [{start, end, text}]
    """
    model = WhisperModel(model_size, compute_type=COMPUTE_TYPE)
    params = dict(
        vad_filter=True,
        beam_size=5,
        best_of=5,
    )
    if language != "auto":
        params["language"] = language
    if initial_prompt:
        params["initial_prompt"] = initial_prompt

    segs: list[dict] = []
    full_txt: list[str] = []
    for s in model.transcribe(audio_path, **params)[0]:
        segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
        full_txt.append(s.text.strip())
    return segs, " ".join(full_txt).strip()

def offset_segments(segments: list[dict], offset: float) -> list[dict]:
    out = []
    for s in segments:
        out.append({
            "start": s["start"] + offset,
            "end": s["end"] + offset,
            "text": s["text"]
        })
    return out

# ----- Export -----

def to_srt(segments: list[dict]) -> str:
    def fmt(t):
        ms = int((t - int(t)) * 1000)
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt(s['start'])} --> {fmt(s['end'])}")
        lines.append(s["text"].strip())
        lines.append("")
    return "\n".join(lines)

def to_vtt(segments: list[dict]) -> str:
    def fmt(t):
        ms = int((t - int(t)) * 1000)
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        # WebVTT usa . anziché ,
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{fmt(s['start'])} --> {fmt(s['end'])}")
        lines.append(s["text"].strip())
        lines.append("")
    return "\n".join(lines)

# ====================== UI PRINCIPALE ======================

ensure_ffmpeg()
uploaded = st.file_uploader("Carica un file audio", type=[
    "mp3","m4a","aac","wav","flac","ogg","opus","wma"
])

if uploaded:
    with tempfile.TemporaryDirectory() as tmpd:
        src_path = str(Path(tmpd) / uploaded.name)
        with open(src_path, "wb") as f:
            f.write(uploaded.read())

        # durata + split opzionale
        dur_sec = get_duration_sec(src_path)
        st.info(f"Durata: **{human_time(dur_sec)}**")
        do_split = False
        chunk_min = adv_chunk_min
        overlap_s = adv_overlap

        if dur_sec / 60.0 > MAX_MINUTES_ALLOWED:
            st.warning(
                f"File più lungo di {MAX_MINUTES_ALLOWED} minuti. "
                f"Propongo **split automatico** in blocchi da {chunk_min} min (overlap {overlap_s}s)."
            )
            do_split = True

        parts = [Path(src_path)]
        split_bar = None
        if do_split:
            st.write("📎 Preparazione split…")
            split_bar = st.progress(0.0)
            out_dir = Path(tmpd) / "chunks"
            out_dir.mkdir(parents=True, exist_ok=True)
            parts = split_audio(src_path, chunk_min, overlap_s, out_dir, progress=split_bar)
            split_bar.progress(1.0)
            st.success(f"Creati {len(parts)} blocchi.")

        # trascrizione
        st.subheader("Trascrizione")
        block_bar = st.progress(0.0, text="In corso…")
        global_bar = st.progress(0.0)
        all_segments: list[dict] = []
        full_texts: list[str] = []
        total = len(parts)
        acc_offset = 0.0

        for i, p in enumerate(parts, 1):
            block_bar.progress(0.0, text=f"Blocco {i}/{total}")
            segs, txt = transcribe_file(
                str(p),
                model_size=model_size,
                language=language,
                initial_prompt=None  # qui NON è il prompt LLM: è il prompt di Whisper (lasciamo vuoto)
            )
            all_segments.extend(offset_segments(segs, acc_offset))
            full_texts.append(txt)
            # aggiorna offset (fine blocco - overlap)
            if segs:
                acc_offset = all_segments[-1]["end"]
            block_bar.progress(1.0, text=f"Blocco {i}/{total} completato")
            global_bar.progress(i / total)

        transcript_text = "\n".join(full_texts).strip()
        st.success("Trascrizione completata.")

        # --- Download trascrizioni ---
        st.download_button("Scarica TXT", transcript_text.encode("utf-8"),
                           file_name=f"{Path(uploaded.name).stem}.txt")
        srt_text = to_srt(all_segments)
        st.download_button("Scarica SRT", srt_text.encode("utf-8"),
                           file_name=f"{Path(uploaded.name).stem}.srt")
        vtt_text = to_vtt(all_segments)
        st.download_button("Scarica VTT", vtt_text.encode("utf-8"),
                           file_name=f"{Path(uploaded.name).stem}.vtt")

        st.divider()

        # ====================== POST-PROCESSING AI (PROMPT) ======================
        st.subheader("Post-processa con AI (opzionale)")
        st.caption("Applica un **tuo prompt** alla trascrizione completa (richiede `OPENAI_API_KEY`).")

        do_llm = st.toggle("Attiva post-processing AI", value=False)
        if do_llm:
            # UI prompt + modello
            prompt_llm = st.text_area(
                "Prompt per l’AI",
                placeholder="Esempio: \"Riassumi in 10 bullet e estrai to-do con responsabili e scadenze.\"",
                height=120
            )
            colA, colB = st.columns([1,1])
            with colA:
                llm_model = st.text_input("Modello OpenAI", value="gpt-4o-mini")
            with colB:
                temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.1)

            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                st.warning("Variabile d’ambiente `OPENAI_API_KEY` non trovata: impossibile usare l’AI.")
            else:
                # chunking difensivo per testi lunghi (per non saturare il contesto)
                max_chars = 12000  # prudenziale; si può alzare in base al modello
                chunks = [transcript_text[i:i+max_chars] for i in range(0, len(transcript_text), max_chars)]

                if st.button("Esegui post-processing"):
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=api_key)

                        partials: list[str] = []
                        with st.spinner("AI in esecuzione…"):
                            for idx, ch in enumerate(chunks, 1):
                                messages = [
                                    {"role":"system","content": "Sei un assistente che rielabora trascrizioni audio in italiano con rigore e sintesi."},
                                    {"role":"user","content": f"{prompt_llm}\n\n---\nTRASCRIZIONE (parte {idx}/{len(chunks)}):\n{ch}"}
                                ]
                                resp = client.chat.completions.create(
                                    model=llm_model,
                                    messages=messages,
                                    temperature=temperature,
                                )
                                out_text = resp.choices[0].message.content.strip()
                                partials.append(out_text)
                        final_llm = "\n\n".join(partials).strip()
                        st.success("Post-processing completato.")
                        st.text_area("Risultato AI", final_llm, height=300)
                        st.download_button("Scarica risultato AI (TXT)",
                                           final_llm.encode("utf-8"),
                                           file_name=f"{Path(uploaded.name).stem}.ai.txt")
                    except Exception as e:
                        st.error(f"Errore AI: {e}")
