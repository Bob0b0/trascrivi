# trascrizione.py
# Streamlit app per trascrivere e revisionare audio/video con faster-whisper.
# - Sidebar: modello/CPU/lingua/prompt iniziale
# - Revisione automatica del testo (default ON)
# - Versione formattata per lettura (wrap ~90 colonne)
# - Prompt suggerito per ulteriori miglioramenti con AI
# - Modello "large" disabilitato per limiti RAM su hosting condivisi

import os
import io
import re
import time
import textwrap
import tempfile
from datetime import timedelta
from typing import List, Tuple

import streamlit as st
from faster_whisper import WhisperModel

try:
    import ffmpeg
except Exception:
    ffmpeg = None


# ------------------------- CONFIG -------------------------

st.set_page_config(
    page_title="Trascrizione audio by Roberto M.",
    page_icon="📝",
    layout="wide",
)

st.title("Trascrizione audio by Roberto M.")


# ------------------------- UTILS -------------------------

def human_time(seconds: float) -> str:
    seconds = max(0, float(seconds))
    return str(timedelta(seconds=int(seconds)))


def probe_duration(file_path: str) -> float:
    """Ritorna durata (sec) via ffprobe, oppure 0 se non disponibile."""
    if ffmpeg is None:
        return 0.0
    try:
        meta = ffmpeg.probe(file_path)
        dur = meta.get("format", {}).get("duration", None)
        if dur is not None:
            return float(dur)
        for s in meta.get("streams", []):
            if "duration" in s and s["duration"] is not None:
                return float(s["duration"])
    except Exception:
        pass
    return 0.0


# ------------------------- PULIZIA & REVISIONE -------------------------

def normalize_spacing_punct(t: str) -> str:
    """Spazi/punteggiatura: rimuove doppi spazi, evita spazi prima di ,.;:?! e garantisce uno spazio dopo."""
    t = t.replace(" \n", "\n").replace("\n ", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    # niente spazio prima dei segni
    t = re.sub(r"\s+([,.;:?!])", r"\1", t)
    # uno spazio dopo ,.;:?! se segue parola
    t = re.sub(r"([,.;:?!])([^\s\n])", r"\1 \2", t)
    # parentesi
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    # punti multipli -> ellissi compatta
    t = re.sub(r"\.{3,}", "…", t)
    # punti esclamativi o interrogativi multipli -> singolo
    t = re.sub(r"([!?])\1{1,}", r"\1", t)
    # vai a capi multipli -> massimo 2
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def dedupe_words(t: str) -> str:
    """Rimuove ripetizioni immediatamente consecutive di parole (case-insensitive)."""
    def repl(m):
        return m.group(1)
    # esempio: "che che", "grazie grazie"
    return re.sub(r"\b([A-Za-zÀ-ÖØ-öø-ÿ0-9']+)(?:\s+\1\b)+", repl, t, flags=re.IGNORECASE)


def remove_filler_it(t: str) -> str:
    """Elimina riempitivi comuni (italiano) in forma isolata o ripetuta."""
    fillers = r"(?:ehm|eh|uhm|uh|mmh|cioè|diciamo|allora)"
    # solo se isolati o seguiti/preceduti da pause
    t = re.sub(rf"\b{fillers}\b[,\.…]*\s*", "", t, flags=re.IGNORECASE)
    # ripetizioni tipo "eh eh eh"
    t = re.sub(rf"(?:\b{fillers}\b[\s,\.…]*){{2,}}", "", t, flags=re.IGNORECASE)
    return t


def sentence_case_paragraphs(t: str) -> str:
    """Metti maiuscola a inizio paragrafo; non altera acronimi."""
    paras = [p.strip() for p in t.split("\n\n")]
    out = []
    for p in paras:
        if not p:
            out.append("")
            continue
        # trova prima lettera alfabetica
        chars = list(p)
        for i, ch in enumerate(chars):
            if ch.isalpha():
                chars[i] = ch.upper()
                break
        out.append("".join(chars))
    return "\n\n".join(out)


def tidy_text_basic(text: str) -> str:
    """Pulizia base (conservativa)."""
    t = normalize_spacing_punct(text)
    t = dedupe_words(t)
    t = sentence_case_paragraphs(t)
    return t


def tidy_text_stronger(text: str, language: str = "it") -> str:
    """Pulizia + micro-revisione (conservativa): ripetizioni, filler, frasi spezzate."""
    t = text

    # Unisci spezzature tipiche da sottotitoli (linee molto corte separate da newline)
    lines = [ln.strip() for ln in t.splitlines()]
    joined: List[str] = []
    buf: List[str] = []
    for ln in lines:
        if not ln:
            if buf:
                joined.append(" ".join(buf))
                buf = []
            joined.append("")
        else:
            buf.append(ln)
    if buf:
        joined.append(" ".join(buf))
    t = "\n".join(joined)

    # Pulizia generale
    t = normalize_spacing_punct(t)
    t = dedupe_words(t)

    # Filler
    if language.startswith("it"):
        t = remove_filler_it(t)

    # Rimuovi frasi duplicate adiacenti (identiche ignorando spazi)
    sentences = re.split(r"(?<=[.!?…])\s+", t)
    cleaned = []
    prev_norm = None
    for s in sentences:
        norm = re.sub(r"\s+", " ", s.strip().lower())
        if not norm:
            continue
        if norm == prev_norm:
            continue
        cleaned.append(s.strip())
        prev_norm = norm
    t = " ".join(cleaned)

    # Spaziatura e maiuscole finali
    t = normalize_spacing_punct(t)
    t = sentence_case_paragraphs(t)

    return t.strip()


def wrap_for_reading(t: str, width: int = 90) -> str:
    """Formattazione per lettura: paragrafi con a capo ~width (senza spezzare parole)."""
    paras = [p.strip() for p in t.split("\n\n")]
    wrapped = [
        textwrap.fill(p, width=width, break_long_words=False, break_on_hyphens=False) if p else ""
        for p in paras
    ]
    return "\n\n".join(wrapped).strip()


def to_srt(segments: List[Tuple[float, float, str]]) -> str:
    def ts(x: float) -> str:
        ms = int((x - int(x)) * 1000)
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, (start, end, txt) in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(txt.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_vtt(segments: List[Tuple[float, float, str]]) -> str:
    def ts(x: float) -> str:
        ms = int((x - int(x)) * 1000)
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = int(x % 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for (start, end, txt) in segments:
        lines.append(f"{ts(start)} --> {ts(end)}")
        lines.append(txt.strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


# ------------------------- SIDEBAR -------------------------

with st.sidebar:
    st.header("Impostazioni")

    model_size = st.selectbox(
        "Modello Whisper",
        options=["tiny", "base", "small", "medium"],  # large disabilitato
        index=1,
        help="Modelli più grandi = migliore qualità ma più tempo/RAM.",
    )

    compute_choice = st.selectbox(
        "Calcolo (CPU)",
        options=["int8 (consigliato)", "float32 (più qualità, più RAM)"],
        index=0,
        help="Lascia int8 se sei su hosting con RAM limitata.",
    )
    compute_type = "int8" if compute_choice.startswith("int8") else "float32"

    language = st.selectbox(
        "Lingua dell'audio",
        options=["auto", "it", "en", "fr", "de", "es", "pt", "ro"],
        index=0,
        help="Se non sei sicuro, lascia **auto**.",
    )

    initial_prompt = st.text_area(
        "Prompt iniziale (opzionale)",
        value="",
        height=90,
        placeholder="Esempio: L'audio è in italiano; nomi propri: Paolo Ricca; usa punteggiatura naturale.",
    )


# ------------------------- OPZIONI PRINCIPALI -------------------------

with_ts = st.checkbox(
    "Includi timestamp nei file di output (SRT/VTT)",
    value=False,
    help="Disattivato di default. Abilitalo solo se ti servono i sottotitoli.",
)

auto_review = st.checkbox(
    "Revisiona e correggi automaticamente (consigliato)",
    value=True,
    help="Corregge punteggiatura, rimuove ripetizioni e frasi spezzate, pulisce i paragrafi.",
)

format_for_reading = st.checkbox(
    "Fornisci anche versione formattata per lettura (a capo ~90 col.)",
    value=True,
)


# ------------------------- UPLOAD -------------------------

st.subheader("Carica un file audio/video")
uploaded = st.file_uploader(
    "Drag and drop file here",
    type=["mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm", "mpeg4"],
    accept_multiple_files=False,
    label_visibility="collapsed",
)

start_btn = st.button("Avvia trascrizione", type="primary", disabled=(uploaded is None))
status_box = st.container()


# ------------------------- ELABORAZIONE -------------------------

if start_btn and uploaded is not None:
    t0 = time.time()

    with status_box:
        st.write("### Inizio elaborazione…")
        status = st.status("Preparazione…", expanded=True)

        # 1) File temporaneo
        status.update(label="Preparazione file…", state="running")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        duration = probe_duration(tmp_path)
        if duration > 0:
            st.info(f"**File:** {uploaded.name} — **{file_size_mb:.1f} MB** — **Durata:** {human_time(duration)}")
        else:
            st.info(f"**File:** {uploaded.name} — **{file_size_mb:.1f} MB** — **Durata:** sconosciuta")

        # 2) Modello
        status.update(label=f"Carico il modello `{model_size}` (CPU {compute_type})…", state="running")
        model = load_model(model_size, compute_type=compute_type)
        status.write("Modello pronto.")

        # 3) Trascrizione + progress
        status.update(label="Trascrivo…", state="running")
        progress = st.progress(0)
        seg_list: List[Tuple[float, float, str]] = []

        transcribe_kwargs = dict(
            language=None if language == "auto" else language,
            beam_size=5,
            best_of=5,
            vad_filter=True,
            initial_prompt=initial_prompt or None,
        )

        total = max(duration, 1.0)
        try:
            segments, info = model.transcribe(tmp_path, **transcribe_kwargs)
            # Estrapola lingua per la revisione
            detected_lang = info.language or (language if language != "auto" else "it")
            for seg in segments:
                seg_list.append((seg.start, seg.end, seg.text.strip()))
                p = min(1.0, float(seg.end) / total) if total > 0 else 0.0
                progress.progress(int(p * 100))
            status.write(f"Lingua rilevata: **{info.language}** — confidenza **{info.language_probability:.2f}**")
        except Exception as e:
            status.update(label="Errore durante la trascrizione.", state="error")
            st.error(f"Si è verificato un errore: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            st.stop()

        raw_text = "\n".join(t for _, _, t in seg_list).strip()

        # 4) Revisione/formatting
        status.update(label="Revisione e formattazione…", state="running")
        if auto_review:
            revised = tidy_text_stronger(raw_text, language=(language if language != "auto" else "it"))
        else:
            revised = tidy_text_basic(raw_text)

        if format_for_reading:
            formatted = wrap_for_reading(revised, width=90)
        else:
            formatted = revised

        # 5) Export
        txt_raw = raw_text.encode("utf-8")
        txt_revised = revised.encode("utf-8")
        txt_formatted = formatted.encode("utf-8")
        srt_bytes = to_srt(seg_list).encode("utf-8") if with_ts else None
        vtt_bytes = to_vtt(seg_list).encode("utf-8") if with_ts else None

        elapsed = time.time() - t0
        rtf = (elapsed / duration) if duration > 0 else float("nan")
        status.update(label="Completato ✅", state="complete")
        st.success(
            f"Elaborazione completata in **{human_time(elapsed)}**"
            + (f" — RTF ~ **{rtf:.2f}x**" if duration > 0 else "")
        )

        # 6) Output & download
        st.subheader("Download")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Testo grezzo (.txt)",
                data=txt_raw,
                file_name=os.path.splitext(uploaded.name)[0] + "_grezzo.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Testo revisionato (.txt)",
                data=txt_revised,
                file_name=os.path.splitext(uploaded.name)[0] + "_revisionato.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c3:
            st.download_button(
                "Formattato per lettura (.txt)",
                data=txt_formatted,
                file_name=os.path.splitext(uploaded.name)[0] + "_formattato.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if with_ts:
            cc1, cc2 = st.columns(2)
            with cc1:
                st.download_button(
                    "Sottotitoli (.srt)",
                    data=srt_bytes,
                    file_name=os.path.splitext(uploaded.name)[0] + ".srt",
                    mime="application/x-subrip",
                    use_container_width=True,
                )
            with cc2:
                st.download_button(
                    "Sottotitoli (.vtt)",
                    data=vtt_bytes,
                    file_name=os.path.splitext(uploaded.name)[0] + ".vtt",
                    mime="text/vtt",
                    use_container_width=True,
                )

        st.subheader("Anteprima")
        tab1, tab2, tab3 = st.tabs(["Revisionato", "Formattato per lettura", "Grezzo"])
        with tab1:
            st.text_area("Testo revisionato", value=revised, height=320, label_visibility="collapsed")
        with tab2:
            st.text_area("Formattato per lettura", value=formatted, height=320, label_visibility="collapsed")
        with tab3:
            st.text_area("Testo grezzo", value=raw_text, height=320, label_visibility="collapsed")

        # 7) Prompt suggerito per ulteriore passaggio AI
        st.subheader("Prompt suggerito per miglioramento con AI")
        lang = (language if language != "auto" else (info.language or "it"))
        if lang.startswith("en"):
            suggested = (
                "You are an expert Italian editor. Improve the following transcript **without adding new content**.\n"
                "Goals: fix grammar and punctuation; remove repetitions and filler words; merge broken sentences; "
                "prefer clear, modern wording; keep proper names and numbers; preserve meaning and structure; "
                "return only the final, clean text in the same language.\n\n"
                "TEXT:\n<<<PASTE HERE>>>"
            )
        else:
            suggested = (
                "Agisci come **editor professionale**. Migliora la trascrizione **senza inventare**.\n"
                "Obiettivi: correggi sintassi e punteggiatura; elimina ripetizioni e riempitivi; unisci frasi spezzate; "
                "usa lessico chiaro e attuale; mantieni nomi propri, numeri e citazioni; preserva il significato; "
                "restituisci solo il testo finale in italiano, in paragrafi leggibili.\n\n"
                "TESTO:\n<<<INCOLLA QUI IL TESTO REVISIONATO DA RIFINIRE>>>"
            )
        st.code(suggested, language="markdown")

        # Clean temp
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
