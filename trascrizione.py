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
    """Ritorna la durata in secondi usando ffmpeg.probe; 0 se non disponibile."""
    try:
        meta = ffmpeg.probe(path)
        d = float(meta["format"]["duration"])
        return max(0.0, d)
    except Exception:
        return 0.0


# ---------------------------
# Pulizia & rifinitura locale del testo
# ------
