# Trascrizione audio by Roberto M.

App Streamlit per **trascrivere file audio/video** con [faster-whisper] (implementazione veloce di Whisper).  
Pensata per essere **semplice**: carichi un file, scegli (se vuoi) un paio di opzioni e ottieni **testo pulito**, più i sottotitoli SRT/VTT se li desideri.

> ✅ Timestamp **disattivati di default**  
> ✅ **Migliora e formatta** automaticamente il testo (opzionale)  
> ✅ Funziona anche su CPU con poca RAM (modelli piccoli/medi)

---

## ✨ Funzionalità

- **Upload** di file audio/video (fino a 200 MB su Streamlit Cloud).
- **Formati supportati**: MP3, WAV, M4A, MP4, AAC, FLAC, OGG, WMA, WEBM, MPEG4 (grazie a FFmpeg).
- **Scelta del modello** Whisper (tiny, base, small, medium).  
  > Il modello **large** è **disabilitato** per evitare crash su ambienti con RAM limitata.
- **Lingua**: rilevamento automatico (o forzatura manuale).
- **Barra di avanzamento** con fasi chiare (caricamento modello → preparazione file → trascrizione).
- **Esportazioni**:
  - `trascrizione.txt` (trascrizione grezza)
  - `testo_pulito.txt` (se attivi “Migliora e formatta automaticamente”)
  - `sottotitoli.srt` / `sottotitoli.vtt` (solo se spunti “Includi timestamp”)
- **Formato migliorato**: punteggiatura, spaziature, rimozione di ripetizioni, virgolette uniformate.
- **Suggeritore di prompt** per il miglioramento testo (puoi personalizzare lo stile desiderato).

---

## 🧰 Requisiti

- **Python 3.10+**
- **FFmpeg** installato nel sistema
- Dipendenze Python in `requirements.txt`
- (Su Streamlit Cloud) dipendenze APT in `packages.txt`  

Il progetto usa `faster-whisper` con `compute_type=int8` per ridurre consumo di memoria.

---

## 🚀 Avvio rapido (locale)

```bash
# 1) Clona il repo
git clone https://github.com/<tuo-utente>/trascrivi.git
cd trascrivi

# 2) Crea un virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3) Installa FFmpeg
# Linux (Debian/Ubuntu):
sudo apt-get update && sudo apt-get install -y ffmpeg
# macOS (Homebrew): brew install ffmpeg
# Windows (choco):  choco install ffmpeg

# 4) Installa le dipendenze Python
pip install -r requirements.txt

# 5) Avvia l'app
streamlit run trascrizione.py
```

Apri il link locale che Streamlit stampa in console.

---

## 🖱️ Come si usa

1. **Carica** un file audio/video.
2. (Opzionale) Apri **Opzioni avanzate** per forzare la lingua o cambiare modello.
3. Spunta:
   - **Includi timestamp** (se vuoi SRT/VTT).
   - **Migliora e formatta automaticamente** (per ottenere il testo pulito immediatamente).
4. Clicca **Avvia trascrizione**.
5. Scarica i file prodotti dai pulsanti che compaiono a fine elaborazione.

> 💡 **Suggerimento di prompt** (nelle opzioni del miglioramento):  
> “Rendi il testo scorrevole in italiano, correggi refusi e punteggiatura, rimuovi ripetizioni e intercalari, mantieni il senso originale senza aggiungere contenuti.”

---

## 🧪 Modelli disponibili & memoria

| Modello | Qualità | RAM indicativa* | Note |
|---|---|---:|---|
| tiny   | bassa   | ~1 GB | molto veloce |
| base   | media- | ~1.5–2 GB | default robusto |
| small  | media  | ~3–4 GB | buon compromesso |
| medium | alta   | ~6–8 GB | più lento |
| large  | molto alta | **>10 GB** | **DISABILITATO** |

\* Stime indicative su CPU con `int8`. Variano per durata/bitrate dell’audio.

---

## 🔐 Privacy

- I file vengono gestiti **in memoria e in una cartella temporanea** durante l’elaborazione.
- L’app **non** salva i file o le trascrizioni lato server in modo persistente.
- Ricordati di non caricare contenuti per cui non hai i diritti.

---

## 🆘 Troubleshooting

**Pagina “Oh no.” / crash durante la trascrizione**  
- Tipicamente è **Out-Of-Memory**. Prova:
  - Un **modello più piccolo** (base/small).
  - **Riduci** la durata del file (taglia in parti da 20–30 min).
  - **Converti** l’audio a mono 16 kHz e bitrate moderato.

**Upload fallisce / limite 200 MB**  
- Comprimi o spezza l’audio. Su Streamlit Cloud il limite è fisso.

**Trascrizione lenta su CPU**  
- Usa `tiny` o `base`. Evita `medium` per file lunghi se non strettamente necessario.

---

## 📦 Struttura del repository

```
trascrivi/
├─ trascrizione.py        # App Streamlit
├─ requirements.txt       # Dipendenze Python
├─ packages.txt           # Dipendenze APT (FFmpeg, ecc.)
└─ README.md              # Questo file
```

---

## 🗺️ Roadmap (idee)

- Elaborazione **batch** di più file.
- **Diarizzazione** (speaker labeling) di base.
- Modalità **traduzione** (es. → IT).
- Evidenziazione temporale sincronizzata nel viewer.
- Esportazione DOCX/Markdown.

---

## 🙏 Riconoscimenti

- [Streamlit](https://streamlit.io/)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) (Whisper ottimizzato)
- [FFmpeg](https://ffmpeg.org/)

---

## 📄 Licenza

Aggiungi una licenza (es. MIT) al repo se intendi condividerlo/publicarlo.
