# Trascrizione audio by Roberto M.

Trasforma **file audio/video** in **testo** con Whisper (implementazione rapida *faster-whisper*).  
Ottieni:
- **Testo grezzo**
- **Testo revisionato e formattato** (punteggiatura corretta, ripetizioni comuni rimosse, a capo ~90 col)
- **Sottotitoli** in **SRT/VTT** (opzionali)

👉 **Usa l’app qui:** https://trascrivi.streamlit.app

---

## Come si usa (30 secondi)

1. Apri l’app e (se serve) scegli in **Impostazioni**:
   - **Modello Whisper**: `base` consigliato (oppure `small`/`medium` per audio più lunghi).  
     `large` è **disabilitato** per motivi di RAM su Streamlit Cloud.
   - **Calcolo (CPU)**: `int8 (consigliato)`.
   - **Lingua dell’audio**: lascia **auto** se non sei sicuro.

2. **Carica** il tuo file (drag & drop o *Browse files*).  
   Formati: MP3, WAV, M4A, MP4, AAC, FLAC, OGG, WMA, WEBM, MPEG4.  
   Limite: ~200 MB per file.

3. Clicca **Avvia trascrizione**.  
   Vedi la barra di avanzamento e la stima dei tempi.

4. A fine elaborazione potrai **scaricare**:
   - `testo_grezzo.txt`
   - `testo_pulito.txt` (revisionato e formattato)
   - `sottotitoli.srt` e/o `sottotitoli.vtt` (se hai attivato i timestamp)

5. Trovi anche un **prompt suggerito** per rifinire il testo con un’altra IA (stile, lessico, sintassi).

---

## Opzioni principali

- **Timestamp SRT/VTT**: *disattivati di default*. Abilitali solo se ti servono i sottotitoli.
- **Revisione & formattazione**: *attiva di default*.  
  Corregge la punteggiatura, attenua ripetizioni tipiche del parlato, e impagina per una lettura più confortevole.
- **Prompt iniziale**: puoi fornire una breve frase che aiuta il modello (es. nomi propri o lingua specifica).  
  Non è obbligatorio.

---

## Privacy & sessione

- I file **non vengono salvati**: restano solo nella tua sessione di lavoro.  
- I risultati restano visibili fino a quando **non avvii una nuova trascrizione** o **chiudi la sessione**.

---

## Suggerimenti

- Per audio lunghi: usa modelli `small` o `medium` e il calcolo `int8`.
- Audio rumorosi migliorano molto con registrazioni più pulite o un pre‑processing (opzionale) in locale.
- Se vuoi solo leggere comodamente, scarica **testo_pulito.txt**.

---

## Problemi noti

- Modello `large` disabilitato su Streamlit Cloud per limiti di memoria.
- Se compare un errore di rete o cache dei modelli, riprova più tardi: la piattaforma potrebbe essere temporaneamente sotto carico.

---

## Per chi è curioso

Repository essenziale:
- `trascrizione.py` – codice dell’app Streamlit
- `requirements.txt` – dipendenze Python
- `packages.txt` – pacchetti di sistema (FFmpeg, ecc.)

Esecuzione **in locale** (Python 3.10+):
```bash
pip install -r requirements.txt
streamlit run trascrizione.py
```
