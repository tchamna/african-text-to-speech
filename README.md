git git stat# AfricanSpeaks

Voice-to-Voice French ↔ Nufi Translator with Whisper Model Comparison

## Features
- Record or upload French audio, get Nufi translation
- Three-way Whisper model comparison (Small, Medium, Large-v3)
- Clickable Nufi words for pronunciation
- Auto-play full sentence audio for exact matches
- Multi-language support (French, English, Spanish, etc.)

## Quick Start
1. Clone the repo:
   ```sh
   git clone <your-repo-url>
   cd AfricanSpeaks
   ```
2. Create a Python virtual environment:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the app:
   ```sh
   python app.py
   ```
5. Open [http://127.0.0.1:5001](http://127.0.0.1:5001) in your browser.

## File Structure
- `app.py` — Flask backend
- `templates/index.html` — Frontend UI
- `assets/` — Phrasebook, word audio mapping, FAISS index
- `audio/word_dictionary/` — Individual word audio files
- `audio/nufi_phrasebook_audio/` — Full sentence audio files
- `requirements.txt` — Python dependencies

## Notes
- Large model files and audio assets are not tracked in git (see `.gitignore`).
- For best results, use Chrome or Edge.

## License
MIT
