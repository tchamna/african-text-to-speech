import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import whisper
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
from datetime import datetime
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

app = Flask(__name__)
APP_MODE = os.environ.get('APP_MODE', 'development')
APP_MODE = os.environ.get('APP_MODE', 'production')
print(f"APP_MODE set to: {APP_MODE}")

# Azure Blob Storage base URLs for audio files (fallback when local files don't exist)
AZURE_BLOB_BASE_URL = "https://africanobjectaudio.blob.core.windows.net"
PHRASEBOOK_AUDIO_URL = f"{AZURE_BLOB_BASE_URL}/nufi-phrasebook-audio"
DICTIONARY_AUDIO_URL = f"{AZURE_BLOB_BASE_URL}/word-dictionary-audio"

# Check if audio files exist locally
LOCAL_AUDIO_DIR = 'audio'
USE_LOCAL_AUDIO = os.path.exists(LOCAL_AUDIO_DIR) and os.path.isdir(LOCAL_AUDIO_DIR)

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from local directory (for local development)"""
    if USE_LOCAL_AUDIO and os.path.exists(os.path.join(LOCAL_AUDIO_DIR, filename)):
        return send_from_directory(LOCAL_AUDIO_DIR, filename)
    # If file doesn't exist locally, redirect to Azure Blob Storage
    from flask import redirect
    if 'nufi_phrasebook' in filename:
        return redirect(f"{PHRASEBOOK_AUDIO_URL}/{os.path.basename(filename)}")
    else:
        return redirect(f"{DICTIONARY_AUDIO_URL}/{filename}")

def get_audio_url(audio_type, filename):
    """
    Get audio URL - returns local URL if files exist locally, otherwise Azure Blob URL.
    audio_type: 'phrasebook' or 'dictionary'
    """
    if USE_LOCAL_AUDIO:
        # Use local serving route
        if audio_type == 'phrasebook':
            return f'/audio/nufi_phrasebook_audio/{filename}'
        else:
            return f'/audio/word_dictionary/{filename}'
    else:
        # Use Azure Blob Storage URLs directly
        if audio_type == 'phrasebook':
            return f'{PHRASEBOOK_AUDIO_URL}/{filename}'
        else:
            return f'{DICTIONARY_AUDIO_URL}/{filename}'

# === WHISPER MODEL SETUP ===
# Use environment variable to select model size
# Options: tiny, base, small, medium, large, large-v2, large-v3
# Default: large-v3 for best accuracy (requires ~3GB RAM)
# For local dev with limited RAM, set WHISPER_MODEL_SIZE=small or medium
WHISPER_MODEL_SIZE = os.environ.get('WHISPER_MODEL_SIZE', 'large-v3')
print(f"Loading Whisper model: {WHISPER_MODEL_SIZE}...")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    print(f"   ✓ Whisper {WHISPER_MODEL_SIZE} loaded successfully!")
except Exception as e:
    print(f"Failed to load Whisper model '{WHISPER_MODEL_SIZE}': {e}")
    # Try falling back to a smaller model to avoid out-of-memory at startup
    fallback = os.environ.get('WHISPER_FALLBACK_MODEL', 'small')
    try:
        print(f"Attempting to load fallback Whisper model: {fallback}")
        whisper_model = whisper.load_model(fallback)
        WHISPER_MODEL_SIZE = fallback
        print(f"   ✓ Fallback Whisper {fallback} loaded successfully!")
    except Exception as e2:
        print(f"Critical: Failed to load fallback Whisper model '{fallback}': {e2}")
        raise

# In development, optionally load additional Whisper models for side-by-side comparison
whisper_models = {
    'large': whisper_model  # map "large" panel to the configured main model
}
# Only load additional models in development when explicitly requested via env var.
# This prevents accidental high-memory loads in constrained hosts (e.g., Azure App Service).
WHISPER_LOAD_EXTRA = os.environ.get('WHISPER_LOAD_EXTRA_MODELS', '0') == '1'
if APP_MODE != 'production' and WHISPER_LOAD_EXTRA:
    try:
        print("Loading additional Whisper models for comparison (development mode): small, medium ...")
        whisper_models['small'] = whisper.load_model('small')
        whisper_models['medium'] = whisper.load_model('medium')
        print("   ✓ Small and Medium models loaded for comparison")
    except Exception as e:
        print(f"Warning: Failed loading additional models for comparison: {e}")
else:
    if APP_MODE != 'production':
        print("Skipping loading additional Whisper models. Set WHISPER_LOAD_EXTRA_MODELS=1 to enable.")

# Load semantic search model and index
print("Loading semantic search model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Semantic search model loaded successfully.")

print("Loading FAISS index...")
faiss_index = faiss.read_index('assets/faiss_index.bin')
with open('assets/index_mapping.pkl', 'rb') as f:
    index_data = pickle.load(f)
print(f"FAISS index loaded: {faiss_index.ntotal} entries")

# --- Logging for rejected semantic candidates ---
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
rejected_log_path = os.path.join(LOG_DIR, 'rejected_semantic.log')
logger = logging.getLogger('african_voice')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(rejected_log_path, encoding='utf-8')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)


# Configuration
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load Phrasebook with row IDs for audio mapping
try:
    df = pd.read_csv('assets/Nufi_Francais_phrasebook.csv')
    # Rename Francais column to French for consistency
    if 'Francais' in df.columns:
        df.rename(columns={'Francais': 'French'}, inplace=True)
    
    # Add row_id column (1-based to match audio file numbering)
    # We do this BEFORE removing duplicates to preserve the mapping to the original audio files
    df['row_id'] = df.index + 1

    # Ensure consistency with the FAISS index (remove duplicates)
    # This MUST match the logic in tests/build_index.py
    df['French'] = df['French'].astype(str).str.strip() # Keep case for display, normalize later for search
    initial_count = len(df)
    df.drop_duplicates(subset=['French'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Ensure columns exist, normalize if needed
    
    # Ensure columns exist, normalize if needed
    # We keep the original 'French' column for display, and create a normalized one for search in find_closest_match
    # df['French'] = df['French'].astype(str).str.lower().str.strip() # This was overwriting the display text with lowercase!
    # Let's fix that too.
    
    print(f"Phrasebook loaded successfully: {len(df)} entries (removed {initial_count - len(df)} duplicates)")
except Exception as e:
    print(f"Error loading phrasebook: {e}")
    df = pd.DataFrame(columns=['French', 'Nufi', 'row_id'])

# Load Word Audio Dictionary
word_audio_dict = {}
word_audio_dict_lower = {}  # Case-insensitive lookup
try:
    word_audio_df = pd.read_csv('assets/nufi_dictionary_dictionnaire_audio_maping.csv')
    # Build dictionary: Nufi word -> audio filename (without .mp3 extension)
    for _, row in word_audio_df.iterrows():
        keyword = str(row['Keyword']).strip()
        audio_file = str(row['clafrica']).strip()
        if keyword and audio_file:
            audio_filename = audio_file + '.mp3'
            word_audio_dict[keyword] = audio_filename
            # Also store lowercase version for case-insensitive lookup
            word_audio_dict_lower[keyword.lower()] = audio_filename
    print(f"Word audio dictionary loaded: {len(word_audio_dict)} entries")
except Exception as e:
    print(f"Error loading word audio dictionary: {e}")

def parse_nufi_words_with_audio(nufi_text):
    """
    Parse Nufi sentence into words and look up audio files for each word.
    Returns list of {word, audio_url} dicts.
    """
    if not nufi_text:
        return []
    
    # Split by spaces and punctuation, but keep the words
    import re
    words = re.findall(r"[\w'̄́̀̂̌]+", nufi_text)  # Match word characters including diacritics
    
    result = []
    for word in words:
        word_clean = word.strip()
        if word_clean:
            # Try exact match first, then case-insensitive
            audio_file = word_audio_dict.get(word_clean)
            if not audio_file:
                audio_file = word_audio_dict_lower.get(word_clean.lower())
            
            result.append({
                'word': word_clean,
                'audio_url': get_audio_url('dictionary', audio_file) if audio_file else None
            })
    
    return result

def find_closest_match(text):
    """
    Hybrid search: Prioritize exact text matches, then use semantic search
    1. Exact match (case insensitive)
    2. Contains match (substring)
    3. Token match (all words present)
    4. Semantic similarity (FAISS)
    """
    if df.empty:
        return None
    
    # Normalize: lowercase and remove trailing punctuation for better matching
    import re
    query_lower = text.lower().strip()
    # Remove trailing punctuation (., !, ?, etc.) but keep internal punctuation like apostrophes
    query_normalized = re.sub(r'[.!?,;:]+$', '', query_lower).strip()
    
    # Step 1: Exact match (highest priority)
    # Try normalized query first (without trailing punctuation)
    # Also normalize French phrases for punctuation
    df['French_normalized'] = df['French'].str.lower().str.strip().str.replace(r'[.!?,;:]+$', '', regex=True)
    exact_matches = df[df['French_normalized'] == query_normalized]
    if not exact_matches.empty:
        result = exact_matches.iloc[0].to_dict()
        result['match_score'] = 100
        result['match_type'] = 'exact'
        # Add audio URLs
        row_id = result.get('row_id')
        result['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
        result['word_audio'] = parse_nufi_words_with_audio(result.get('Nufi', ''))
        print(f"Exact match: '{result['French']}' (score: 100)")
        return result
    
    # Initialize best match candidate
    best_match = None
    best_score = -1
    
    # Helper to update best match
    def update_best_match(candidate, score, match_type):
        nonlocal best_match, best_score
        if score > best_score:
            best_score = score
            best_match = candidate
            best_match['match_score'] = score
            best_match['match_type'] = match_type
            # Add audio URLs
            row_id = best_match.get('row_id')
            best_match['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
            best_match['word_audio'] = parse_nufi_words_with_audio(best_match.get('Nufi', ''))
            print(f"New best match ({match_type}): '{best_match['French']}' (score: {score})")

    # Prepare content-token helper and compute query content tokens
    import re as _re
    def _content_tokens_local(s):
        toks = _re.findall(r"[\w'’̄́̀̂̌]+", str(s).lower())
        stop = set(['je','tu','il','elle','nous','vous','ils','elles','le','la','les','de','des','un','une',
                    'et','à','a','au','aux','dans','pour','par','avec','sans','sur','sous','en','est','suis','es','êtes','sommes','être','avoir'])
        return [t for t in toks if t and t not in stop and len(t) > 1]

    q_content = set(_content_tokens_local(query_normalized))

    # Step 2: Multi-word phrase match - prioritize matches with multiple query words
    # Keep contractions together (e.g., "m'appelle" stays as one word)
    # Also keep short important words like "je", "tu", "il"
    query_words_raw = query_normalized.split()
    query_words = [w for w in query_words_raw if len(w) >= 2]  # Changed from 3 to 2 to keep "je", "tu", etc.
    
    if len(query_words) >= 2:
        # Try to find phrases containing multiple words from query (prioritize phrase matches)
        for i in range(len(query_words), 1, -1):  # Start with all words, then decrease
            # Try different word combinations
            for j in range(len(query_words) - i + 1):
                word_combo = query_words[j:j+i]
                # Skip combos that don't include any content tokens to avoid matching only function words
                if q_content and not any(w in q_content for w in word_combo):
                    continue

                # Check if phrase contains all these words
                multi_matches = df.copy()
                for word in word_combo:
                    multi_matches = multi_matches[multi_matches['French'].str.lower().str.contains(word, na=False, regex=False)]

                if not multi_matches.empty:
                    # Prioritize phrases that start with the same beginning
                    multi_matches = multi_matches.copy()
                    # Check if phrase starts with first query word
                    first_word = word_combo[0]
                    multi_matches['starts_with_query'] = multi_matches['French'].str.lower().str.startswith(first_word)
                    multi_matches['length'] = multi_matches['French'].str.len()
                    # Sort: prioritize phrases starting with query, then prefer shorter (more general) phrases
                    multi_matches = multi_matches.sort_values(['starts_with_query', 'length'], ascending=[False, True])

                    candidate = multi_matches.iloc[0].to_dict()
                    # Penalize if we only matched a small part of the query
                    match_ratio = i / len(query_words)
                    base_score = 95 - (5 * (len(query_words) - i))
                    adjusted_score = base_score * match_ratio if match_ratio < 0.5 else base_score

                    update_best_match(candidate, adjusted_score, 'multi-word')
                    break # Found the longest combo, stop looking for shorter ones in this step
            if best_match and best_match['match_type'] == 'multi-word':
                break
    
    # Step 3: Single-word partial match - find sentences containing any significant word
    if query_words and (not best_match or best_score < 60):
        # Only use words >=3 chars for single-word matching to avoid false positives
        significant_words = [w for w in query_words if len(w) >= 3]
        for word in significant_words[:3]:  # Check first 3 significant words
            word_matches = df[df['French'].str.lower().str.contains(word, na=False, regex=False)]
            if not word_matches.empty:
                # Prioritize phrases starting with the word, then shorter matches
                word_matches = word_matches.copy()
                word_matches['starts_with_word'] = word_matches['French'].str.lower().str.startswith(word)
                word_matches['length'] = word_matches['French'].str.len()
                word_matches = word_matches.sort_values(['starts_with_word', 'length'], ascending=[False, True])
                candidate = word_matches.iloc[0].to_dict()
                update_best_match(candidate, 60, 'single-word') # Lower score for single word
                break
    
    # Step 4: Substring contains match (for exact phrase)
    contains_matches = df[df['French'].str.lower().str.contains(query_normalized, na=False, regex=False)]
    if not contains_matches.empty:
        # Prioritize shorter matches (more specific)
        contains_matches = contains_matches.copy()
        contains_matches['length'] = contains_matches['French'].str.len()
        contains_matches = contains_matches.sort_values('length')
        candidate = contains_matches.iloc[0].to_dict()
        update_best_match(candidate, 90, 'contains')
    
    # Step 6: Token-based match (all query words present)
    query_tokens = set(query_normalized.split())
    if query_tokens:
        def contains_all_tokens(french_text):
            french_tokens = set(french_text.lower().split())
            return query_tokens.issubset(french_tokens)
        
        token_matches = df[df['French'].apply(contains_all_tokens)]
        if not token_matches.empty:
            # Prioritize shorter matches
            token_matches = token_matches.copy()
            token_matches['length'] = token_matches['French'].str.len()
            token_matches = token_matches.sort_values('length')
            candidate = token_matches.iloc[0].to_dict()
            update_best_match(candidate, 80, 'token')
    
    # Step 7: Semantic search as fallback (for paraphrases and close meanings)
    query_embedding = embedding_model.encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    k = 5
    distances, indices = faiss_index.search(query_embedding, k)
    
    best_idx = indices[0][0]
    semantic_score = float(distances[0][0]) * 100
    
    print(f"Semantic match candidate: '{df.iloc[best_idx]['French']}' (score: {semantic_score:.1f})")

    # Lexical/content safeguard to reduce false positives:
    # - compute content tokens (remove common stopwords/auxiliary words)
    # - require at least some overlap of content tokens OR a very high semantic score
    import re as _re
    def _content_tokens(s):
        toks = _re.findall(r"[\w'’̄́̀̂̌]+", str(s).lower())
        stop = set(['je','tu','il','elle','nous','vous','ils','elles','le','la','les','de','des','un','une',
                    'et','à','a','au','aux','dans','pour','par','avec','sans','sur','sous','en','est','suis','es','êtes','sommes','être','avoir'])
        return [t for t in toks if t and t not in stop and len(t) > 1]

    candidate = df.iloc[best_idx].to_dict()
    q_content = set(_content_tokens(text))
    cand_content = set(_content_tokens(candidate.get('French','')))
    content_overlap = 0.0
    if q_content:
        content_overlap = len(q_content & cand_content) / float(len(q_content))

    # Debug print
    print(f"  query_content={q_content}, candidate_content={cand_content}, content_overlap={content_overlap:.2f}")

    # Acceptance rules:
    # - If semantic_score >= 90: accept (very strong semantic match)
    # - Else if semantic_score >= 70 AND content_overlap >= 0.25: accept
    # - Otherwise reject semantic fallback to avoid misleading matches
    if semantic_score >= 90 or (semantic_score >= 70 and content_overlap >= 0.25):
        if not best_match or semantic_score > best_score:
            update_best_match(candidate, semantic_score, 'semantic')
    else:
        # Log rejected semantic candidate for offline analysis
        msg = (f"REJECTED SEMANTIC - query='{text}', candidate='{candidate.get('French','')}', "
               f"score={semantic_score:.2f}, overlap={content_overlap:.2f}")
        print(msg)
        try:
            logger.info(msg)
        except Exception:
            # Fallback to printing if logger misconfigured
            print('Failed to write rejected semantic to log')
    
    return best_match

def find_top_semantic_matches(text, top_k=5):
    """
    Return top K semantic matches for a query text
    Used in production mode to show multiple relevant matches
    """
    if df.empty:
        return []
    
    # Generate embedding for the query
    query_embedding = embedding_model.encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search for top K matches
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        idx = indices[0][i]
        score = float(distances[0][i])
        score_percentage = score * 100
        
        result = df.iloc[idx].to_dict()
        result['match_score'] = score_percentage
        result['match_type'] = 'semantic'
        # Add audio URLs
        row_id = result.get('row_id')
        result['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
        result['word_audio'] = parse_nufi_words_with_audio(result.get('Nufi', ''))
        results.append(result)
    
    return results


@app.route('/debug_match', methods=['GET', 'POST'])
def debug_match():
    """Debug endpoint: accept ?text=... (GET) or form/body param 'text' (POST)
    Returns the best hybrid match and top semantic matches with debug info.
    """
    text = request.values.get('text', '')
    if not text:
        return jsonify({'error': 'Please provide a `text` parameter'}), 400

    # Allow caller to request how many top semantic matches to return (debug convenience)
    try:
        top_k = int(request.values.get('top_k', 5))
    except Exception:
        top_k = 5
    # Constrain top_k to a reasonable range to avoid heavy queries
    top_k = max(1, min(50, top_k))

    best = find_closest_match(text)
    top = find_top_semantic_matches(text, top_k)

    return jsonify({
        'query': text,
        'top_k': top_k,
        'best_match': best,
        'top_matches': top
    })


@app.route('/debug')
def debug_ui():
    return render_template('debug.html')

@app.route('/')
def index():
    return render_template('index.html', app_mode=APP_MODE)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get language from request (default to auto-detect)
    language = request.form.get('language', 'auto')
    if language == 'auto':
        language = None  # None means auto-detect in Whisper
    
    # Save temporary file with original format
    file_ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'webm'
    temp_input = os.path.join(UPLOAD_FOLDER, f'temp_audio_input.{file_ext}')
    audio_file.save(temp_input)
    
    # Check file size for debugging
    file_size = os.path.getsize(temp_input)
    print(f"Audio file saved: {file_ext}, size: {file_size} bytes")
    print(f"Language setting: {language if language else 'auto-detect'}")
    
    # Check audio file properties
    try:
        import soundfile as sf
        info = sf.info(temp_input)
        print(f"Audio info: format={info.format}, subtype={info.subtype}, samplerate={info.samplerate}, channels={info.channels}, duration={info.duration:.2f}s")
    except Exception as e:
        print(f"Audio info error: {e}")
    
    if file_size < 1000:  # Less than 1KB is likely empty/silent
        return jsonify({
            'error': 'Audio file is too small. Please record for at least 2-3 seconds and speak clearly.',
            'transcription': '',
            'match': None
        }), 400
    
    try:
        lang_text = f"language: {language}" if language else "auto-detect"

        # Development: run three models for comparison
        if APP_MODE != 'production' and all(k in whisper_models for k in ['small', 'medium', 'large']):
            print(f"Transcribing with three Whisper models (dev): small, medium, {WHISPER_MODEL_SIZE} ...")
            model_outputs = {}
            detected_lang_large = 'unknown'

            # Order matters for logging; small -> medium -> large
            for key in ['small', 'medium', 'large']:
                mdl = whisper_models[key]
                print(f" - {key}: transcribing ({lang_text})")
                res = mdl.transcribe(temp_input, language=language, fp16=False, verbose=False)
                txt = res.get('text', '').strip()
                model_outputs[key] = {
                    'text': txt,
                    'detected_language': res.get('language', 'unknown')
                }
                if key == 'large':
                    detected_lang_large = model_outputs[key]['detected_language']

            # Validate at least large has output
            if not model_outputs['large']['text']:
                return jsonify({
                    'error': 'No speech detected in audio. Please speak clearly and try again.',
                    'transcription': '',
                    'match': None
                }), 400

            # Compute matches for each transcription
            matches = {}
            for key in ['small', 'medium', 'large']:
                txt = model_outputs[key]['text']
                matches[key] = find_closest_match(txt) if txt else None

            # Calculate top semantic matches for the large model (best quality)
            top_matches = []
            if model_outputs['large']['text']:
                top_matches = find_top_semantic_matches(model_outputs['large']['text'], 5)

            response = {
                'transcription': model_outputs['large']['text'],
                'detected_language': detected_lang_large,
                'model_used': WHISPER_MODEL_SIZE,
                'match': matches['large'],
                'top_matches': top_matches,
                'transcription_small': model_outputs['small']['text'],
                'transcription_medium': model_outputs['medium']['text'],
                'transcription_large': model_outputs['large']['text'],
                'match_small': matches['small'],
                'match_medium': matches['medium'],
                'match_large': matches['large']
            }
            return jsonify(response)

        # Production: single model path
        print(f"Transcribing audio with Whisper {WHISPER_MODEL_SIZE} (format: {file_ext}, {lang_text})...")
        result = whisper_model.transcribe(temp_input, language=language, fp16=False, verbose=False)
        transcription = result.get("text", "").strip()
        detected_lang = result.get("language", "unknown")
        print(f"   Transcription: '{transcription}' (detected: {detected_lang})")
        print(f"   Full result keys: {list(result.keys())}")
        if "segments" in result:
            print(f"   Segments: {len(result['segments'])}")
            for i, seg in enumerate(result["segments"][:3]):
                print(f"     Segment {i}: '{seg['text']}' (start: {seg['start']:.2f}, end: {seg['end']:.2f})")

        if not transcription:
            return jsonify({
                'error': 'No speech detected in audio. Please speak clearly and try again.',
                'transcription': '',
                'match': None
            }), 400

        # Calculate best hybrid match (Exact > Token > Semantic)
        best_match = find_closest_match(transcription) if transcription else None
        
        # Calculate top semantic matches for the list
        top_matches = find_top_semantic_matches(transcription, 5) if transcription else []

        response = {
            'transcription': transcription,
            'detected_language': detected_lang,
            'model_used': WHISPER_MODEL_SIZE,
            'match': best_match,  # Show best hybrid match in main panel
            'top_matches': top_matches,
            # For frontend compatibility: duplicate in production
            'transcription_small': transcription,
            'transcription_medium': transcription,
            'transcription_large': transcription,
            'match_small': best_match,
            'match_medium': best_match,
            'match_large': best_match
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    finally:
        # Cleanup
        if os.path.exists(temp_input):
            os.remove(temp_input)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
