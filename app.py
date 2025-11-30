import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
import whisper
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

app = Flask(__name__)

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
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
print(f"   ✓ Whisper {WHISPER_MODEL_SIZE} loaded successfully!")

# Load semantic search model and index
print("Loading semantic search model...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Semantic search model loaded successfully.")

print("Loading FAISS index...")
faiss_index = faiss.read_index('assets/faiss_index.bin')
with open('assets/index_mapping.pkl', 'rb') as f:
    index_data = pickle.load(f)
print(f"FAISS index loaded: {faiss_index.ntotal} entries")

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
    df['row_id'] = df.index + 1
    # Ensure columns exist, normalize if needed
    df['French'] = df['French'].astype(str).str.lower().str.strip()
    print(f"Phrasebook loaded successfully: {len(df)} entries")
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
                    
                    result = multi_matches.iloc[0].to_dict()
                    result['match_score'] = 95 - (5 * (len(query_words) - i))  # Higher score for more words matched
                    result['match_type'] = 'multi-word'
                    # Add audio URLs
                    row_id = result.get('row_id')
                    result['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
                    result['word_audio'] = parse_nufi_words_with_audio(result.get('Nufi', ''))
                    print(f"Multi-word match on {word_combo}: '{result['French']}' (score: {result['match_score']})")
                    return result
    
    # Step 3: Single-word partial match - find sentences containing any significant word
    if query_words:
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
                result = word_matches.iloc[0].to_dict()
                result['match_score'] = 85
                result['match_type'] = 'single-word'
                # Add audio URLs
                row_id = result.get('row_id')
                result['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
                result['word_audio'] = parse_nufi_words_with_audio(result.get('Nufi', ''))
                print(f"Single-word match on '{word}': '{result['French']}' (score: 85)")
                return result
    
    # Step 4: Substring contains match (for exact phrase)
    contains_matches = df[df['French'].str.lower().str.contains(query_normalized, na=False, regex=False)]
    if not contains_matches.empty:
        # Prioritize shorter matches (more specific)
        contains_matches = contains_matches.copy()
        contains_matches['length'] = contains_matches['French'].str.len()
        contains_matches = contains_matches.sort_values('length')
        result = contains_matches.iloc[0].to_dict()
        result['match_score'] = 90
        result['match_type'] = 'contains'
        # Add audio URLs
        row_id = result.get('row_id')
        result['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
        result['word_audio'] = parse_nufi_words_with_audio(result.get('Nufi', ''))
        print(f"Contains match: '{result['French']}' (score: 90)")
        return result
    
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
            result = token_matches.iloc[0].to_dict()
            result['match_score'] = 80
            result['match_type'] = 'token'
            # Add audio URLs
            row_id = result.get('row_id')
            result['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
            result['word_audio'] = parse_nufi_words_with_audio(result.get('Nufi', ''))
            print(f"Token match: '{result['French']}' (score: 80)")
            return result
    
    # Step 7: Semantic search as fallback (for paraphrases and close meanings)
    query_embedding = embedding_model.encode([text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    k = 5
    distances, indices = faiss_index.search(query_embedding, k)
    
    best_idx = indices[0][0]
    best_score = float(distances[0][0])
    score_percentage = best_score * 100
    
    print(f"Semantic match: '{df.iloc[best_idx]['French']}' (score: {score_percentage:.1f})")
    
    # Only use semantic if score is reasonably high
    if best_score > 0.65:  # 65% similarity threshold
        result = df.iloc[best_idx].to_dict()
        result['match_score'] = score_percentage
        result['match_type'] = 'semantic'
        # Add audio URLs
        row_id = result.get('row_id')
        result['sentence_audio_url'] = get_audio_url('phrasebook', f'nufi_phrasebook_{row_id}.mp3') if row_id else None
        result['word_audio'] = parse_nufi_words_with_audio(result.get('Nufi', ''))
        return result
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

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
    
    if file_size < 1000:  # Less than 1KB is likely empty/silent
        return jsonify({
            'error': 'Audio file is too small. Please record for at least 2-3 seconds and speak clearly.',
            'transcription': '',
            'match': None
        }), 400
    
    try:
        # Transcribe with Whisper model
        lang_text = f"language: {language}" if language else "auto-detect"
        print(f"Transcribing audio with Whisper {WHISPER_MODEL_SIZE} (format: {file_ext}, {lang_text})...")
        
        result = whisper_model.transcribe(temp_input, language=language, fp16=False, verbose=False)
        transcription = result["text"].strip()
        detected_lang = result.get("language", "unknown")
        print(f"   Transcription: {transcription} (detected: {detected_lang})")
        
        # Check if transcription is empty
        if not transcription:
            return jsonify({
                'error': 'No speech detected in audio. Please speak clearly and try again.',
                'transcription': '',
                'match': None
            }), 400
        
        # Find match for the transcription
        match = find_closest_match(transcription) if transcription else None
        
        response = {
            'transcription': transcription,
            'detected_language': detected_lang,
            'model_used': WHISPER_MODEL_SIZE,
            'match': match
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
