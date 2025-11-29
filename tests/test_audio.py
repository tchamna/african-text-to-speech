"""
Test script for AfricanSpeaks audio processing
Tests Whisper transcription and dictionary matching
"""
import whisper
import pandas as pd
from fuzzywuzzy import process, fuzz
import os

# Load Whisper model
print("Loading Whisper model...")
model = whisper.load_model("base")
print("✓ Whisper model loaded\n")

# Load dictionary
print("Loading dictionary...")
df = pd.read_csv('assets/Nufi_Francais_Sentences_Phrases_From_Nufi_Tchamna_Dictionary_.csv')
df['French'] = df['French'].astype(str).str.lower().str.strip()
print(f"✓ Dictionary loaded: {len(df)} entries\n")

def find_match(text):
    """Find best match using fuzzy matching"""
    if df.empty:
        return None
    
    choices = df['French'].tolist()
    
    # Try multiple matching strategies
    match1 = process.extractOne(text.lower(), choices, scorer=fuzz.token_sort_ratio)
    match2 = process.extractOne(text.lower(), choices, scorer=fuzz.partial_ratio)
    match3 = process.extractOne(text.lower(), choices, scorer=fuzz.token_set_ratio)
    
    matches = [m for m in [match1, match2, match3] if m]
    if not matches:
        return None
    
    best_match = max(matches, key=lambda x: x[1])
    match_text, score = best_match
    
    if score > 50:
        index = df[df['French'] == match_text].index[0]
        result = df.iloc[index].to_dict()
        result['match_score'] = score
        return result
    return None

def test_audio_file(file_path):
    """Test a single audio file"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(file_path)}")
    print('='*60)
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size:,} bytes")
    
    # Transcribe
    print("\nTranscribing...")
    result = model.transcribe(file_path, language="fr", fp16=False)
    transcription = result["text"].strip()
    
    print(f"✓ Transcription: '{transcription}'")
    
    if not transcription:
        print("✗ No speech detected")
        return
    
    # Find match
    print("\nSearching dictionary...")
    match = find_match(transcription)
    
    if match:
        print(f"✓ Match found!")
        print(f"  French: {match['French']}")
        print(f"  Nufi: {match['Nufi']}")
        print(f"  Score: {match['match_score']}")
    else:
        print("✗ No match found in dictionary")
        # Show top 3 closest matches
        choices = df['French'].tolist()
        top_matches = process.extract(transcription.lower(), choices, scorer=fuzz.token_sort_ratio, limit=3)
        print("\n  Closest matches:")
        for m, score in top_matches:
            print(f"    - '{m}' (score: {score})")

def test_text(text):
    """Test text matching without audio"""
    print(f"\n{'='*60}")
    print(f"Testing text: '{text}'")
    print('='*60)
    
    match = find_match(text)
    
    if match:
        print(f"✓ Match found!")
        print(f"  French: {match['French']}")
        print(f"  Nufi: {match['Nufi']}")
        print(f"  Score: {match['match_score']}")
    else:
        print("✗ No match found")
        # Show top 3 closest matches
        choices = df['French'].tolist()
        top_matches = process.extract(text.lower(), choices, scorer=fuzz.token_sort_ratio, limit=3)
        print("\n  Closest matches:")
        for m, score in top_matches:
            print(f"    - '{m}' (score: {score})")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AfricanSpeaks Test Script")
    print("="*60)
    
    # Test some text phrases
    test_phrases = [
        "je m'appelle",
        "Je m'appelle Gainvier",
        "Mon nom est Nufi",
        "bonjour",
        "merci",
        "l'assiette"
    ]
    
    print("\n\n" + "="*60)
    print("TEXT MATCHING TESTS")
    print("="*60)
    
    for phrase in test_phrases:
        test_text(phrase)
    
    # Test audio files if provided
    print("\n\n" + "="*60)
    print("AUDIO FILE TESTS")
    print("="*60)
    
    # You can add your audio file paths here
    audio_files = [
        r"G:\My Drive\Mbú'ŋwɑ̀'nì\Livres Nufi\audio\fr-33.mp3",
        # Add more files as needed
    ]
    
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            test_audio_file(audio_file)
        else:
            print(f"\n✗ Skipping (file not found): {os.path.basename(audio_file)}")
    
    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60 + "\n")
