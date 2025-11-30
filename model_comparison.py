"""
Model Comparison Script for Whisper Models

Usage:
python model_comparison.py <audio_file_path>

This script loads three Whisper models (small, medium, large-v3) and transcribes
the provided audio file with each, showing the results for comparison.
"""

import sys
import os
import whisper

def compare_models(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    models = ['small', 'medium', 'large-v3']
    results = {}

    print(f"🎯 Three-Way Model Comparison for: {audio_path}")
    print("=" * 60)

    for model_size in models:
        print(f"\n🔄 Loading {model_size} model...")
        try:
            model = whisper.load_model(model_size)
            print(f"✅ {model_size} loaded successfully")

            print(f"🎙️  Transcribing with {model_size}...")
            result = model.transcribe(audio_path, fp16=False, verbose=False)
            transcription = result["text"].strip()
            detected_lang = result.get("language", "unknown")

            results[model_size] = {
                'transcription': transcription,
                'language': detected_lang,
                'segments': len(result.get('segments', []))
            }

            print(f"📝 Transcription: '{transcription}'")
            print(f"🌐 Detected Language: {detected_lang}")
            print(f"🎵 Segments: {results[model_size]['segments']}")

        except Exception as e:
            print(f"❌ Error with {model_size}: {e}")
            results[model_size] = {'error': str(e)}

    print(f"\n⚖️  Comparison Summary")
    print("=" * 60)

    transcriptions = [results[m].get('transcription', 'ERROR') for m in models]
    all_same = all(t == transcriptions[0] for t in transcriptions)

    if all_same:
        print("✅ All models agree!")
        print(f"📝 Transcription: '{transcriptions[0]}'")
    else:
        print("⚠️  Models disagree:")
        for model, result in results.items():
            trans = result.get('transcription', 'ERROR')
            print(f"  {model}: '{trans}'")

    # Check for empty transcriptions
    empty_count = sum(1 for t in transcriptions if t == "" or t == "ERROR")
    if empty_count == len(models):
        print("\n🚨 All transcriptions are empty! Possible issues:")
        print("  - Audio file is silent or corrupted")
        print("  - Audio format not supported")
        print("  - Audio too short (< 1 second)")
        print("  - Speech not in supported languages")

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model_comparison.py <audio_file_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    compare_models(audio_file)