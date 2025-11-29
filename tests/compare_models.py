"""
Compare Whisper Medium vs Wav2Vec2 French for speech-to-text accuracy
"""
import os
import whisper
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import numpy as np

print("=" * 80)
print("LOADING MODELS...")
print("=" * 80)

# Load Whisper Medium
print("\n1. Loading Whisper Medium model...")
whisper_model = whisper.load_model("medium")
print("   ✓ Whisper Medium loaded (1.5GB)")

# Load Wav2Vec2 French
print("\n2. Loading Wav2Vec2 French model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
print("   ✓ Wav2Vec2 French loaded (~1.2GB)")

print("\n" + "=" * 80)
print("MODELS READY - Now test with your voice")
print("=" * 80)

def transcribe_with_whisper(audio_path):
    """Transcribe using Whisper Medium"""
    result = whisper_model.transcribe(audio_path, language="fr", fp16=False)
    return result["text"].strip()

def transcribe_with_wav2vec(audio_path):
    """Transcribe using Wav2Vec2 French"""
    # Load audio
    speech, sample_rate = sf.read(audio_path)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
    
    # Process audio
    input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values
    
    # Get logits
    with torch.no_grad():
        logits = wav2vec_model(input_values).logits
    
    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    
    return transcription.strip()

# Test with a sample phrase
print("\n📝 Instructions:")
print("   1. Record yourself saying a French phrase (save as test_audio.wav)")
print("   2. Put the file in the 'uploads' folder")
print("   3. Run this script to compare both models")
print("\nExample test phrases:")
print("   • Je suis guéri (I am healed)")
print("   • Je m'appelle Zona (My name is Zona)")
print("   • L'enfant pleure (The child is crying)")
print("   • Comment allez-vous? (How are you?)")

# Check for test audio
test_audio = "uploads/test_audio.wav"
if os.path.exists(test_audio):
    print(f"\n\n{'=' * 80}")
    print("TESTING WITH: uploads/test_audio.wav")
    print("=" * 80)
    
    print("\n🎤 Whisper Medium transcription:")
    whisper_result = transcribe_with_whisper(test_audio)
    print(f"   → {whisper_result}")
    
    print("\n🎤 Wav2Vec2 French transcription:")
    wav2vec_result = transcribe_with_wav2vec(test_audio)
    print(f"   → {wav2vec_result}")
    
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("=" * 80)
    print(f"Whisper:  {whisper_result}")
    print(f"Wav2Vec2: {wav2vec_result}")
    
    if whisper_result.lower() == wav2vec_result.lower():
        print("\n✓ Both models produced the same result!")
    else:
        print("\n⚠ Models produced different results - compare which is more accurate")
else:
    print(f"\n⚠ No test file found at: {test_audio}")
    print("   Place a .wav file there and run again to test")

print("\n" + "=" * 80)
print("To use a model in app.py:")
print("=" * 80)
print("• For Whisper Medium: whisper_model.transcribe(audio, language='fr')")
print("• For Wav2Vec2: Use the transcribe_with_wav2vec() function above")
