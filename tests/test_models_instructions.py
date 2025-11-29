"""
Quick test: Record "Je suis guéri" and compare both models
"""
import os

print("\n" + "=" * 80)
print("TESTING BOTH MODELS - Record your voice saying 'Je suis guéri'")
print("=" * 80)

# Instructions
print("\n📝 How to test:")
print("   1. Open http://127.0.0.1:5001 in your browser")
print("   2. Record yourself saying: 'Je suis guéri' (I am healed)")
print("   3. Check the transcription result")
print("   4. Come back here and we'll switch models to compare")

print("\n🔧 Current configuration:")
print("   • File: app.py, line 14")
print("   • Current setting: STT_MODEL = 'whisper'")
print("")
print("📊 To test Wav2Vec2:")
print("   1. Stop the server (Ctrl+C in terminal)")
print("   2. Change line 14 to: STT_MODEL = 'wav2vec2'")
print("   3. Restart server: python app.py")
print("   4. Record the same phrase again")

print("\n🎯 Expected comparison:")
print("   Whisper Medium:  'Je suis guérée' (incorrect feminine form)")
print("   Wav2Vec2 French: 'Je suis guéri' (correct masculine form)")
print("\nWav2Vec2 is trained specifically on French, so it should handle")
print("context and grammar better for French-specific sounds and patterns.")

print("\n" + "=" * 80)
print("Or run compare_models.py to test both at once with a saved audio file")
print("=" * 80)
