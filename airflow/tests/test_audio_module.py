"""
Test script for audio analysis module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.preprocessing import extract_audio
from pipeline.audio_analysis import (
    transcribe_with_whisper,
    analyze_emotion_wav2vec,
    compute_vocal_metrics,
    combine_audio_analyses
)

def test_audio_pipeline():
    print("🎤 STARTING AUDIO PIPELINE TEST")
    
    # Test video
    test_video = "data/input/demo_video_1.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False
    
    try:
        # Step 1: Extract audio
        print("1️⃣  Extracting audio...")
        extraction_result = extract_audio(test_video, video_id="test_audio")
        audio_path = extraction_result['audio_path']  # Extract the string path
        print(f"   ✅ Audio extracted: {audio_path}")
        
        # Step 2: Whisper transcription
        print("2️⃣  Running Whisper transcription...")
        transcription = transcribe_with_whisper(audio_path)
        print(f"   ✅ Transcribed: {transcription['word_count']} words")
        print(f"   📝 Preview: {transcription['transcript'][:100]}...")
        
        # Step 3: Emotion analysis
        print("3️⃣  Running emotion analysis...")
        emotion = analyze_emotion_wav2vec(audio_path)
        print(f"   ✅ Emotion: {emotion['dominant_emotion']} ({emotion['confidence']:.2f})")
        
        # Step 4: Vocal metrics
        print("4️⃣  Computing vocal metrics...")
        vocal_metrics = compute_vocal_metrics(
            transcription['transcript'],
            transcription['duration']
        )
        print(f"   ✅ Metrics: {vocal_metrics['wpm']} WPM, {vocal_metrics['filler_count']} fillers")
        
        # Step 5: Combine all
        print("5️⃣  Combining results...")
        combined = combine_audio_analyses(transcription, emotion, vocal_metrics)
        
        # Validate output
        assert 'transcript_data' in combined
        assert 'emotion_data' in combined
        assert 'vocal_data' in combined
        assert 'summary' in combined
        
        print("✅ TEST PASSED: Audio Module is Production-Ready")
        print(f"\nSample Metrics:")
        print(f"  - WPM: {combined['vocal_data']['wpm']}")
        print(f"  - Emotion: {combined['emotion_data']['dominant_emotion']}")
        print(f"  - Articulation: {combined['vocal_data']['articulation_score']}/10")
        print(f"  - Filler words: {combined['vocal_data']['filler_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup temp audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"🧹 Cleaned up: {audio_path}")

if __name__ == "__main__":
    success = test_audio_pipeline()
    sys.exit(0 if success else 1)
    