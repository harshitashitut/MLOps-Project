"""
Test script for content analysis module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.preprocessing import extract_audio
from pipeline.audio_analysis import transcribe_with_whisper
from pipeline.content_analysis import analyze_content_gemini, extract_content_summary

def test_content_pipeline():
    print("📊 STARTING CONTENT ANALYSIS TEST")
    
    # Test video
    test_video = "data/input/demo_video_1.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False
    
    audio_path = None
    
    try:
        # Step 1: Extract audio
        print("1️⃣  Extracting audio...")
        audio_result = extract_audio(test_video)
        audio_path = audio_result if isinstance(audio_result, str) else audio_result.get('audio_path')
        print(f"   ✅ Audio extracted: {audio_path}")
        
        # Step 2: Get transcript
        print("2️⃣  Transcribing...")
        transcription = transcribe_with_whisper(audio_path)
        transcript = transcription['transcript']
        print(f"   ✅ Transcript: {len(transcript)} chars")
        print(f"   📝 Preview: {transcript[:150]}...")
        
        # Step 3: Analyze content
        print("3️⃣  Analyzing pitch content with Gemini...")
        content_result = analyze_content_gemini(transcript)
        print(f"   ✅ Content analysis received")
        
        # Step 4: Extract summary
        print("4️⃣  Extracting key metrics...")
        summary = extract_content_summary(content_result)
        
        # Validate output
        assert 'structure_score' in summary
        assert 'overall_content_score' in summary
        assert 'strengths' in summary
        
        print("✅ TEST PASSED: Content Module is Production-Ready")
        print(f"\nSpeech Content Scores:")
        print(f"  - Structure: {summary['structure_score']}/10")
        print(f"  - Clarity: {summary['clarity_score']}/10")
        print(f"  - Content Depth: {summary['content_depth_score']}/10")
        print(f"  - Engagement: {summary['engagement_score']}/10")
        print(f"  - Language Quality: {summary['language_quality_score']}/10")
        print(f"  - Overall: {summary['overall_content_score']}/10")
        print(f"  - Rating: {summary['content_rating']}")
        print(f"\nStrengths: {summary['strengths'][:2]}")
        print(f"Weaknesses: {summary['weaknesses'][:2]}")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if audio_path:
            path_to_clean = audio_path if isinstance(audio_path, str) else audio_path.get('audio_path') if isinstance(audio_path, dict) else None
            if path_to_clean and os.path.exists(path_to_clean):
                os.remove(path_to_clean)
                print(f"🧹 Cleaned up: {path_to_clean}")

if __name__ == "__main__":
    success = test_content_pipeline()
    sys.exit(0 if success else 1)