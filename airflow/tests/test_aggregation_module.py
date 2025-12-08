"""
Test script for aggregation module - FULL END-TO-END PIPELINE TEST
"""
import sys
import json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.preprocessing import extract_frames, extract_audio
from pipeline.visual_analysis import analyze_pose_mediapipe, analyze_visual_gemini, combine_visual_analyses
from pipeline.audio_analysis import (
    transcribe_with_whisper, 
    analyze_emotion_wav2vec, 
    compute_vocal_metrics,
    combine_audio_analyses
)
from pipeline.content_analysis import analyze_content_gemini, extract_content_summary
from pipeline.aggregation import aggregate_all_results, save_results, generate_summary_stats

def test_full_pipeline():
    print("🎯 STARTING FULL END-TO-END PIPELINE TEST")
    print("=" * 60)
    
    # Test video
    test_video = "data/input/demo_video_1.mp4"
    video_id = "full_test"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return False
    
    audio_path = None
    frame_dir = None
    
    try:
        # PHASE 1: PREPROCESSING
        print("\n📹 PHASE 1: PREPROCESSING")
        print("1️⃣  Extracting frames...")
        frame_result = extract_frames(test_video)
        frame_paths = frame_result if isinstance(frame_result, list) else frame_result.get('frame_paths', [])
        print(f"   ✅ Extracted {len(frame_paths)} frames")
        
        print("2️⃣  Extracting audio...")
        audio_result = extract_audio(test_video)
        audio_path = audio_result if isinstance(audio_result, str) else audio_result.get('audio_path')
        print(f"   ✅ Audio extracted")
        
        # PHASE 2: VISUAL ANALYSIS
        print("\n👁️  PHASE 2: VISUAL ANALYSIS")
        print("3️⃣  Analyzing pose (MediaPipe)...")
        pose_data = analyze_pose_mediapipe(frame_paths)
        print(f"   ✅ Posture score: {pose_data.get('posture_score', 0)}")
        
        print("4️⃣  Analyzing visual presentation (Gemini)...")
        visual_gemini = analyze_visual_gemini(frame_paths)
        print(f"   ✅ Visual analysis complete")
        
        print("5️⃣  Combining visual analyses...")
        visual_combined = combine_visual_analyses(pose_data, visual_gemini)
        print("\n🔍 DEBUG - Visual Combined Data:")
        print(json.dumps(visual_combined, indent=2))
        print(f"   ✅ Combined visual score: {visual_combined.get('visual_presentation', {}).get('overall_visual_score', 0)}")
        
        # PHASE 3: AUDIO ANALYSIS
        print("\n🎤 PHASE 3: AUDIO ANALYSIS")
        print("6️⃣  Transcribing (Whisper)...")
        transcription = transcribe_with_whisper(audio_path)
        print(f"   ✅ Transcribed: {transcription['word_count']} words")
        
        print("7️⃣  Analyzing emotion (Wav2Vec2)...")
        emotion = analyze_emotion_wav2vec(audio_path)
        print(f"   ✅ Emotion: {emotion['dominant_emotion']}")
        
        print("8️⃣  Computing vocal metrics...")
        vocal_metrics = compute_vocal_metrics(transcription['transcript'], transcription['duration'])
        print(f"   ✅ Speaking pace: {vocal_metrics['wpm']} WPM")
        
        print("9️⃣  Combining audio analyses...")
        audio_combined = combine_audio_analyses(transcription, emotion, vocal_metrics)
        print(f"   ✅ Audio analysis complete")
        
        # PHASE 4: CONTENT ANALYSIS
        print("\n📊 PHASE 4: CONTENT ANALYSIS")
        print("🔟 Analyzing speech content (Gemini)...")
        content_result = analyze_content_gemini(transcription['transcript'])
        content_summary = extract_content_summary(content_result)
        print(f"   ✅ Content score: {content_summary['overall_content_score']}/10")
        
        # PHASE 5: AGGREGATION
        print("\n🎯 PHASE 5: FINAL AGGREGATION")
        print("1️⃣1️⃣  Aggregating all results (Gemini Pro)...")
        final_results = aggregate_all_results(visual_combined, audio_combined, content_result)
        print(f"   ✅ Overall score: {final_results.get('overall_score', 0)}/100")
        print(f"   ✅ Performance level: {final_results.get('performance_level', 'unknown')}")
        
        print("1️⃣2️⃣  Saving results...")
        output_path = save_results(final_results, video_id)
        print(f"   ✅ Saved to: {output_path}")
        
        print("1️⃣3️⃣  Generating summary stats...")
        summary = generate_summary_stats(final_results)
        
        # VALIDATION
        print("\n✅ VALIDATION")
        assert final_results.get('overall_score', 0) > 0, "Overall score should be > 0"
        assert 'category_scores' in final_results, "Missing category_scores"
        assert 'improvements' in final_results, "Missing improvements"
        assert 'strengths' in final_results, "Missing strengths"
        assert 'detailed_feedback' in final_results, "Missing detailed_feedback"
        assert os.path.exists(output_path), "Output file not created"
        
        # DISPLAY RESULTS
        print("\n" + "=" * 60)
        print("🎉 TEST PASSED: FULL PIPELINE IS PRODUCTION-READY")
        print("=" * 60)
        
        print("\n📊 FINAL SCORES:")
        cat_scores = final_results.get('category_scores', {})
        print(f"  Overall: {final_results.get('overall_score', 0)}/100")
        print(f"  Content: {cat_scores.get('content', {}).get('score', 0)}/100")
        print(f"  Delivery: {cat_scores.get('vocal_delivery', {}).get('score', 0)}/100")
        print(f"  Visual: {cat_scores.get('visual_presentation', {}).get('score', 0)}/100")
        print(f"  Tone: {cat_scores.get('tone_emotion', {}).get('score', 0)}/100")
        
        print("\n🎯 KEY METRICS:")
        key_metrics = final_results.get('key_metrics', {})
        print(f"  Words per minute: {key_metrics.get('words_per_minute', 0)}")
        print(f"  Filler words: {key_metrics.get('filler_words_count', 0)}")
        print(f"  Speech duration: {key_metrics.get('speech_duration_seconds', 0)}s")
        
        print("\n💪 TOP STRENGTHS:")
        for i, strength in enumerate(final_results.get('strengths', [])[:3], 1):
            print(f"  {i}. [{strength.get('category', 'N/A')}] {strength.get('strength', 'N/A')}")
        
        print("\n🔧 TOP IMPROVEMENTS:")
        for i, improvement in enumerate(final_results.get('improvements', [])[:3], 1):
            print(f"  {i}. [{improvement.get('category', 'N/A')}] {improvement.get('issue', 'N/A')}")
            print(f"     Priority: {improvement.get('priority', 'N/A')}")
        
        print("\n📝 NEXT STEPS:")
        for i, step in enumerate(final_results.get('next_steps', [])[:3], 1):
            print(f"  {i}. {step}")
        
        print("\n" + "=" * 60)
        print(f"✅ Full results saved to: {output_path}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("\n🧹 CLEANUP...")
        if audio_path:
            path_to_clean = audio_path if isinstance(audio_path, str) else audio_path.get('audio_path') if isinstance(audio_path, dict) else None
            if path_to_clean and os.path.exists(path_to_clean):
                os.remove(path_to_clean)
                print(f"   ✅ Removed: {path_to_clean}")
        
        # Note: Frame cleanup happens in preprocessing.py's extract_frames function

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)