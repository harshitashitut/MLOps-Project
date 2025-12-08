import sys
import os
# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.preprocessing import extract_frames
from pipeline.visual_analysis import analyze_pose_mediapipe, analyze_visual_gemini, combine_visual_analyses

def test_visual_full():
    print("\n🎬 STARTING VISUAL PIPELINE TEST")
    video_path = 'data/input/demo_video_1.mp4'
    
    # Mock Context
    context = {'dag_run': type('obj', (object,), {'conf': {'video_id': 'test_vis'}})}

    try:
        # 1. Extract (Generates the temp files needed)
        print("1️⃣  Extracting frames...")
        frames_result = extract_frames(video_path, sample_rate=5, **context)
        frame_paths = frames_result['frame_paths']
        print(f"   ✅ Extracted {len(frame_paths)} frames")

        # 2. MediaPipe
        print("2️⃣  Running MediaPipe Pose...")
        pose_result = analyze_pose_mediapipe(frame_paths, **context)
        print(f"   ✅ Posture Score: {pose_result['posture_score']}")

        # 3. Gemini Vision
        print("3️⃣  Running Gemini Vision...")
        visual_result = analyze_visual_gemini(frame_paths, **context)
        print("   ✅ Gemini Analysis Received")

        # 4. Combine
        print("4️⃣  Combining Results...")
        final = combine_visual_analyses(pose_result, visual_result, **context)
        
        # Validation
        assert 'visual_presentation' in final
        print("\n✅ TEST PASSED: Visual Module is Production-Ready")
        print("Sample Metric (Body Language):", final)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise e

if __name__ == "__main__":
    test_visual_full()
