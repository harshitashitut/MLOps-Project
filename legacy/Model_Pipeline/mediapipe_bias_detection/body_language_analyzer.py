"""
MediaPipe Body Language Analyzer
Analyzes posture and body language from video
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

class BodyLanguageAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        return angle
    
    def analyze_posture(self, landmarks):
        """Analyze posture from landmarks"""
        try:
            # Extract key landmarks
            nose = [landmarks[0].x, landmarks[0].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]
            right_shoulder = [landmarks[12].x, landmarks[12].y]
            left_hip = [landmarks[23].x, landmarks[23].y]
            right_hip = [landmarks[24].x, landmarks[24].y]
            
            # Calculate shoulder alignment (should be level)
            shoulder_slope = abs(left_shoulder[1] - right_shoulder[1])
            
            # Calculate forward lean (nose position relative to shoulders)
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            forward_lean = abs(nose[0] - shoulder_center_x)
            
            # Calculate spine angle (shoulder to hip alignment)
            mid_shoulder = [(left_shoulder[0] + right_shoulder[0])/2, 
                           (left_shoulder[1] + right_shoulder[1])/2]
            mid_hip = [(left_hip[0] + right_hip[0])/2, 
                      (left_hip[1] + right_hip[1])/2]
            
            # Vertical alignment (upright vs hunched)
            vertical_alignment = abs(mid_shoulder[0] - mid_hip[0])
            
            # Posture scoring (0-1)
            posture_score = 1.0
            
            # Penalize poor alignment
            if shoulder_slope > 0.05:  # Shoulders not level
                posture_score -= 0.2
            
            if forward_lean > 0.08:  # Leaning forward
                posture_score -= 0.3
            
            if vertical_alignment > 0.05:  # Not upright
                posture_score -= 0.3
            
            posture_score = max(0.0, posture_score)
            
            # Determine posture type
            if posture_score > 0.7:
                posture_type = "upright"
            elif forward_lean > 0.08:
                posture_type = "hunched_forward"
            elif vertical_alignment > 0.05:
                posture_type = "slouching"
            else:
                posture_type = "neutral"
            
            return {
                'posture_score': posture_score,
                'posture_type': posture_type,
                'shoulder_alignment': 1 - min(shoulder_slope * 10, 1.0),
                'forward_lean': forward_lean,
                'vertical_alignment': 1 - min(vertical_alignment * 10, 1.0)
            }
            
        except Exception as e:
            return None
    
    def analyze_video(self, video_path, frame_interval=30):
        """
        Analyze video for body language
        
        Args:
            video_path: Path to video file
            frame_interval: Analyze every Nth frame (default: 30 = 1fps for 30fps video)
        
        Returns:
            Dictionary with overall metrics and per-frame results
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Analyzing video: {Path(video_path).name}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Analyzing every {frame_interval} frames...")
        
        frame_results = []
        frame_count = 0
        analyzed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Only analyze every Nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    # Get visibility score (how confident MediaPipe is)
                    visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
                    avg_visibility = np.mean(visibilities)
                    
                    # Analyze posture
                    posture_metrics = self.analyze_posture(results.pose_landmarks.landmark)
                    
                    if posture_metrics:
                        frame_results.append({
                            'frame': frame_count,
                            'timestamp': frame_count / fps,
                            'detection_confidence': avg_visibility,
                            'keypoints_detected': sum(1 for v in visibilities if v > 0.5),
                            **posture_metrics
                        })
                        analyzed_count += 1
                else:
                    # No pose detected
                    frame_results.append({
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'detection_confidence': 0.0,
                        'keypoints_detected': 0,
                        'posture_score': 0.0,
                        'posture_type': 'not_detected'
                    })
            
            frame_count += 1
        
        cap.release()
        
        if not frame_results:
            return {
                'error': 'No frames analyzed',
                'overall_confidence': 0.0
            }
        
        # Calculate overall metrics
        detected_frames = [f for f in frame_results if f['detection_confidence'] > 0]
        
        if not detected_frames:
            return {
                'error': 'No pose detected in any frame',
                'overall_confidence': 0.0,
                'frames_analyzed': len(frame_results),
                'detection_rate': 0.0
            }
        
        overall_metrics = {
            'video_name': Path(video_path).name,
            'frames_analyzed': len(frame_results),
            'frames_detected': len(detected_frames),
            'detection_rate': len(detected_frames) / len(frame_results),
            'overall_confidence': np.mean([f['detection_confidence'] for f in detected_frames]),
            'avg_posture_score': np.mean([f['posture_score'] for f in detected_frames]),
            'avg_keypoints_detected': np.mean([f['keypoints_detected'] for f in detected_frames]),
            'posture_breakdown': self._get_posture_breakdown(detected_frames),
            'frame_results': frame_results
        }
        
        return overall_metrics
    
    def _get_posture_breakdown(self, frames):
        """Count posture types"""
        from collections import Counter
        postures = [f['posture_type'] for f in frames]
        counts = Counter(postures)
        total = len(frames)
        
        return {
            posture: count/total for posture, count in counts.items()
        }
    
    def print_summary(self, results):
        """Print human-readable summary"""
        if 'error' in results:
            print(f"\n❌ Error: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("BODY LANGUAGE ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nVideo: {results['video_name']}")
        print(f"Frames analyzed: {results['frames_analyzed']}")
        print(f"Detection rate: {results['detection_rate']:.1%}")
        print(f"\nOverall Confidence: {results['overall_confidence']:.2f}")
        print(f"Average Posture Score: {results['avg_posture_score']:.2f}")
        print(f"Average Keypoints Detected: {results['avg_keypoints_detected']:.1f}/33")
        
        print("\nPosture Breakdown:")
        for posture, percentage in results['posture_breakdown'].items():
            print(f"  {posture}: {percentage:.1%}")
        
        # Overall assessment
        print("\nAssessment:")
        if results['overall_confidence'] < 0.6:
            print("  ⚠️  LOW CONFIDENCE - Poor video quality or detection issues")
        elif results['avg_posture_score'] > 0.7:
            print("  ✅ Good posture maintained throughout")
        elif results['avg_posture_score'] > 0.5:
            print("  ⚠️  Moderate posture - some slouching detected")
        else:
            print("  ❌ Poor posture - significant hunching/slouching")


# Example usage
if __name__ == "__main__":
    analyzer = BodyLanguageAnalyzer()
    
    # Analyze a video (change this path)
    video_path = "1.mp4"
    
    # Analyze every 30 frames (about 1 second for 30fps video)
    results = analyzer.analyze_video(video_path, frame_interval=30)
    
    # Print summary
    analyzer.print_summary(results)
    
    # Access detailed metrics
    print(f"\nDetailed metrics available:")
    print(f"  results['overall_confidence']: {results.get('overall_confidence', 0):.2f}")
    print(f"  results['avg_posture_score']: {results.get('avg_posture_score', 0):.2f}")
    print(f"  results['detection_rate']: {results.get('detection_rate', 0):.1%}")