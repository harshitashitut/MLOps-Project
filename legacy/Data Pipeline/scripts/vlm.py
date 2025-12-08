"""
VLM Body Language Analyzer
Analyzes body language and visual presentation from interview videos
"""

import os
import cv2
import tempfile
from pathlib import Path
from PIL import Image
from transformers import pipeline

class BodyLanguageAnalyzer:
    def __init__(self, use_gpu=True):
        """
        Initialize the VLM analyzer for body language
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        import torch
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.vlm_pipeline = None
        
    def load_vlm_model(self, model_name="microsoft/git-large-coco"):
        """
        Load VLM (Vision Language Model) for visual analysis
        
        Args:
            model_name: Hugging Face model ID for image-to-text
            
        Recommended models:
        - "microsoft/git-large-coco" (lightweight, ~2GB, RECOMMENDED for CPU)
        - "Salesforce/blip2-opt-2.7b" (better quality, ~5GB)
        - "Salesforce/blip-image-captioning-large" (alternative, ~2GB)
        - "nlpconnect/vit-gpt2-image-captioning" (very light, ~1GB)
        """
        print(f"Loading VLM model: {model_name}")
        
        self.vlm_pipeline = pipeline(
            "image-to-text",
            model=model_name,
            device=0 if self.device == "cuda" else -1,
        )
        print("VLM model loaded successfully")
    
    def extract_frames(self, video_path, num_frames=5, output_dir=None):
        """
        Extract key frames from video for VLM analysis
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            output_dir: Directory to save frames (optional)
            
        Returns:
            List of frame paths
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        print(f"Extracting {num_frames} frames from {video_path}")
        
        # Open video
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        if total_frames == 0:
            raise ValueError(f"Could not read video: {video_path}")
        
        # Calculate frame indices to extract (evenly spaced)
        # Skip first and last 10% to avoid intro/outro
        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
        usable_frames = end_frame - start_frame
        
        frame_indices = [start_frame + int(i * usable_frames / (num_frames - 1)) for i in range(num_frames)]
        
        frame_paths = []
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, frame_num in enumerate(frame_indices):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video.read()
            
            if ret:
                frame_path = os.path.join(output_dir, f"frame_{idx}.jpg")
                # Convert BGR to RGB for better image quality
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(frame_path, frame_rgb)
                frame_paths.append(frame_path)
                print(f"  ✓ Extracted frame {idx + 1}/{num_frames} (frame #{frame_num})")
        
        video.release()
        print(f"Frames saved to {output_dir}")
        return frame_paths
    
    def analyze_single_frame(self, image_path, frame_num):
        """
        Analyze a single frame for body language
        
        Args:
            image_path: Path to image file
            frame_num: Frame number for reference
            
        Returns:
            Dictionary with analysis
        """
        if self.vlm_pipeline is None:
            raise ValueError("VLM model not loaded. Call load_vlm_model() first.")
        
        print(f"Analyzing frame {frame_num}...")
        
        # Load image
        image = Image.open(image_path)
        
        # Generate caption/description
        result = self.vlm_pipeline(image, max_new_tokens=150)
        description = result[0]['generated_text']
        
        return {
            'frame_num': frame_num,
            'description': description,
            'image_path': image_path
        }
    
    def analyze_body_language(self, frame_paths):
        """
        Analyze body language from multiple frames
        
        Args:
            frame_paths: List of paths to extracted frames
            
        Returns:
            Comprehensive body language analysis
        """
        print("\n=== Analyzing Body Language ===")
        
        frame_analyses = []
        for i, frame_path in enumerate(frame_paths):
            analysis = self.analyze_single_frame(frame_path, i + 1)
            frame_analyses.append(analysis)
        
        # Compile results
        print("\n=== Frame-by-Frame Analysis ===")
        for analysis in frame_analyses:
            print(f"\nFrame {analysis['frame_num']}:")
            print(f"  {analysis['description']}")
        
        return frame_analyses
    
    def analyze_video(self, video_path, num_frames=5, cleanup=True):
        """
        Complete pipeline: extract frames and analyze body language
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            cleanup: Whether to delete frames after analysis
            
        Returns:
            Dictionary with frame analyses and summary
        """
        print(f"\n{'='*60}")
        print("BODY LANGUAGE ANALYSIS")
        print(f"{'='*60}")
        
        # Create temporary directory for frames
        frame_dir = tempfile.mkdtemp()
        
        try:
            # Extract frames
            frame_paths = self.extract_frames(video_path, num_frames, frame_dir)
            
            # Analyze frames
            frame_analyses = self.analyze_body_language(frame_paths)
            
            # Generate summary
            summary = self._generate_summary(frame_analyses)
            
            return {
                "frame_analyses": frame_analyses,
                "summary": summary,
                "frame_dir": frame_dir if not cleanup else None
            }
        finally:
            # Clean up temporary files if requested
            if cleanup:
                for frame_path in frame_paths:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                if os.path.exists(frame_dir):
                    os.rmdir(frame_dir)
    
    def _generate_summary(self, frame_analyses):
        """
        Generate a summary of body language across all frames
        
        Args:
            frame_analyses: List of frame analysis dictionaries
            
        Returns:
            Summary text
        """
        # Simple keyword-based summary
        all_descriptions = " ".join([a['description'].lower() for a in frame_analyses])
        
        observations = []
        
        # Check for positive indicators
        if any(word in all_descriptions for word in ['smiling', 'smile', 'happy']):
            observations.append(" Positive facial expressions detected")
        
        if any(word in all_descriptions for word in ['sitting', 'standing', 'upright']):
            observations.append(" Proper posture maintained")
        
        if any(word in all_descriptions for word in ['professional', 'formal', 'business']):
            observations.append(" Professional appearance")
        
        # Check for potential issues
        if any(word in all_descriptions for word in ['looking down', 'distracted', 'away']):
            observations.append(" Consider maintaining better eye contact with camera")
        
        if len(observations) == 0:
            observations.append("• General professional demeanor observed")
        
        summary = "\n".join(observations)
        return summary


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = BodyLanguageAnalyzer(use_gpu=False)  # Set to True if you have GPU
    
    # Load VLM model
    print("\n=== Loading VLM Model ===")
    
    # Choose one model:
    
    # Option 1: Lightweight (RECOMMENDED for CPU) - ~2GB
    analyzer.load_vlm_model("microsoft/git-large-coco")
    
    # Option 2: Very lightweight - ~1GB
    # analyzer.load_vlm_model("nlpconnect/vit-gpt2-image-captioning")
    
    # Option 3: Better quality - ~5GB (slower on CPU)
    # analyzer.load_vlm_model("Salesforce/blip2-opt-2.7b")
    
    # Option 4: Alternative lightweight - ~2GB
    # analyzer.load_vlm_model("Salesforce/blip-image-captioning-large")
    
    # Analyze a video
    video_path = "/home/mohit/Downloads/project_mlops/MLOps-Project/Data Pipeline/Data/video2.webm"  # Update with your video path
    
    if os.path.exists(video_path):
        result = analyzer.analyze_video(
            video_path=video_path,
            num_frames=5,  # Extract 5 frames
            cleanup=True   # Delete frames after analysis(we could also save these but we chorse not to as our model is pretty small)
        )
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(result['summary'])
        
    else:
        print(f"Video file not found: {video_path}")