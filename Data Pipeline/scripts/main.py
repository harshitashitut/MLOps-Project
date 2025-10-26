"""
Interview Analysis Backend
Analyzes interview videos using speech transcription and LLM feedback
"""

import sys
sys.path.append("/home/mohit/.local/lib/python3.13/site-packages")


"""
Interview Analysis Backend
Analyzes interview videos using speech transcription and LLM feedback
"""

import os
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor,
    pipeline
)
import cv2
import subprocess
import tempfile
from pathlib import Path

class InterviewAnalyzer:
    def __init__(self, use_gpu=True):
        """
        Initialize the analyzer with speech recognition and LLM models
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize speech recognition model (Whisper)
        self.transcription_pipeline = None
        self.llm_pipeline = None
        
    def load_transcription_model(self, model_name="openai/whisper-base"):
        """
        Load Whisper model for speech-to-text
        
        Args:
            model_name: Hugging Face model ID for transcription
        """
        print(f"Loading transcription model: {model_name}")
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        self.transcription_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device,
        )
        print("Transcription model loaded successfully")
    
    def load_llm_model(self, model_name="google/flan-t5-base"):
        """
        Load LLM for answer analysis
        
        Args:
            model_name: Hugging Face model ID for text generation
        """
        print(f"Loading LLM model: {model_name}")
        
        # Detect model type based on name
        if "t5" in model_name.lower() or "flan" in model_name.lower():
            # T5 models use text2text-generation
            task = "text2text-generation"
        else:
            # Most other models use text-generation
            task = "text-generation"
        
        self.llm_pipeline = pipeline(
            task,
            model=model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.model_type = task
        print("LLM model loaded successfully")
    
    def extract_audio(self, video_path, output_audio_path=None):
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to input video file
            output_audio_path: Path for output audio file (optional)
            
        Returns:
            Path to extracted audio file
        """
        if output_audio_path is None:
            output_audio_path = tempfile.mktemp(suffix=".wav")
        
        print(f"Extracting audio from {video_path}")
        
        # Use ffmpeg to extract audio
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file
            output_audio_path
        ]
        
        try:
            subprocess.run(command, check=True, capture_output=True)
            print(f"Audio extracted to {output_audio_path}")
            return output_audio_path
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            raise
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text
        """
        if self.transcription_pipeline is None:
            raise ValueError("Transcription model not loaded. Call load_transcription_model() first.")
        
        print(f"Transcribing audio from {audio_path}")
        result = self.transcription_pipeline(audio_path)
        transcription = result["text"]
        print(f"Transcription complete: {len(transcription)} characters")
        return transcription
    
    def analyze_answer(self, question, answer, context="job interview"):
        """
        Analyze interview answer using LLM
        
        Args:
            question: The interview question asked
            answer: The transcribed answer
            context: Context of the interview
            
        Returns:
            Detailed feedback on the answer
        """
        if self.llm_pipeline is None:
            raise ValueError("LLM model not loaded. Call load_llm_model() first.")
        
        print("Analyzing answer with LLM...")
        
        # Simplified prompt for better results with smaller models
        prompt = f"""Question: {question}
Answer: {answer}

Rate this interview answer (1-10) and list 3 strengths and 3 areas to improve:"""

        # Handle different model types
        if self.model_type == "text2text-generation":
            # T5/FLAN models: simple text-to-text with better generation params
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=300,
                min_length=50,
                temperature=0.8,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
            feedback = outputs[0]["generated_text"]
        else:
            # Chat models: use messages format
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            outputs = self.llm_pipeline(
                messages,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
            )
            
            feedback = outputs[0]["generated_text"][-1]["content"]
        
        print("Analysis complete")
        return feedback
    
    def analyze_video(self, video_path, question, context="job interview"):
        """
        Complete pipeline: extract audio, transcribe, and analyze
        
        Args:
            video_path: Path to video file
            question: The interview question
            context: Interview context
            
        Returns:
            Dictionary with transcription and feedback
        """
        # Extract audio
        audio_path = self.extract_audio(video_path)
        
        try:
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Analyze answer
            feedback = self.analyze_answer(question, transcription, context)
            
            return {
                "transcription": transcription,
                "feedback": feedback,
                "question": question
            }
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = InterviewAnalyzer(use_gpu=True)
    
    # Load models
    analyzer.load_transcription_model("openai/whisper-base")
    
    # RECOMMENDED: Use one of these better models for LLM analysis
    # Option 1: Best quality (needs ~6GB RAM, but much better results)
    # analyzer.load_llm_model("microsoft/Phi-3-mini-4k-instruct")
    
    # Option 2: Larger FLAN-T5 (needs ~3GB RAM, better than base)
    # analyzer.load_llm_model("google/flan-t5-large")
    
    # Option 3: Current - smallest model (often gives poor results)
    analyzer.load_llm_model("google/flan-t5-large")
    
    # Analyze a video
    video_path = "/home/mohit/Downloads/project_mlops/MLOps-Project/Data Pipeline/Data/video1.webm"
    question = "Tell me about yourself"
    
    if os.path.exists(video_path):
        result = analyzer.analyze_video(video_path, question)
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"\nQuestion: {result['question']}")
        print(f"\nTranscription:\n{result['transcription']}")
        print(f"\nFeedback:\n{result['feedback']}")
    else:
        print(f"Video file not found: {video_path}")
        print("\nTo use this script:")
        print("1. Install dependencies: pip install transformers torch opencv-python")
        print("2. Install ffmpeg: apt-get install ffmpeg (Linux) or brew install ffmpeg (Mac)")
        print("3. Provide a video file path")
        print("\nRECOMMENDED: Use microsoft/Phi-3-mini-4k-instruct for much better feedback!")