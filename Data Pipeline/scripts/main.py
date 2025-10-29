"""
Interview Analysis Backend
Analyzes interview videos using speech transcription and LLM feedback
Saves transcriptions to local storage
"""

import sys
sys.path.append("/home/mohit/.local/lib/python3.13/site-packages")

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
from datetime import datetime

class InterviewAnalyzer:
    def __init__(self, use_gpu=True, storage_dir="store"):
        """
        Initialize the analyzer with speech recognition and LLM models
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            storage_dir: Directory to save transcriptions (default: "store")
        """
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Set up storage directory
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        print(f"Transcriptions will be saved to: {self.storage_dir.absolute()}")
        
        # Initialize speech recognition model (Whisper)
        self.transcription_pipeline = None
        self.llm_pipeline = None
        
    def load_transcription_model(self, model_name="openai/whisper-base"):
        """
        Load Whisper model for speech-to-text
        
        Args:
            model_name: Hugging Face model ID for transcription
        """
        try:
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
            
        except Exception as e:
            print(f"Error loading transcription model: {e}")
            sys.exit(1)
    
    def load_llm_model(self, model_name="google/flan-t5-base"):
        """
        Load LLM for answer analysis
        
        Args:
            model_name: Hugging Face model ID for text generation
        """
        try:
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
            
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            sys.exit(1)
    
    def extract_audio(self, video_path, output_audio_path=None):
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to input video file
            output_audio_path: Path for output audio file (optional)
            
        Returns:
            Path to extracted audio file
        """
        try:
            # Validate video file exists
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                sys.exit(1)
            
            # Validate file is not empty
            if os.path.getsize(video_path) == 0:
                print(f"Error: Video file is empty: {video_path}")
                sys.exit(1)
            
            # Validate video format
            valid_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v']
            file_extension = os.path.splitext(video_path)[1].lower()
            
            if file_extension not in valid_extensions:
                print(f"Error: Invalid video format: {file_extension}")
                sys.exit(1)
            
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
            
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Audio extracted to {output_audio_path}")
            return output_audio_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: ffmpeg failed")
            sys.exit(1)
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg")
            sys.exit(1)
        except Exception as e:
            print(f"Error during audio extraction: {e}")
            sys.exit(1)
    
    def save_analysis(self, transcription, feedback, video_path, question=None):
        """
        Save transcription and LLM feedback to a text file in the storage directory
        
        Args:
            transcription: The transcription text
            feedback: LLM analysis feedback
            video_path: Original video path (used for naming)
            question: Optional question to include in the file
            
        Returns:
            Path to saved analysis file
        """
        try:
            # Generate filename based on video name and timestamp
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{video_name}_{timestamp}.txt"
            
            output_path = self.storage_dir / filename
            
            # Write analysis to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("INTERVIEW ANALYSIS REPORT\n")
                f.write("="*60 + "\n\n")
                
                # Metadata
                f.write(f"Video: {Path(video_path).name}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Transcription length: {len(transcription)} characters\n\n")
                
                # Question
                if question:
                    f.write("="*60 + "\n")
                    f.write("QUESTION\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"{question}\n\n")
                
                # Transcription
                f.write("="*60 + "\n")
                f.write("TRANSCRIPTION\n")
                f.write("="*60 + "\n\n")
                f.write(transcription)
                f.write("\n\n")
                
                # LLM Feedback
                f.write("="*60 + "\n")
                f.write("AI FEEDBACK\n")
                f.write("="*60 + "\n\n")
                f.write(feedback)
                f.write("\n")
            
            print(f"Analysis saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error saving analysis: {e}")
            sys.exit(1)
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text
        """
        try:
            if self.transcription_pipeline is None:
                print("Error: Transcription model not loaded")
                sys.exit(1)
            
            print(f"Transcribing audio from {audio_path}")
            result = self.transcription_pipeline(audio_path)
            transcription = result["text"]
            print(f"Transcription complete: {len(transcription)} characters")
            
            return transcription
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)
    
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
        try:
            if self.llm_pipeline is None:
                print("Error: LLM model not loaded")
                sys.exit(1)
            
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
            
        except Exception as e:
            print(f"Error during LLM analysis: {e}")
            sys.exit(1)
    
    def analyze_video(self, video_path, question, context="job interview", save_analysis=True):
        """
        Complete pipeline: extract audio, transcribe, and analyze
        
        Args:
            video_path: Path to video file
            question: The interview question
            context: Interview context
            save_analysis: Whether to save full analysis (transcription + feedback) to file
            
        Returns:
            Dictionary with transcription, feedback, and file path
        """
        try:
            # Validate inputs
            if not video_path:
                print("Error: video_path cannot be empty")
                sys.exit(1)
            
            if not question or not question.strip():
                print("Error: question cannot be empty")
                sys.exit(1)
            
            if not isinstance(video_path, (str, Path)):
                print(f"Error: video_path must be a string or Path")
                sys.exit(1)
            
            if not isinstance(question, str):
                print(f"Error: question must be a string")
                sys.exit(1)
            
            # Check models are loaded
            if self.transcription_pipeline is None:
                print("Error: Transcription model not loaded")
                sys.exit(1)
            
            if self.llm_pipeline is None:
                print("Error: LLM model not loaded")
                sys.exit(1)
            
            # Extract audio
            audio_path = self.extract_audio(video_path)
            
            try:
                # Transcribe audio
                transcription = self.transcribe_audio(audio_path)
                
                # Check if transcription is empty or too short
                if not transcription or len(transcription.strip()) < 10:
                    print("Error: Transcription is too short or empty")
                    sys.exit(1)
                
                # Analyze answer
                feedback = self.analyze_answer(question, transcription, context)
                
                # Save complete analysis to file
                analysis_file = None
                if save_analysis:
                    analysis_file = self.save_analysis(transcription, feedback, video_path, question)
                
                return {
                    "transcription": transcription,
                    "feedback": feedback,
                    "question": question,
                    "analysis_file": analysis_file
                }
            finally:
                # Clean up temporary audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                    
        except Exception as e:
            print(f"Error during video analysis: {e}")
            sys.exit(1)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with storage directory
    analyzer = InterviewAnalyzer(
        use_gpu=True,
        storage_dir="store"  # Transcriptions will be saved here
    )
    
    # Load models
    analyzer.load_transcription_model("openai/whisper-base")
    
    # RECOMMENDED: Use one of these better models for LLM analysis
    # Option 1: Best quality (needs ~6GB RAM, but much better results)
    # analyzer.load_llm_model("microsoft/Phi-3-mini-4k-instruct")
    
    # Option 2: Larger FLAN-T5 (needs ~3GB RAM, better than base)
    analyzer.load_llm_model("google/flan-t5-large")
    
    # Option 3: Smallest model (often gives poor results)
    # analyzer.load_llm_model("google/flan-t5-base")
    
    # Analyze a video
    video_path = "/home/mohit/Downloads/project_mlops/MLOps-Project/Data Pipeline/Data/video1.webm"
    question = "Tell me about yourself"
    
    if os.path.exists(video_path):
        result = analyzer.analyze_video(
            video_path, 
            question,
            save_analysis=True  # Set to False if you don't want to save
        )
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        print(f"\nQuestion: {result['question']}")
        print(f"\nTranscription:\n{result['transcription']}")
        print(f"\nFeedback:\n{result['feedback']}")
        
        if result['analysis_file']:
            print(f"\nFull analysis saved to: {result['analysis_file']}")
    else:
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)