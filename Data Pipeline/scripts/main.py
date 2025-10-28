"""
Interview Analysis Backend
Analyzes interview videos using speech transcription and LLM feedback
Saves transcriptions to local storage
"""

import sys
sys.path.append("/home/mohit/.local/lib/python3.13/site-packages")

# Add project root to path for config imports
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from config.logging_config import get_logger, PipelineLogger

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
from datetime import datetime
import time

# Initialize logging
logger = get_logger(__name__, component='pipeline')
pipeline_logger = PipelineLogger()

class InterviewAnalyzer:
    def __init__(self, use_gpu=True, storage_dir="store"):
        """
        Initialize the analyzer with speech recognition and LLM models
        
        Args:
            use_gpu: Whether to use GPU acceleration if available
            storage_dir: Directory to save transcriptions (default: "store")
        """
        logger.info("Initializing InterviewAnalyzer")
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Set up storage directory
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"Transcriptions will be saved to: {self.storage_dir.absolute()}")
        
        # Initialize speech recognition model (Whisper)
        self.transcription_pipeline = None
        self.llm_pipeline = None
        
    def load_transcription_model(self, model_name="openai/whisper-base"):
        """
        Load Whisper model for speech-to-text
        
        Args:
            model_name: Hugging Face model ID for transcription
        """
        logger.info(f"Loading transcription model: {model_name}")
        start_time = time.time()
        
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
        
        load_time = time.time() - start_time
        logger.info(f"Transcription model loaded successfully in {load_time:.2f} seconds")
    
    def load_llm_model(self, model_name="google/flan-t5-base"):
        """
        Load LLM for answer analysis
        
        Args:
            model_name: Hugging Face model ID for text generation
        """
        logger.info(f"Loading LLM model: {model_name}")
        start_time = time.time()
        
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
        
        load_time = time.time() - start_time
        logger.info(f"LLM model loaded successfully in {load_time:.2f} seconds")
    
    def get_video_metadata(self, video_path):
        """
        Get video metadata using ffprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video metadata
        """
        try:
            command = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            import json
            metadata = json.loads(result.stdout)
            
            # Extract useful info
            duration = float(metadata['format'].get('duration', 0))
            size_mb = float(metadata['format'].get('size', 0)) / (1024 * 1024)
            format_name = metadata['format'].get('format_name', 'unknown')
            
            return {
                'duration': duration,
                'size_mb': size_mb,
                'format': format_name
            }
        except Exception as e:
            logger.warning(f"Could not extract video metadata: {e}")
            return None
    
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
        
        # Get video metadata
        video_metadata = self.get_video_metadata(video_path)
        if video_metadata:
            logger.info(f"Video metadata - Duration: {video_metadata['duration']:.2f}s, "
                       f"Size: {video_metadata['size_mb']:.2f}MB, "
                       f"Format: {video_metadata['format']}")
        
        logger.info(f"Extracting audio from {video_path}")
        start_time = time.time()
        
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
            extraction_time = time.time() - start_time
            
            # Get audio file size
            audio_size_mb = os.path.getsize(output_audio_path) / (1024 * 1024)
            
            logger.info(f"Audio extracted in {extraction_time:.2f}s - "
                       f"Size: {audio_size_mb:.2f}MB - "
                       f"Path: {output_audio_path}")
            return output_audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
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
        # Generate filename based on video name and timestamp
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_name}_{timestamp}.txt"
        
        output_path = self.storage_dir / filename
        
        logger.info(f"Saving analysis to {output_path}")
        
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
        
        # Log file size
        file_size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"Analysis report saved successfully - Size: {file_size_kb:.2f}KB")
        return str(output_path)
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text
        """
        if self.transcription_pipeline is None:
            logger.error("Transcription model not loaded")
            raise ValueError("Transcription model not loaded. Call load_transcription_model() first.")
        
        logger.info(f"Starting transcription of audio file")
        start_time = time.time()
        
        result = self.transcription_pipeline(audio_path)
        transcription = result["text"]
        
        transcription_time = time.time() - start_time
        logger.info(f"Transcription complete in {transcription_time:.2f}s - "
                   f"Length: {len(transcription)} characters, "
                   f"Words: {len(transcription.split())}")
        
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
            logger.error("LLM model not loaded")
            raise ValueError("LLM model not loaded. Call load_llm_model() first.")
        
        logger.info("Starting LLM analysis")
        start_time = time.time()
        
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
        
        analysis_time = time.time() - start_time
        logger.info(f"LLM analysis complete in {analysis_time:.2f}s - "
                   f"Feedback length: {len(feedback)} characters")
        
        return feedback
    
    def analyze_video(self, video_path, question, context="job interview", save_analysis=True):
        """
        Complete pipeline: extract audio, transcribe, and analyze
        
        Args:
            video_path = "../Data/video1.webm"
            question: The interview question
            context: Interview context
            save_analysis: Whether to save full analysis (transcription + feedback) to file
            
        Returns:
            Dictionary with transcription, feedback, and file path
        """
        logger.info(f"Starting video analysis pipeline for: {os.path.basename(video_path)}")
        pipeline_start_time = time.time()
        
        # Extract audio
        audio_path = self.extract_audio(video_path)
        
        try:
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Analyze answer
            feedback = self.analyze_answer(question, transcription, context)
            
            # Save complete analysis to file
            analysis_file = None
            if save_analysis:
                analysis_file = self.save_analysis(transcription, feedback, video_path, question)
            
            total_time = time.time() - pipeline_start_time
            logger.info(f"Video analysis pipeline completed successfully in {total_time:.2f}s")
            
            return {
                "transcription": transcription,
                "feedback": feedback,
                "question": question,
                "analysis_file": analysis_file,
                "processing_time": total_time
            }
        except Exception as e:
            logger.error(f"Video analysis pipeline failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Temporary audio file cleaned up")


# Example usage
if __name__ == "__main__":
    # Log pipeline start
    pipeline_logger.log_pipeline_start(
        logger,
        "Interview Analysis Pipeline",
        config={
            'transcription_model': 'openai/whisper-base',
            'llm_model': 'google/flan-t5-large',
            'gpu_enabled': True,
            'storage_dir': '../store'
        }
    )
    
    pipeline_start = time.time()
    
    try:
        # Initialize analyzer with storage directory
        analyzer = InterviewAnalyzer(
            use_gpu=True,
            storage_dir = "store"
        )
        
        # Load models once (reused for all videos)
        analyzer.load_transcription_model("openai/whisper-base")
        analyzer.load_llm_model("google/flan-t5-large")
        
        # Find all video files in Data folder
        data_folder = Path(__file__).parent.parent / "Data"
        video_extensions = ['.webm', '.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(data_folder.glob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in {data_folder.absolute()}")
            logger.info(f"Supported formats: {', '.join(video_extensions)}")
        else:
            logger.info(f"Found {len(video_files)} video(s) to process")
            
            # Default question (you can customize per video if needed)
            default_question = "Tell me about yourself"
            
            # Process each video
            successful = 0
            failed = 0
            
            for i, video_path in enumerate(video_files, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing video {i}/{len(video_files)}: {video_path.name}")
                logger.info(f"{'='*60}")
                
                try:
                    result = analyzer.analyze_video(
                        str(video_path), 
                        default_question,
                        save_analysis=True
                    )
                    
                    print(f"\n{'='*60}")
                    print(f"ANALYSIS RESULTS - Video {i}/{len(video_files)}")
                    print(f"{'='*60}")
                    print(f"Video: {video_path.name}")
                    print(f"Question: {result['question']}")
                    print(f"\nTranscription ({len(result['transcription'])} chars):\n{result['transcription'][:200]}...")
                    print(f"\nFeedback:\n{result['feedback']}")
                    print(f"\nProcessing Time: {result['processing_time']:.2f}s")
                    
                    if result['analysis_file']:
                        print(f"✓ Full analysis saved to: {result['analysis_file']}")
                    
                    successful += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {video_path.name}: {str(e)}", exc_info=True)
                    failed += 1
                    continue
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Total videos: {len(video_files)}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            
            # Log statistics
            pipeline_logger.log_data_stats(
                logger,
                "Video Processing Summary",
                {
                    'total_videos': len(video_files),
                    'successful': successful,
                    'failed': failed,
                    'success_rate': f"{(successful/len(video_files)*100):.1f}%"
                }
            )
        
        # Log pipeline completion
        total_duration = time.time() - pipeline_start
        pipeline_logger.log_pipeline_end(
            logger,
            "Interview Analysis Pipeline",
            status="SUCCESS",
            duration=total_duration
        )
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        total_duration = time.time() - pipeline_start
        pipeline_logger.log_pipeline_end(
            logger,
            "Interview Analysis Pipeline",
            status="FAILED",
            duration=total_duration
        )
        raise