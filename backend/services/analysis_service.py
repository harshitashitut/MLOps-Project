"""
Analysis Service: Orchestrates the entire video analysis pipeline
Handles progress tracking, error handling, and result standardization
"""
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from pipeline.preprocessing import extract_frames, extract_audio, cleanup_temp_files
from pipeline.visual_analysis import analyze_pose_mediapipe, analyze_visual_gemini, combine_visual_analyses
from pipeline.audio_analysis import transcribe_with_whisper, analyze_emotion_wav2vec, compute_vocal_metrics, combine_audio_analyses
from pipeline.content_analysis import analyze_content_gemini
from pipeline.aggregation import aggregate_all_results
from services.result_mapper import ResultMapper
from utils.db_helper import update_video_progress, update_video_status, save_results_to_db

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for orchestrating video analysis pipeline
    Handles all business logic, leaving app.py for HTTP only
    """
    
    def __init__(self):
        self.result_mapper = ResultMapper()
    
    def process_video(
        self,
        video_path: str,
        video_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Process a video through the entire analysis pipeline
        
        Args:
            video_path: Path to uploaded video file
            video_id: UUID of video
            user_id: UUID of user
            
        Returns:
            Complete analysis results ready for API response
            
        Raises:
            PipelineError: If any step fails
        """
        start_time = time.time()
        frames_dir = None
        audio_path = None
        
        try:
            logger.info(f"🎬 [{video_id[:8]}] Starting pipeline for {Path(video_path).name}")
            
            # ==================== STEP 1: PREPROCESSING ====================
            preprocessing_result = self._run_preprocessing(video_path, video_id)
            frames_dir = preprocessing_result['frames_dir']
            frame_paths = preprocessing_result['frame_paths']
            audio_path = preprocessing_result['audio_path']
            audio_duration = preprocessing_result['audio_duration']
            
            # ==================== STEP 2: VISUAL ANALYSIS ====================
            visual_results = self._run_visual_analysis(frame_paths, video_id)
            
            # ==================== STEP 3: AUDIO ANALYSIS ====================
            audio_results = self._run_audio_analysis(audio_path, video_id)
            
            # ==================== STEP 4: CONTENT ANALYSIS ====================
            content_results = self._run_content_analysis(
                audio_results['transcript_data']['full_transcript'],
                video_id
            )
            
            # ==================== STEP 5: AGGREGATION ====================
            aggregated_results = self._run_aggregation(
                visual_results,
                audio_results,
                content_results,
                video_id
            )
            
            # ==================== STEP 6: SAVE TO DATABASE ====================
            processing_time = time.time() - start_time
            self._save_results(
                video_id,
                user_id,
                aggregated_results,
                processing_time
            )
            
            # ==================== CLEANUP ====================
            self._cleanup(frames_dir, audio_path, video_id)
            
            # Mark complete
            update_video_progress(video_id, 100, "Analysis complete")
            update_video_status(video_id, "completed")
            
            total_time = time.time() - start_time
            logger.info(f"🎉 [{video_id[:8]}] Pipeline complete in {total_time:.2f}s")
            
            # Return standardized response
            return {
                "status": "success",
                "video_id": video_id,
                "processing_time_seconds": total_time,
                "results": aggregated_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            # Handle failure
            logger.error(f"❌ [{video_id[:8]}] Pipeline failed: {e}", exc_info=True)
            
            # Cleanup on error
            if frames_dir or audio_path:
                self._cleanup(frames_dir, audio_path, video_id)
            
            # Update status
            update_video_status(video_id, "failed")
            
            raise PipelineError(f"Analysis failed at {self._get_current_step()}: {str(e)}")
    
    def _run_preprocessing(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """
        Step 1: Extract frames and audio
        Progress: 10% → 20%
        """
        step_name = "Preprocessing"
        logger.info(f"🔧 [{video_id[:8]}] STEP 1/6: {step_name}")
        update_video_progress(video_id, 10, "Extracting frames and audio")
        
        step_start = time.time()
        
        try:
            # Extract frames
            frame_data = extract_frames(str(video_path), video_id=video_id)
            frames_dir = frame_data['frames_dir']
            frame_paths = frame_data['frame_paths']
            logger.info(f"   ✅ Extracted {len(frame_paths)} frames")
            
            # Extract audio
            audio_data = extract_audio(str(video_path), video_id=video_id)
            audio_path = audio_data['audio_path']
            audio_duration = audio_data.get('duration', 0)
            logger.info(f"   ✅ Extracted audio: {audio_duration:.1f}s")
            
            step_time = time.time() - step_start
            logger.info(f"✅ [{video_id[:8]}] {step_name} complete in {step_time:.2f}s")
            update_video_progress(video_id, 20, "Preprocessing complete")
            
            return {
                'frames_dir': frames_dir,
                'frame_paths': frame_paths,
                'audio_path': audio_path,
                'audio_duration': audio_duration
            }
            
        except Exception as e:
            logger.error(f"❌ [{video_id[:8]}] {step_name} failed: {e}")
            raise PipelineError(f"{step_name} failed: {e}")
    
    def _run_visual_analysis(self, frame_paths: list, video_id: str) -> Dict[str, Any]:
        """
        Step 2: Analyze body language and visual presentation
        Progress: 25% → 45%
        """
        step_name = "Visual Analysis"
        logger.info(f"👁️  [{video_id[:8]}] STEP 2/6: {step_name}")
        update_video_progress(video_id, 25, "Analyzing body language")
        
        step_start = time.time()
        
        try:
            # MediaPipe pose analysis
            mediapipe_start = time.time()
            mediapipe_results = analyze_pose_mediapipe(frame_paths)
            logger.info(f"   ✅ MediaPipe: {time.time() - mediapipe_start:.2f}s")
            
            # Gemini visual analysis
            update_video_progress(video_id, 35, "Analyzing visual presentation")
            gemini_start = time.time()
            gemini_results = analyze_visual_gemini(frame_paths)
            logger.info(f"   ✅ Gemini Vision: {time.time() - gemini_start:.2f}s")
            
            # Combine results
            visual_results = combine_visual_analyses(mediapipe_results, gemini_results)
            
            step_time = time.time() - step_start
            logger.info(f"✅ [{video_id[:8]}] {step_name} complete in {step_time:.2f}s")
            update_video_progress(video_id, 45, "Visual analysis complete")
            
            return visual_results
            
        except Exception as e:
            logger.error(f"❌ [{video_id[:8]}] {step_name} failed: {e}")
            raise PipelineError(f"{step_name} failed: {e}")
    
    def _run_audio_analysis(self, audio_path: str, video_id: str) -> Dict[str, Any]:
        """
        Step 3: Transcribe and analyze audio
        Progress: 50% → 70%
        """
        step_name = "Audio Analysis"
        logger.info(f"🔊 [{video_id[:8]}] STEP 3/6: {step_name}")
        update_video_progress(video_id, 50, "Transcribing audio")
        
        step_start = time.time()
        
        try:
            # Whisper transcription
            whisper_start = time.time()
            transcription = transcribe_with_whisper(audio_path)
            transcript_length = len(transcription.get('transcript', ''))
            logger.info(f"   ✅ Whisper: {time.time() - whisper_start:.2f}s ({transcript_length} chars)")
            
            # Emotion analysis
            update_video_progress(video_id, 60, "Analyzing vocal delivery")
            wav2vec_start = time.time()
            emotion = analyze_emotion_wav2vec(audio_path)
            dominant_emotion = emotion.get('dominant_emotion', 'neutral')
            logger.info(f"   ✅ Emotion: {time.time() - wav2vec_start:.2f}s (dominant: {dominant_emotion})")
            
            # Vocal metrics
            vocal_metrics = compute_vocal_metrics(
                transcription['transcript'],
                transcription['duration']
            )
            wpm = vocal_metrics.get('wpm', 0)
            filler_count = vocal_metrics.get('filler_count', 0)
            logger.info(f"   ✅ Metrics: {wpm:.0f} WPM, {filler_count} fillers")
            
            # Combine all audio analyses
            audio_results = combine_audio_analyses(transcription, emotion, vocal_metrics)
            
            step_time = time.time() - step_start
            logger.info(f"✅ [{video_id[:8]}] {step_name} complete in {step_time:.2f}s")
            update_video_progress(video_id, 70, "Audio analysis complete")
            
            return audio_results
            
        except Exception as e:
            logger.error(f"❌ [{video_id[:8]}] {step_name} failed: {e}")
            raise PipelineError(f"{step_name} failed: {e}")
    
    def _run_content_analysis(self, transcript: str, video_id: str) -> Dict[str, Any]:
        """
        Step 4: Analyze speech content quality
        Progress: 75% → 85%
        """
        step_name = "Content Analysis"
        logger.info(f"📝 [{video_id[:8]}] STEP 4/6: {step_name}")
        update_video_progress(video_id, 75, "Analyzing content quality")
        
        step_start = time.time()
        
        try:
            content_results = analyze_content_gemini(transcript)
            
            content_score = content_results.get('content_analysis', {}).get('overall_content_score', 0)
            logger.info(f"   ✅ Content score: {content_score}/100")
            
            step_time = time.time() - step_start
            logger.info(f"✅ [{video_id[:8]}] {step_name} complete in {step_time:.2f}s")
            update_video_progress(video_id, 85, "Content analysis complete")
            
            return content_results
            
        except Exception as e:
            logger.error(f"❌ [{video_id[:8]}] {step_name} failed: {e}")
            raise PipelineError(f"{step_name} failed: {e}")
    
    def _run_aggregation(
        self,
        visual_results: Dict,
        audio_results: Dict,
        content_results: Dict,
        video_id: str
    ) -> Dict[str, Any]:
        """
        Step 5: Combine all analyses into final report
        Progress: 90% → 95%
        """
        step_name = "Aggregation"
        logger.info(f"🔄 [{video_id[:8]}] STEP 5/6: {step_name}")
        update_video_progress(video_id, 90, "Generating final report")
        
        step_start = time.time()
        
        try:
            aggregated_results = aggregate_all_results(
                visual_results,
                audio_results,
                content_results
            )
            
            overall_score = aggregated_results.get('overall_score', 0)
            logger.info(f"   ✅ Overall score: {overall_score}/100")
            
            step_time = time.time() - step_start
            logger.info(f"✅ [{video_id[:8]}] {step_name} complete in {step_time:.2f}s")
            
            return aggregated_results
            
        except Exception as e:
            logger.error(f"❌ [{video_id[:8]}] {step_name} failed: {e}")
            raise PipelineError(f"{step_name} failed: {e}")
    
    def _save_results(
        self,
        video_id: str,
        user_id: str,
        aggregated_results: Dict,
        processing_time: float
    ) -> None:
        """
        Step 6: Save results to database using ResultMapper
        Progress: 95% → 100%
        """
        step_name = "Saving Results"
        logger.info(f"💾 [{video_id[:8]}] STEP 6/6: {step_name}")
        update_video_progress(video_id, 95, "Saving results")
        
        step_start = time.time()
        
        try:
            # Map results to DB schema
            db_record = self.result_mapper.map_to_db_schema(
                video_id=video_id,
                user_id=user_id,
                aggregation_output=aggregated_results,
                processing_time_seconds=processing_time
            )
            
            # Save to database
            from utils.db_helper import supabase
            result = supabase.table("analysis_results").insert(db_record).execute()
            
            if not result.data:
                raise Exception("Database insert returned no data")
            
            # Extract and save emotion timeline if exists
            emotion_timeline = self.result_mapper.extract_emotion_timeline(
                video_id, user_id, aggregated_results
            )
            
            if emotion_timeline:
                supabase.table("emotion_timeline").insert(emotion_timeline).execute()
                logger.info(f"   ✅ Saved {len(emotion_timeline)} emotion points")
            
            step_time = time.time() - step_start
            logger.info(f"✅ [{video_id[:8]}] {step_name} complete in {step_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ [{video_id[:8]}] {step_name} failed: {e}")
            raise PipelineError(f"{step_name} failed: {e}")
    
    def _cleanup(
        self,
        frames_dir: Optional[str],
        audio_path: Optional[str],
        video_id: str
    ) -> None:
        """Cleanup temporary files"""
        logger.info(f"🧹 [{video_id[:8]}] Cleaning up temporary files")
        
        cleanup_start = time.time()
        
        try:
            cleanup_temp_files(frames_dir, audio_path)
            cleanup_time = time.time() - cleanup_start
            logger.info(f"   ✅ Cleanup complete in {cleanup_time:.2f}s")
        except Exception as e:
            logger.warning(f"   ⚠️ Cleanup failed: {e}")
    
    def _get_current_step(self) -> str:
        """Get current step for error reporting (can be enhanced with context)"""
        return "unknown step"


class PipelineError(Exception):
    """Custom exception for pipeline failures"""
    pass