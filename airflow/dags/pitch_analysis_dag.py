"""
PitchQuest Video Analysis DAG - Production Grade MLOps Pipeline
5-Task Consolidated Architecture for Compute Efficiency
"""
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import pipeline modules
from pipeline.preprocessing import extract_frames, extract_audio
from pipeline.visual_analysis import (
    analyze_pose_mediapipe, 
    analyze_visual_gemini, 
    combine_visual_analyses
)
from pipeline.audio_analysis import (
    transcribe_with_whisper,
    analyze_emotion_wav2vec,
    compute_vocal_metrics,
    combine_audio_analyses
)
from pipeline.content_analysis import analyze_content_gemini
from pipeline.aggregation import aggregate_all_results, save_results

logger = logging.getLogger(__name__)

# DAG Configuration
default_args = {
    'owner': 'pitchquest',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}


def preprocess_task(**context):
    """
    Task 1: Extract frames and audio from video
    Lightweight - just file I/O operations
    Returns: Paths stored as artifacts
    """
    try:
        video_id = context['dag_run'].conf.get('video_id', 'demo_video_2')
        video_path = context['dag_run'].conf.get('video_path', f"data/input/{video_id}.mp4")
        
        logger.info(f"Starting preprocessing for video: {video_id}")
        
        # Extract frames
        frame_result = extract_frames(video_path, **context)
        frame_paths = frame_result if isinstance(frame_result, list) else frame_result.get('frame_paths', [])
        
        # Extract audio
        audio_result = extract_audio(video_path, **context)
        audio_path = audio_result if isinstance(audio_result, str) else audio_result.get('audio_path')
        
        logger.info(f"✅ Preprocessing complete: {len(frame_paths)} frames, audio at {audio_path}")
        
        return {
            'video_id': video_id,
            'video_path': video_path,
            'frame_paths': frame_paths,
            'audio_path': audio_path
        }
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def visual_analysis_task(**context):
    """
    Task 2: Analyze visual presentation
    Consolidated: MediaPipe + Gemini Vision in single execution
    Models stay in memory - no context switching
    """
    try:
        ti = context['task_instance']
        preprocess_data = ti.xcom_pull(task_ids='preprocess')
        frame_paths = preprocess_data['frame_paths']
        
        logger.info(f"Starting visual analysis on {len(frame_paths)} frames")
        
        # Run pose analysis (MediaPipe - loads model once)
        pose_data = analyze_pose_mediapipe(frame_paths, **context)
        
        # Run visual analysis (Gemini Vision)
        visual_gemini = analyze_visual_gemini(frame_paths, **context)
        
        # Combine results
        visual_combined = combine_visual_analyses(pose_data, visual_gemini, **context)
        
        logger.info("✅ Visual analysis complete")
        return visual_combined
        
    except Exception as e:
        logger.error(f"Visual analysis failed: {e}")
        raise


def audio_analysis_task(**context):
    """
    Task 3: Analyze audio/speech
    Consolidated: Whisper + Wav2Vec2 + Metrics in single execution
    CRITICAL: This is the heaviest task (loads 2 large models)
    Keeps all audio data in memory - no XCom serialization of transcripts
    """
    try:
        ti = context['task_instance']
        preprocess_data = ti.xcom_pull(task_ids='preprocess')
        audio_path = preprocess_data['audio_path']
        
        logger.info(f"Starting audio analysis on {audio_path}")
        
        # Step 1: Transcribe (Whisper API - lightweight)
        logger.info("Running Whisper transcription...")
        transcription = transcribe_with_whisper(audio_path, **context)
        
        # Step 2: Emotion analysis (Wav2Vec2 - heavy model)
        logger.info("Running emotion analysis (Wav2Vec2)...")
        emotion = analyze_emotion_wav2vec(audio_path, **context)
        
        # Step 3: Compute vocal metrics (lightweight - just text processing)
        logger.info("Computing vocal metrics...")
        vocal_metrics = compute_vocal_metrics(
            transcription['transcript'], 
            transcription['duration'], 
            **context
        )
        
        # Step 4: Combine all audio analyses (in memory - no XCom)
        logger.info("Combining audio analyses...")
        audio_combined = combine_audio_analyses(
            transcription, 
            emotion, 
            vocal_metrics, 
            **context
        )
        
        logger.info("✅ Audio analysis complete")
        return audio_combined
        
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise


def content_analysis_task(**context):
    """
    Task 4: Analyze speech content
    Depends on audio_analysis for transcript
    Uses Gemini for content evaluation
    """
    try:
        ti = context['task_instance']
        audio_data = ti.xcom_pull(task_ids='audio_analysis')
        transcript = audio_data['transcript_data']['full_transcript']
        
        logger.info(f"Starting content analysis on {len(transcript)} char transcript")
        
        # Analyze content with Gemini
        content_result = analyze_content_gemini(transcript, **context)
        
        logger.info("✅ Content analysis complete")
        return content_result
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise


def aggregate_and_save_task(**context):
    """
    Task 5: Final aggregation and save
    Combines all analyses using Gemini Pro
    Saves final JSON output
    """
    try:
        ti = context['task_instance']
        
        # Pull results from all previous tasks
        preprocess_data = ti.xcom_pull(task_ids='preprocess')
        visual_data = ti.xcom_pull(task_ids='visual_analysis')
        audio_data = ti.xcom_pull(task_ids='audio_analysis')
        content_data = ti.xcom_pull(task_ids='content_analysis')
        
        video_id = preprocess_data['video_id']
        
        logger.info(f"Starting final aggregation for {video_id}")
        
        # Aggregate all results with Gemini Pro
        final_results = aggregate_all_results(
            visual_data,
            audio_data,
            content_data,
            **context
        )
        
        # Save to JSON (placeholder for DB)
        output_path = save_results(final_results, video_id, **context)
        
        logger.info(f"✅ Pipeline complete! Results saved to {output_path}")
        
        return {
            'video_id': video_id,
            'output_path': output_path,
            'overall_score': final_results.get('overall_score', 0),
            'performance_level': final_results.get('performance_level', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise


# Define the DAG
with DAG(
    dag_id='pitchquest_video_analysis',
    default_args=default_args,
    description='End-to-end video pitch analysis pipeline',
    schedule_interval=None,  # Triggered via API
    start_date=datetime(2025, 11, 23),
    catchup=False,
    max_active_runs=1,  # Process one video at a time
    tags=['mlops', 'video-analysis', 'pitchquest'],
) as dag:

    # Task 1: Preprocessing
    preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess_task,
        provide_context=True,
    )

    # Task 2: Visual Analysis (MediaPipe + Gemini Vision)
    visual_analysis = PythonOperator(
        task_id='visual_analysis',
        python_callable=visual_analysis_task,
        provide_context=True,
    )

    # Task 3: Audio Analysis (Whisper + Wav2Vec2 + Metrics)
    audio_analysis = PythonOperator(
        task_id='audio_analysis',
        python_callable=audio_analysis_task,
        provide_context=True,
    )

    # Task 4: Content Analysis (Gemini LLM)
    content_analysis = PythonOperator(
        task_id='content_analysis',
        python_callable=content_analysis_task,
        provide_context=True,
    )

    # Task 5: Aggregation & Save (Gemini Pro)
    aggregate_and_save = PythonOperator(
        task_id='aggregate_and_save',
        python_callable=aggregate_and_save_task,
        provide_context=True,
    )

    # Define task dependencies
    # Parallel execution of visual and audio after preprocessing
    preprocess >> [visual_analysis, audio_analysis]
    
    # Content analysis depends on audio (needs transcript)
    audio_analysis >> content_analysis
    
    # Final aggregation needs all three analyses
    [visual_analysis, audio_analysis, content_analysis] >> aggregate_and_save