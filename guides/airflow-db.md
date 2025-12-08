# Airflow-DB Integration Guide

**Owner:** Person 3 (MLOps + DevOps Lead)  
**Timeline:** Day 1-2  
**Objective:** Modify existing Airflow pipeline to save results to Supabase instead of JSON files

---

## Overview

Your Airflow pipeline currently saves results to `data/output/results_{video_id}.json`. You need to:
1. Replace JSON file output with Supabase database writes
2. Update video status throughout the pipeline
3. Add error handling for database failures
4. Test the full flow

---

## Current State Assessment

### What Already Works
- ✅ Airflow DAG runs end-to-end
- ✅ All analysis modules produce results
- ✅ JSON output is well-structured

### What Needs to Change
- ❌ `airflow/utils/db_helper.py` - Currently has placeholder functions
- ❌ `airflow/pipeline/aggregation.py` - Saves to JSON, needs to save to DB
- ❌ DAG tasks - Don't update video status in DB
- ❌ Error handling - Doesn't handle DB connection failures

---

## Day 1: Database Integration (4-6 hours)

### Step 1: Install Dependencies (15 min)

Add to `airflow/requirements.txt`:

```txt
supabase==1.2.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
```

Install:
```bash
cd airflow
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Update Configuration (15 min)

**File: `airflow/utils/config.py`**

Add Supabase configuration:

```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Existing configs...
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Add Supabase configs
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Validation
    def __post_init__(self):
        required = {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
            "SUPABASE_URL": self.SUPABASE_URL,
            "SUPABASE_KEY": self.SUPABASE_KEY,
            "DATABASE_URL": self.DATABASE_URL
        }
        
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

config = Config()
```

### Step 3: Create Database Helper (1.5 hours)

**File: `airflow/utils/db_helper.py`**

Replace the entire file with:

```python
from supabase import create_client, Client
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
from typing import Dict, Any, Optional
from airflow/utils/logger import logger
from airflow/utils/config import config

# Supabase client for simple operations
def get_supabase_client() -> Client:
    """Get Supabase client"""
    try:
        client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        logger.info("✅ Supabase client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client: {e}")
        raise

supabase: Client = get_supabase_client()

# SQLAlchemy engine for complex operations
engine = create_engine(config.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ==================== VIDEO OPERATIONS ====================

def update_video_status(video_id: str, status: str, job_id: Optional[str] = None) -> None:
    """
    Update video processing status
    
    Args:
        video_id: Video identifier
        status: New status ('uploaded' | 'processing' | 'completed' | 'failed')
        job_id: Optional Airflow job ID
    """
    try:
        data = {"status": status, "updated_at": datetime.utcnow().isoformat()}
        if job_id:
            data["job_id"] = job_id
        
        result = supabase.table("videos").update(data).eq("video_id", video_id).execute()
        
        if result.data:
            logger.info(f"✅ Updated video status: {video_id} -> {status}")
        else:
            logger.warning(f"⚠️ No video found with ID: {video_id}")
    
    except Exception as e:
        logger.error(f"❌ Failed to update video status: {e}")
        # Don't raise - we don't want to fail the pipeline for status update issues
        # But log it so we can investigate

def get_video_by_id(video_id: str) -> Optional[Dict]:
    """Fetch video record from database"""
    try:
        result = supabase.table("videos").select("*").eq("video_id", video_id).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"❌ Failed to fetch video: {e}")
        return None

# ==================== RESULTS OPERATIONS ====================

def save_results_to_db(video_id: str, results: Dict[str, Any]) -> None:
    """
    Save analysis results to Supabase
    
    Args:
        video_id: Video identifier
        results: Complete analysis results dictionary
        
    Raises:
        Exception: If database save fails
    """
    try:
        logger.info(f"💾 Saving results to database for video: {video_id}")
        
        # Extract data from results structure
        analysis_results = results.get("results", {})
        feedback = results.get("feedback", {})
        metadata = results.get("metadata", {})
        
        # Prepare database record
        db_record = {
            "video_id": video_id,
            "job_id": results.get("job_id", ""),
            "overall_score": analysis_results.get("overall_score", 0),
            "content_analysis": json.dumps(analysis_results.get("content_analysis", {})),
            "delivery_analysis": json.dumps(analysis_results.get("delivery_analysis", {})),
            "visual_analysis": json.dumps(analysis_results.get("visual_analysis", {})),
            "feedback": json.dumps(feedback),
            "metadata": json.dumps(metadata),
            "processing_time_seconds": results.get("processing_time_seconds", 0),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Insert into database
        result = supabase.table("analysis_results").insert(db_record).execute()
        
        if result.data:
            logger.info(f"✅ Results saved to database for video: {video_id}")
            
            # Update video status to completed
            update_video_status(video_id, "completed")
        else:
            logger.error(f"❌ Failed to save results - no data returned")
            raise Exception("Database insert failed")
    
    except Exception as e:
        logger.error(f"❌ Failed to save results to database: {e}", exc_info=True)
        
        # Update video status to failed
        update_video_status(video_id, "failed")
        
        # Still save JSON as backup
        backup_json_path = f"data/output/backup_results_{video_id}.json"
        with open(backup_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📝 Saved backup JSON to: {backup_json_path}")
        
        raise  # Re-raise to fail the task

# ==================== MONITORING OPERATIONS ====================

def save_monitoring_metric(metric_type: str, metric_value: Dict[str, Any]) -> None:
    """
    Save monitoring metric to database
    
    Args:
        metric_type: Type of metric ('drift' | 'system_health' | 'api_cost')
        metric_value: Metric data as dictionary
    """
    try:
        db_record = {
            "metric_type": metric_type,
            "metric_value": json.dumps(metric_value),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = supabase.table("monitoring_metrics").insert(db_record).execute()
        
        if result.data:
            logger.info(f"✅ Saved monitoring metric: {metric_type}")
    
    except Exception as e:
        logger.error(f"❌ Failed to save monitoring metric: {e}")
        # Don't raise - monitoring failures shouldn't break the pipeline

def get_recent_scores(n: int = 50) -> list:
    """Get last N analysis scores for drift detection"""
    try:
        result = supabase.table("analysis_results")\
            .select("overall_score, created_at")\
            .order("created_at", desc=True)\
            .limit(n)\
            .execute()
        
        return result.data if result.data else []
    
    except Exception as e:
        logger.error(f"❌ Failed to fetch recent scores: {e}")
        return []

# ==================== HELPER FUNCTIONS ====================

def test_db_connection() -> bool:
    """Test database connection"""
    try:
        result = supabase.table("videos").select("id").limit(1).execute()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def cleanup_old_results(days: int = 30) -> None:
    """Delete results older than N days (optional maintenance task)"""
    try:
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        result = supabase.table("analysis_results")\
            .delete()\
            .lt("created_at", cutoff_date)\
            .execute()
        
        deleted_count = len(result.data) if result.data else 0
        logger.info(f"🗑️ Cleaned up {deleted_count} old results")
    
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
```

### Step 4: Update Aggregation Module (1 hour)

**File: `airflow/pipeline/aggregation.py`**

Find the `aggregate_results` function (around line 150) and modify the end:

```python
def aggregate_results(**context):
    """
    Aggregate all analysis results into final output
    """
    ti = context['ti']
    
    # Pull data from previous tasks
    preprocess_data = ti.xcom_pull(task_ids='preprocess')
    visual_data = ti.xcom_pull(task_ids='visual_analysis')
    audio_data = ti.xcom_pull(task_ids='audio_analysis')
    content_data = ti.xcom_pull(task_ids='content_analysis')
    
    # CRITICAL: Validate all data exists
    if not all([preprocess_data, visual_data, audio_data, content_data]):
        missing = []
        if not preprocess_data: missing.append('preprocess')
        if not visual_data: missing.append('visual_analysis')
        if not audio_data: missing.append('audio_analysis')
        if not content_data: missing.append('content_analysis')
        
        error_msg = f"Missing data from tasks: {', '.join(missing)}"
        logger.error(f"❌ {error_msg}")
        
        # Update video status to failed
        video_id = context['dag_run'].conf.get('video_id', 'unknown')
        from airflow.utils.db_helper import update_video_status
        update_video_status(video_id, "failed")
        
        raise ValueError(error_msg)
    
    video_id = preprocess_data['video_id']
    job_id = context['dag_run'].run_id
    
    logger.info(f"🔄 Aggregating results for video: {video_id}")
    
    try:
        # ... existing aggregation logic ...
        # (all your scoring and feedback generation code stays the same)
        
        # Calculate overall score
        overall_score = calculate_overall_score(
            content_analysis, 
            delivery_analysis, 
            visual_analysis
        )
        
        # Generate feedback
        feedback = generate_feedback(
            content_analysis,
            delivery_analysis,
            visual_analysis,
            overall_score
        )
        
        # Prepare final results
        final_results = {
            "video_id": video_id,
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_seconds": time.time() - preprocess_data.get('start_time', time.time()),
            "status": "success",
            "results": {
                "overall_score": overall_score,
                "content_analysis": content_analysis,
                "delivery_analysis": delivery_analysis,
                "visual_analysis": visual_analysis
            },
            "feedback": feedback,
            "metadata": {
                "video_duration_seconds": preprocess_data.get('duration', 0),
                "frames_analyzed": len(visual_data.get('frame_paths', [])),
                "models_used": {
                    "transcription": "whisper-1",
                    "emotion": "wav2vec2-large-robust",
                    "vision": "gemini-1.5-flash",
                    "content": "gemini-1.5-flash"
                },
                "costs": {
                    "whisper_api": audio_data.get('cost', 0),
                    "gemini_vision": visual_data.get('cost', 0),
                    "gemini_text": content_data.get('cost', 0),
                    "total_usd": audio_data.get('cost', 0) + visual_data.get('cost', 0) + content_data.get('cost', 0)
                }
            }
        }
        
        # ==================== CHANGED: Save to Database ====================
        from airflow.utils.db_helper import save_results_to_db
        
        save_results_to_db(video_id, final_results)
        logger.info(f"✅ Results saved to database for video: {video_id}")
        
        # OPTIONAL: Still save JSON as backup during testing
        if os.getenv("SAVE_JSON_BACKUP", "false").lower() == "true":
            json_path = f"data/output/results_{video_id}.json"
            with open(json_path, 'w') as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"📝 Backup JSON saved to: {json_path}")
        
        return final_results
    
    except Exception as e:
        logger.error(f"❌ Aggregation failed: {e}", exc_info=True)
        
        # Update video status to failed
        from airflow.utils.db_helper import update_video_status
        update_video_status(video_id, "failed")
        
        raise
```

### Step 5: Update DAG to Track Status (1 hour)

**File: `airflow/dags/pitch_analysis_dag.py`**

Add status update callbacks:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.db_helper import update_video_status

# Add callback functions at the top of the file

def on_task_success(context):
    """Called when any task succeeds"""
    video_id = context['dag_run'].conf.get('video_id')
    task_id = context['task_instance'].task_id
    logger.info(f"✅ Task {task_id} completed for video {video_id}")

def on_task_failure(context):
    """Called when any task fails"""
    video_id = context['dag_run'].conf.get('video_id')
    task_id = context['task_instance'].task_id
    logger.error(f"❌ Task {task_id} failed for video {video_id}")
    
    # Update video status to failed
    update_video_status(video_id, "failed")

def on_dag_failure(context):
    """Called when DAG fails"""
    video_id = context['dag_run'].conf.get('video_id')
    logger.error(f"❌ DAG failed for video {video_id}")
    update_video_status(video_id, "failed")

# Update default_args
default_args = {
    'owner': 'pitchquest',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
    'on_success_callback': on_task_success,  # Add this
    'on_failure_callback': on_task_failure,  # Add this
}

with DAG(
    'pitchquest_video_analysis',
    default_args=default_args,
    description='Analyze pitch videos with multimodal AI',
    schedule_interval=None,  # Triggered via API
    catchup=False,
    max_active_runs=3,
    on_failure_callback=on_dag_failure,  # Add this
    tags=['pitchquest', 'mlops', 'production']
) as dag:
    
    # Add a status update task at the beginning
    def mark_processing(**context):
        """Mark video as processing in database"""
        video_id = context['dag_run'].conf.get('video_id')
        job_id = context['dag_run'].run_id
        update_video_status(video_id, "processing", job_id)
        logger.info(f"🚀 Started processing video: {video_id}")
    
    start_processing = PythonOperator(
        task_id='mark_processing',
        python_callable=mark_processing,
        provide_context=True
    )
    
    # ... existing tasks (preprocess, extract_frames, etc.) ...
    
    # Update task dependencies to include status tracking
    start_processing >> download >> [extract_frames, extract_audio]
    # ... rest of dependencies ...
```

### Step 6: Test Database Connection (30 min)

Create a test script:

**File: `airflow/tests/test_db_connection.py`**

```python
#!/usr/bin/env python3
"""
Test database connection and operations
Run this before testing the full DAG
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.db_helper import (
    test_db_connection,
    update_video_status,
    save_results_to_db,
    get_recent_scores
)
from utils.logger import logger

def test_connection():
    """Test basic connection"""
    print("Testing database connection...")
    success = test_db_connection()
    assert success, "❌ Database connection failed"
    print("✅ Database connection successful")

def test_status_update():
    """Test video status update"""
    print("\nTesting status update...")
    video_id = "test_video_123"
    
    # This should work even if video doesn't exist (it will log a warning)
    update_video_status(video_id, "processing", "test_job_456")
    print("✅ Status update successful")

def test_save_results():
    """Test saving results"""
    print("\nTesting save results...")
    
    # Create mock results
    mock_results = {
        "video_id": "test_video_123",
        "job_id": "test_job_456",
        "timestamp": "2025-12-06T10:00:00Z",
        "processing_time_seconds": 45.2,
        "status": "success",
        "results": {
            "overall_score": 85,
            "content_analysis": {
                "problem_clarity": 8.5,
                "solution_fit": 9.0
            },
            "delivery_analysis": {
                "speaking_pace": "optimal",
                "vocal_confidence": 8.5
            },
            "visual_analysis": {
                "body_language_score": 8.0,
                "posture_quality": "good"
            }
        },
        "feedback": {
            "strengths": ["Clear problem statement"],
            "improvements": ["Add more data"]
        },
        "metadata": {
            "video_duration_seconds": 120,
            "frames_analyzed": 50
        }
    }
    
    try:
        save_results_to_db("test_video_123", mock_results)
        print("✅ Save results successful")
    except Exception as e:
        print(f"❌ Save results failed: {e}")

def test_fetch_scores():
    """Test fetching recent scores"""
    print("\nTesting fetch recent scores...")
    scores = get_recent_scores(n=10)
    print(f"✅ Fetched {len(scores)} recent scores")
    if scores:
        print(f"   Most recent score: {scores[0]['overall_score']}")

if __name__ == "__main__":
    print("=" * 50)
    print("Database Connection Tests")
    print("=" * 50)
    
    try:
        test_connection()
        test_status_update()
        # test_save_results()  # Uncomment when ready to test inserts
        test_fetch_scores()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED")
        print("=" * 50)
    
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

Run the test:
```bash
cd airflow
python tests/test_db_connection.py
```

### End of Day 1 Deliverable
- [ ] Database helper implemented
- [ ] Aggregation saves to DB instead of JSON
- [ ] DAG tracks video status
- [ ] Connection test passes

---

## Day 2: Testing & Drift Monitoring (4-6 hours)

### Step 1: End-to-End Test (1.5 hours)

Test the full pipeline with a real video:

```bash
# 1. Make sure Airflow is running
airflow webserver -p 8080 &
airflow scheduler &

# 2. Make sure Supabase credentials are in .env
cat .env | grep SUPABASE

# 3. Trigger DAG via API
curl -X POST http://localhost:8080/api/v1/dags/pitchquest_video_analysis/dagRuns \
  -H "Content-Type: application/json" \
  -u admin:admin \
  -d '{
    "conf": {
      "video_id": "demo_video_1",
      "video_path": "/path/to/airflow/data/input/demo_video_1.mp4"
    }
  }'

# 4. Monitor in Airflow UI
open http://localhost:8080

# 5. Check database for results
# Go to Supabase dashboard and check analysis_results table
```

### Step 2: Add Drift Monitoring Task (2 hours)

**File: `airflow/pipeline/drift_monitoring.py`**

```python
"""
Drift monitoring using Evidently AI
"""

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from datetime import datetime
from airflow.utils.logger import logger
from airflow.utils.db_helper import get_recent_scores, save_monitoring_metric

def check_drift(**context):
    """
    Check for drift in pitch analysis scores
    """
    try:
        logger.info("🔍 Starting drift detection...")
        
        # Get recent scores
        recent_data = get_recent_scores(n=100)
        
        if len(recent_data) < 20:
            logger.warning("⚠️ Not enough data for drift detection (need at least 20 analyses)")
            return {"drift_detected": False, "reason": "insufficient_data"}
        
        # Split into reference (older) and current (recent) data
        split_point = len(recent_data) // 2
        reference_data = recent_data[split_point:]  # Older half
        current_data = recent_data[:split_point]     # Recent half
        
        # Convert to DataFrames
        ref_df = pd.DataFrame([{"score": item["overall_score"]} for item in reference_data])
        curr_df = pd.DataFrame([{"score": item["overall_score"]} for item in current_data])
        
        # Calculate statistics
        ref_mean = ref_df["score"].mean()
        curr_mean = curr_df["score"].mean()
        ref_std = ref_df["score"].std()
        curr_std = curr_df["score"].std()
        
        # Simple drift detection: mean difference > 15 points or std change > 50%
        mean_diff = abs(curr_mean - ref_mean)
        std_change = abs(curr_std - ref_std) / ref_std if ref_std > 0 else 0
        
        drift_detected = mean_diff > 15 or std_change > 0.5
        
        drift_metrics = {
            "drift_detected": drift_detected,
            "reference_mean": float(ref_mean),
            "current_mean": float(curr_mean),
            "mean_difference": float(mean_diff),
            "reference_std": float(ref_std),
            "current_std": float(curr_std),
            "std_change_pct": float(std_change * 100),
            "sample_sizes": {
                "reference": len(reference_data),
                "current": len(current_data)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save to monitoring metrics
        save_monitoring_metric("drift", drift_metrics)
        
        if drift_detected:
            logger.warning(f"⚠️ DRIFT DETECTED!")
            logger.warning(f"   Mean difference: {mean_diff:.2f} points")
            logger.warning(f"   Reference mean: {ref_mean:.2f}, Current mean: {curr_mean:.2f}")
            logger.warning(f"   Std change: {std_change * 100:.1f}%")
            
            # TODO: Send alert (Slack, email, etc.)
            # send_drift_alert(drift_metrics)
        else:
            logger.info(f"✅ No drift detected")
            logger.info(f"   Reference mean: {ref_mean:.2f}, Current mean: {curr_mean:.2f}")
            logger.info(f"   Mean difference: {mean_diff:.2f} points (threshold: 15)")
        
        return drift_metrics
    
    except Exception as e:
        logger.error(f"❌ Drift detection failed: {e}", exc_info=True)
        return {"drift_detected": False, "error": str(e)}

def send_drift_alert(metrics: dict):
    """
    Send alert when drift is detected
    TODO: Implement Slack/email integration
    """
    logger.info("📧 Sending drift alert...")
    # Example: Post to Slack webhook
    # requests.post(SLACK_WEBHOOK_URL, json={"text": f"Drift detected! {metrics}"})
```

Add to DAG:

```python
from airflow.pipeline.drift_monitoring import check_drift

drift_check = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    provide_context=True,
)

# Add to end of pipeline
aggregate >> drift_check >> cleanup
```

### Step 3: Add Cleanup Task (1 hour)

**File: `airflow/pipeline/cleanup.py`**

```python
"""
Cleanup temporary files after processing
"""

import os
import shutil
from pathlib import Path
from airflow.utils.logger import logger

def cleanup_temp_files(**context):
    """
    Clean up temporary files (frames, audio) after analysis
    """
    ti = context['ti']
    
    # Get paths from previous tasks
    preprocess_data = ti.xcom_pull(task_ids='preprocess')
    
    if not preprocess_data:
        logger.warning("⚠️ No preprocess data found, skipping cleanup")
        return
    
    frames_dir = preprocess_data.get('frames_dir')
    audio_path = preprocess_data.get('audio_path')
    
    cleaned = []
    errors = []
    
    # Clean up frames directory
    if frames_dir and os.path.exists(frames_dir):
        try:
            shutil.rmtree(frames_dir)
            cleaned.append(f"frames: {frames_dir}")
            logger.info(f"🗑️ Cleaned up frames directory: {frames_dir}")
        except Exception as e:
            errors.append(f"frames: {e}")
            logger.error(f"❌ Failed to clean up frames: {e}")
    
    # Clean up audio file
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            cleaned.append(f"audio: {audio_path}")
            logger.info(f"🗑️ Cleaned up audio file: {audio_path}")
        except Exception as e:
            errors.append(f"audio: {e}")
            logger.error(f"❌ Failed to clean up audio: {e}")
    
    # Summary
    logger.info(f"✅ Cleanup complete: {len(cleaned)} items removed, {len(errors)} errors")
    
    return {
        "cleaned": cleaned,
        "errors": errors,
        "success": len(errors) == 0
    }
```

Add to DAG:

```python
from airflow.pipeline.cleanup import cleanup_temp_files

cleanup = PythonOperator(
    task_id='cleanup',
    python_callable=cleanup_temp_files,
    provide_context=True,
    trigger_rule='all_done',  # Run even if upstream fails
)
```

### Step 4: Integration Testing (1.5 hours)

Test all edge cases:

1. **Normal flow:**
   ```bash
   # Upload video via backend API
   curl -X POST http://localhost:8000/api/upload -F "file=@test.mp4"
   # Trigger analysis
   curl -X POST http://localhost:8000/api/analyze/VIDEO_ID
   # Wait and check results
   curl http://localhost:8000/api/results/VIDEO_ID
   ```

2. **Failure scenarios:**
   - Corrupt video file
   - Missing video file
   - Database connection lost
   - API key invalid

3. **Concurrent uploads:**
   ```bash
   # Upload 3 videos simultaneously
   for i in {1..3}; do
     curl -X POST http://localhost:8000/api/upload \
       -F "file=@video$i.mp4" &
   done
   wait
   ```

### End of Day 2 Deliverable
- [ ] Full end-to-end test passes
- [ ] Results appear in Supabase
- [ ] Drift monitoring works
- [ ] Cleanup removes temp files
- [ ] All edge cases handled

---

## Troubleshooting Guide

### Issue: "Supabase connection failed"

**Check:**
```bash
# Verify environment variables
echo $SUPABASE_URL
echo $SUPABASE_KEY

# Test connection
python -c "
from supabase import create_client
client = create_client('YOUR_URL', 'YOUR_KEY')
print(client.table('videos').select('*').limit(1).execute())
"
```

**Fix:**
- Check `.env` file has correct credentials
- Verify Supabase project is active
- Check if using `anon` key (not `service_role` key)

---

### Issue: "Database save fails but task succeeds"

**Problem:** Exception not being raised

**Fix:** In `save_results_to_db`, ensure you `raise` after logging error:

```python
except Exception as e:
    logger.error(f"Failed: {e}")
    raise  # CRITICAL: Must raise to fail the task
```

---

### Issue: "Status not updating in database"

**Check Airflow logs:**
```bash
tail -f logs/dag_id/pitchquest_video_analysis/task_id/*/1.log
```

**Common causes:**
- Video doesn't exist in database (backend didn't create it)
- `video_id` mismatch between backend and Airflow
- Database permissions

---

### Issue: "Drift detection always returns insufficient data"

**Check:**
```python
# In Python shell
from airflow.utils.db_helper import get_recent_scores
scores = get_recent_scores(100)
print(f"Found {len(scores)} scores")
```

**If empty:** No analyses in database yet, run a few videos first

---

## Success Criteria

- [ ] Airflow saves results to Supabase (not JSON)
- [ ] Video status updates throughout pipeline
- [ ] Drift detection runs after each analysis
- [ ] Cleanup removes temporary files
- [ ] End-to-end test completes successfully
- [ ] Backend can fetch results from database
- [ ] All error cases handled gracefully

**Database integration complete! 🎉**