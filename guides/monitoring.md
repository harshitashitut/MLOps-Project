# Monitoring Setup - Compact Guide

**Owner:** Person 3 | **Timeline:** Day 2 (2-3 hours)

---

## Goal
Track system health, drift, and performance with minimal overhead.

---

## Components

1. **Evidently AI** - Drift detection
2. **Structured Logging** - JSON logs
3. **Frontend Dashboard** - `/report` page (already built)
4. **Backend Monitoring Endpoint** - `/api/monitoring` (already built)

---

## Step 1: Evidently AI Integration (45 min)

**Install:**
```bash
cd airflow
pip install evidently==0.4.11
```

**File: `airflow/pipeline/drift_monitoring.py`**

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from datetime import datetime
from utils.logger import logger
from utils.db_helper import get_recent_scores, save_monitoring_metric

def check_drift(**context):
    """Detect drift in analysis scores"""
    recent = get_recent_scores(n=100)
    
    if len(recent) < 20:
        logger.warning("Not enough data for drift detection")
        return {"drift_detected": False, "reason": "insufficient_data"}
    
    # Split: older 50% vs recent 50%
    split = len(recent) // 2
    ref_data = recent[split:]
    curr_data = recent[:split]
    
    ref_df = pd.DataFrame([{"score": r["overall_score"]} for r in ref_data])
    curr_df = pd.DataFrame([{"score": r["overall_score"]} for r in curr_data])
    
    # Detect drift
    ref_mean, curr_mean = ref_df["score"].mean(), curr_df["score"].mean()
    mean_diff = abs(curr_mean - ref_mean)
    drift = mean_diff > 15
    
    metrics = {
        "drift_detected": drift,
        "mean_difference": float(mean_diff),
        "reference_mean": float(ref_mean),
        "current_mean": float(curr_mean),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    save_monitoring_metric("drift", metrics)
    
    if drift:
        logger.warning(f"⚠️ DRIFT DETECTED: {mean_diff:.1f} point difference")
    
    return metrics
```

**Add to DAG:**
```python
from pipeline.drift_monitoring import check_drift

drift_task = PythonOperator(
    task_id='drift_check',
    python_callable=check_drift
)

aggregate >> drift_task
```

---

## Step 2: Structured Logging (30 min)

**File: `airflow/utils/logger.py`**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger
```

**Usage:**
```python
from utils.logger import setup_logger
logger = setup_logger(__name__)

logger.info("Processing video", extra={"video_id": "123", "duration": 120})
```

---

## Step 3: Cost Tracking (30 min)

**File: `airflow/utils/cost_tracker.py`**

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class APIUsage:
    model: str
    tokens: int
    cost_per_1k: float
    
    @property
    def cost(self) -> float:
        return (self.tokens / 1000) * self.cost_per_1k

class CostTracker:
    # Pricing (as of Dec 2025)
    COSTS = {
        "whisper-1": 0.006,  # per minute
        "gemini-1.5-flash": 0.075,  # per 1M tokens (input)
        "gemini-1.5-flash-output": 0.30,  # per 1M tokens (output)
    }
    
    def __init__(self):
        self.usage = []
    
    def track_whisper(self, duration_seconds: float):
        minutes = duration_seconds / 60
        cost = minutes * self.COSTS["whisper-1"]
        self.usage.append({"model": "whisper-1", "cost": cost})
        return cost
    
    def track_gemini(self, input_tokens: int, output_tokens: int):
        input_cost = (input_tokens / 1_000_000) * self.COSTS["gemini-1.5-flash"]
        output_cost = (output_tokens / 1_000_000) * self.COSTS["gemini-1.5-flash-output"]
        total = input_cost + output_cost
        self.usage.append({"model": "gemini-1.5-flash", "cost": total})
        return total
    
    def total_cost(self) -> float:
        return sum(u["cost"] for u in self.usage)
    
    def summary(self) -> Dict:
        return {
            "total_cost_usd": round(self.total_cost(), 4),
            "breakdown": self.usage
        }

# Usage in pipeline
tracker = CostTracker()
tracker.track_whisper(video_duration)
tracker.track_gemini(input_tokens=1000, output_tokens=500)
print(tracker.summary())
```

**Integrate in aggregation.py:**
```python
metadata["costs"] = cost_tracker.summary()
```

---

## Step 4: Health Checks (15 min)

**File: `airflow/utils/health.py`**

```python
import requests
from utils.config import config

def check_services() -> dict:
    """Check health of external services"""
    health = {}
    
    # Supabase
    try:
        from utils.db_helper import test_db_connection
        health["database"] = "healthy" if test_db_connection() else "unhealthy"
    except:
        health["database"] = "unhealthy"
    
    # OpenAI
    try:
        requests.get("https://api.openai.com/v1/models", timeout=5)
        health["openai"] = "healthy"
    except:
        health["openai"] = "unhealthy"
    
    # Gemini (check if key is set)
    health["gemini"] = "healthy" if config.GEMINI_API_KEY else "unhealthy"
    
    return health
```

---

## Step 5: Monitoring Dashboard Data (30 min)

**Backend already has `/api/monitoring` endpoint. Enhance it:**

**File: `backend/app/database/crud.py`**

Add:
```python
async def get_system_metrics() -> Dict:
    """Enhanced monitoring metrics"""
    
    # Basic stats
    total = supabase.table("videos").select("*", count="exact").execute().count
    
    # Scores
    results = supabase.table("analysis_results").select("overall_score, processing_time_seconds, created_at").execute()
    scores = [r["overall_score"] for r in results.data]
    times = [r["processing_time_seconds"] for r in results.data]
    
    # Recent drift
    drift_metrics = supabase.table("monitoring_metrics")\
        .select("*")\
        .eq("metric_type", "drift")\
        .order("timestamp", desc=True)\
        .limit(1)\
        .execute()
    
    drift_detected = False
    if drift_metrics.data:
        latest_drift = json.loads(drift_metrics.data[0]["metric_value"])
        drift_detected = latest_drift.get("drift_detected", False)
    
    return {
        "total_analyses": total,
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "avg_processing_time": sum(times) / len(times) if times else 0,
        "drift_detected": drift_detected,
        "system_health": calculate_health(scores),
        "score_history": [
            {"score": r["overall_score"], "date": r["created_at"]}
            for r in results.data[-30:]  # Last 30
        ]
    }

def calculate_health(scores: list) -> str:
    """System health based on scores"""
    if not scores or len(scores) < 5:
        return "healthy"
    
    avg = sum(scores[-10:]) / len(scores[-10:])  # Last 10
    if avg < 30:
        return "critical"
    elif avg < 50:
        return "warning"
    return "healthy"
```

---

## Step 6: Alerting (Optional - 30 min)

**Slack Webhook Integration:**

```python
# airflow/utils/alerts.py
import requests
from utils.config import config

SLACK_WEBHOOK = config.SLACK_WEBHOOK_URL  # Add to .env

def send_alert(message: str, level: str = "warning"):
    """Send alert to Slack"""
    emoji = "⚠️" if level == "warning" else "🚨"
    
    payload = {
        "text": f"{emoji} PitchQuest Alert",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{level.upper()}*\n{message}"
                }
            }
        ]
    }
    
    try:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Usage
if drift_detected:
    send_alert(f"Drift detected: {mean_diff:.1f} point change", level="warning")
```

---

## Step 7: Logs Aggregation (15 min)

**View logs across services:**

```bash
# All logs
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Search logs
docker-compose logs | grep ERROR

# Last 100 lines
docker-compose logs --tail=100
```

**Save logs to file:**

Add to `docker-compose.yml`:
```yaml
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Step 8: Performance Metrics (15 min)

**Track API response times:**

**File: `backend/app/middleware/metrics.py`**

```python
from fastapi import Request
import time
from app.utils.logger import logger

async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    logger.info(
        "api_request",
        extra={
            "path": request.url.path,
            "method": request.method,
            "duration_ms": round(duration * 1000, 2),
            "status_code": response.status_code
        }
    )
    
    return response
```

**Add to `main.py`:**
```python
from app.middleware.metrics import metrics_middleware
app.middleware("http")(metrics_middleware)
```

---

## Quick Commands

```bash
# Check drift
docker-compose exec airflow-scheduler python -c "
from pipeline.drift_monitoring import check_drift
result = check_drift()
print(result)
"

# View monitoring metrics
curl http://localhost:8000/api/monitoring | jq

# Check service health
docker-compose ps

# Resource usage
docker stats

# Database query
docker-compose exec backend python -c "
from app.database.crud import get_system_metrics
import asyncio
print(asyncio.run(get_system_metrics()))
"
```

---

## Monitoring Checklist

- [ ] Evidently AI integrated in DAG
- [ ] Drift detection runs after each analysis
- [ ] JSON logging implemented
- [ ] Cost tracking added to metadata
- [ ] `/api/monitoring` returns accurate data
- [ ] Frontend dashboard shows metrics
- [ ] Health checks working
- [ ] Logs accessible via `docker-compose logs`
- [ ] (Optional) Slack alerts configured

---

## What to Monitor

| Metric | Threshold | Action |
|--------|-----------|--------|
| Drift detected | True | Review model, check data quality |
| Avg score | < 30 | Investigate low scores |
| Processing time | > 3 min | Optimize pipeline |
| Error rate | > 10% | Check logs, fix issues |
| API latency | > 1 sec | Optimize backend queries |
| Disk usage | > 80% | Clean up temp files |

---

## Success Criteria

- [ ] Can detect drift
- [ ] Dashboard shows real-time metrics
- [ ] All logs structured (JSON)
- [ ] Cost tracking accurate
- [ ] No silent failures
- [ ] Easy to debug issues

**Monitoring complete! 📊**