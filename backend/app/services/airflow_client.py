import requests
from requests.auth import HTTPBasicAuth
from app.config import settings

class AirflowClient:
    def __init__(self):
        self.base_url = settings.AIRFLOW_URL
        self.dag_id = "pitchquest_video_analysis"
        self.auth = HTTPBasicAuth(settings.AIRFLOW_USERNAME, settings.AIRFLOW_PASSWORD)  # YOU'RE MISSING THIS!
    
    def trigger_dag(self, video_id: str, video_path: str) -> dict:
        url = f"{self.base_url}/api/v1/dags/{self.dag_id}/dagRuns"
        payload = {"conf": {"video_id": video_id, "video_path": video_path}}
        
        response = requests.post(
            url, 
            json=payload, 
            auth=self.auth,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return {"job_id": response.json()["dag_run_id"], "status": "triggered"}
        else:
            raise Exception(f"Airflow failed: {response.text}")

    def get_dag_run_status(self, job_id: str) -> dict:
        """Get DAG run status from Airflow"""
        url = f"{self.base_url}/api/v1/dags/{self.dag_id}/dagRuns/{job_id}"
        
        response = requests.get(url, auth=self.auth, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            state = data["state"]  # "running", "success", "failed"
            
            status_map = {
                "running": "processing",
                "success": "completed",
                "failed": "failed",
                "queued": "processing"
            }
            
            return {
                "status": status_map.get(state, "processing"),
                "airflow_state": state
            }
        else:
            raise Exception(f"Failed to get status: {response.text}")
    def get_task_instances(self, job_id: str) -> list:
        """Get individual task statuses"""
        url = f"{self.base_url}/api/v1/dags/{self.dag_id}/dagRuns/{job_id}/taskInstances"
        
        response = requests.get(url, auth=self.auth, timeout=10)
        
        if response.status_code == 200:
            return response.json().get("task_instances", [])
        else:
            return []

airflow_client = AirflowClient()