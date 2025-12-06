import pytest
from airflow.models import DagBag

def test_dag_loads_without_errors():
    """Ensure DAG file has no syntax errors"""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"

def test_dag_exists():
    """Validate pitch_video_analysis DAG exists"""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert 'pitch_video_analysis' in dagbag.dags, "DAG not found"

def test_dag_has_required_tasks():
    """Validate DAG has minimum required tasks"""
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('pitch_video_analysis')
    
    assert dag is not None
    assert len(dag.tasks) >= 8, f"Expected at least 8 tasks, found {len(dag.tasks)}"