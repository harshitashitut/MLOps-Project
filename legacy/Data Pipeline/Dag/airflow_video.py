"""
Airflow DAG for Video Processing Pipeline
"""
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.video_processing import load_video, extract_audio, extract_frames, save_results
from airflow import configuration as conf

conf.set('core', 'enable_xcom_pickling', 'True')

default_args = {
    'owner': 'airflow_video_pipeline',
    'start_date': datetime(2024, 10, 19),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'Video_Processing_Pipeline',
    default_args=default_args,
    description='Automated video breakdown',
    schedule_interval=None,
    catchup=False,
    tags=['video', 'mlops']
)

load_video_task = PythonOperator(
    task_id='load_video_task',
    python_callable=load_video,
    provide_context=True,
    dag=dag,
)

extract_audio_task = PythonOperator(
    task_id='extract_audio_task',
    python_callable=extract_audio,
    provide_context=True,
    dag=dag,
)

extract_frames_task = PythonOperator(
    task_id='extract_frames_task',
    python_callable=extract_frames,
    provide_context=True,
    dag=dag,
)

save_results_task = PythonOperator(
    task_id='save_results_task',
    python_callable=save_results,
    provide_context=True,
    dag=dag,
)

load_video_task >> [extract_audio_task, extract_frames_task] >> save_results_task