"""
Video processing functions for Airflow DAG
"""
import pickle
from pathlib import Path
import imageio.v3 as iio

BASE_DIR = Path("/opt/airflow/dags")
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
AUDIO_DIR = OUTPUT_DIR / "audio"
FRAMES_DIR = OUTPUT_DIR / "frames"

def load_video(**kwargs):
    print("=" * 50)
    print("TASK 1: Loading Video")
    print("=" * 50)
    
    video_files = list(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in {VIDEO_DIR}")
    
    video_path = video_files[0]
    print(f"Found video: {video_path.name}")
    
    video_info = {'path': str(video_path), 'name': video_path.name}
    print(f"✅ Video loaded: {video_path.name}")
    return pickle.dumps(video_info)

def extract_audio(ti, **kwargs):
    print("=" * 50)
    print("TASK 2: Extracting Audio")
    print("=" * 50)
    
    video_info = pickle.loads(ti.xcom_pull(task_ids='load_video_task'))
    video_name = Path(video_info['name']).stem
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    audio_info = {'audio_path': None, 'video_name': video_name}
    print("⚠️  Audio extraction skipped")
    return pickle.dumps(audio_info)

def extract_frames(ti, **kwargs):
    print("=" * 50)
    print("TASK 3: Extracting Frames")
    print("=" * 50)
    
    video_info = pickle.loads(ti.xcom_pull(task_ids='load_video_task'))
    video_path = video_info['path']
    video_name = Path(video_info['name']).stem
    
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx % 30 == 0:
            frame_path = FRAMES_DIR / f"{video_name}_frame_{frame_count:04d}.jpg"
            iio.imwrite(frame_path, frame)
            frame_paths.append(str(frame_path))
            frame_count += 1
            print(f"  Frame {frame_count}")
            if frame_count >= 10:
                break
    
    print(f"✅ Extracted {frame_count} frames")
    return pickle.dumps({'frame_paths': frame_paths, 'frame_count': frame_count, 'video_name': video_name})

def save_results(ti, **kwargs):
    print("=" * 50)
    print("TASK 4: Saving Results")
    print("=" * 50)
    
    video_info = pickle.loads(ti.xcom_pull(task_ids='load_video_task'))
    frames_info = pickle.loads(ti.xcom_pull(task_ids='extract_frames_task'))
    
    print(f"\n📹 Video: {video_info['name']}")
    print(f"🖼️  Frames: {frames_info['frame_count']}")
    
    summary_path = OUTPUT_DIR / "processing_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Video: {video_info['name']}\nFrames: {frames_info['frame_count']}\n")
    
    print("✅ Complete!")
    return "Complete!"