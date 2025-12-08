import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
# OpenAI API (Whisper)
    OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

# Google Gemini API
    GEMINI_API_KEY=os.getenv('GEMINI_API_KEY')
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    INPUT_DIR = DATA_DIR / 'input'
    TEMP_DIR = DATA_DIR / 'temp'
    OUTPUT_DIR = DATA_DIR / 'output'
    PROMPTS_DIR = BASE_DIR / 'prompts'
    
    # Processing Settings
    FRAME_SAMPLE_RATE = 2  # Extract 1 frame per 2 seconds
    MAX_FRAMES_FOR_VISION = 10  # Limit frames sent to Gemini
    
    # Model Settings
    WHISPER_MODEL = 'whisper-1'
    GEMINI_MODEL = 'gemini-2.5-flash'
    AGGREGATION_MODEL = 'gemini-2.5-flash'
    WAV2VEC_MODEL = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'

def load_prompt(prompt_name: str) -> str:
    """Load prompt from prompts/ directory"""
    prompt_path = Config.PROMPTS_DIR / f'{prompt_name}.txt'
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r') as f:
        return f.read().strip()

config = Config()


# # OpenAI API (Whisper)
# OPENAI_API_KEY="sfuhKT2w5J5_tB66_ODjVXFhwKwXYHrlUC9KauWspTcAEW1JB2Gxc8cXEAG3LQA"

# # Google Gemini API
# GEMINI_API_KEY="AIWWeriGj41WuL68"

# Airflow
# AIRFLOW_UID=50000
# AIRFLOW_HOME=/opt/airflow

# # Optional (for future)
# GCS_BUCKET_NAME=pitchquest-videos
# DATABASE_URL=postgresql://user:pass@localhost/pitchquest
# ```

# ### `.gitignore`
# ```
# # Airflow
# logs/
# airflow.db
# airflow.cfg
# unittests.cfg

# # Python
# __pycache__/
# *.pyc
# *.pyo
# .pytest_cache/
# *.egg-info/

# # Environment
# .env
# *.env

# # Data
# data/input/*
# !data/input/.gitkeep
# data/temp/*
# !data/temp/.gitkeep
# data/output/*
# !data/output/.gitkeep

# # IDE
# .vscode/
# .idea/
# *.swp

# # Docker
# *.log