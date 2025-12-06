import pytest
from utils.config import config, load_prompt

def test_config_has_api_keys():
    """Ensure API keys are loaded"""
    # This will fail if .env not set up - expected
    assert config.OPENAI_API_KEY is not None or config.GEMINI_API_KEY is not None

def test_config_paths_exist():
    """Ensure directory paths are valid"""
    assert config.BASE_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.PROMPTS_DIR.exists()

def test_load_prompt_function():
    """Test prompt loading utility"""
    # This will pass once prompts are created
    try:
        prompt = load_prompt('visual_analysis')
        assert len(prompt) > 0
    except FileNotFoundError:
        pytest.skip("Prompts not yet created")