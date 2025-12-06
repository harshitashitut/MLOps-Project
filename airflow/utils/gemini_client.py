import requests
import json
import base64
import logging
from typing import List, Dict, Any
from utils.config import config
from utils.error_handlers import PipelineError

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
    def generate_content(self, prompt: str, image_paths: List[str] = None) -> Dict:
        """
        Send a request to Gemini via REST API (Bypassing SDK)
        """
        model = config.GEMINI_MODEL
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        
        # 1. Build Payload
        parts = [{"text": prompt}]
        
        # 2. Add Images (Base64)
        if image_paths:
            for path in image_paths:
                try:
                    with open(path, "rb") as img_file:
                        b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg", 
                            "data": b64_data
                        }
                    })
                except Exception as e:
                    logger.warning(f"Skipping image {path}: {e}")

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json"  # Force JSON mode
            }
        }

        # 3. Request
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # 4. Parse (Claude's robust extraction logic)
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return self._clean_json(text)
            except (KeyError, IndexError):
                logger.error(f"Unexpected response structure: {result}")
                raise PipelineError("Gemini API returned unparsable response")

        except Exception as e:
            logger.error(f"Gemini REST call failed: {e}")
            raise PipelineError(f"Gemini REST call failed: {e}")

    def _clean_json(self, text: str) -> Dict:
        """Sanitize JSON string"""
        text = text.strip()
        # Remove Markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("\n", 1)[0]
        # Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            text = text[start:end]
        return json.loads(text)

gemini_client = GeminiClient()
