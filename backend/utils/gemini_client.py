import requests
import json
import base64
import logging
import time
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
        Send a request to Gemini via REST API with retry logic for rate limits
        """
        import time
        
        max_retries = 3
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
                "temperature": 0.2,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json"
            }
        }

        # 3. Request with retry logic
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                # Success - parse and return
                # 4. Parse response (handle different response structures)
                try:
                    candidate = result['candidates'][0]
                    content = candidate.get('content', {})
                    
                    # Check if there are parts with text
                    parts = content.get('parts', [])
                    if parts and 'text' in parts[0]:
                        text = parts[0]['text']
                        return self._clean_json(text)
                    
                    # If no parts, check if response was blocked or truncated
                    finish_reason = candidate.get('finishReason', '')
                    if finish_reason == 'MAX_TOKENS':
                        logger.warning("Response truncated due to MAX_TOKENS.")
                        raise PipelineError("Response truncated - try increasing maxOutputTokens in config")
                    
                    logger.error(f"No text in response. Finish reason: {finish_reason}")
                    logger.error(f"Response structure: {result}")
                    raise PipelineError(f"Gemini returned no text content (finish reason: {finish_reason})")
                    
                except (KeyError, IndexError) as e:
                    logger.error(f"Unexpected response structure: {result}")
                    raise PipelineError(f"Failed to parse Gemini response: {e}")
            
            except requests.exceptions.HTTPError as e:
                # Handle rate limiting with exponential backoff
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 30  # 30, 60, 90 seconds
                    logger.warning(f"Rate limit hit (429). Waiting {wait_time}s before retry {attempt+2}/{max_retries}...")
                    time.sleep(wait_time)
                    continue  # Retry
                else:
                    # Not a rate limit or max retries reached
                    logger.error(f"Gemini REST call failed: {e}")
                    raise PipelineError(f"Gemini REST call failed: {e}")
            
            except requests.exceptions.RequestException as e:
                # Network errors, timeouts, etc.
                logger.error(f"Request failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 10  # Brief wait for network issues
                    logger.warning(f"Retrying in {wait_time}s... ({attempt+2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise PipelineError(f"Gemini REST call failed after {max_retries} attempts: {e}")
            
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error: {e}")
                raise PipelineError(f"Gemini REST call failed: {e}")
        
        # Should never reach here, but just in case
        raise PipelineError("Failed to get response from Gemini after all retries")
    def _clean_json(self, text: str) -> Dict:
        """Sanitize JSON string with robust error handling"""
        import re
        
        text = text.strip()
        
        # Remove Markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # Find the outermost JSON object
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            text = text[start:end]
        
        # Try to parse directly
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed at position {e.pos}: {e.msg}")
            
            # Common fixes
            # Fix 1: Remove trailing commas before ] or }
            text = re.sub(r',(\s*[}\]])', r'\1', text)
            
            # Fix 2: Add missing commas between objects/arrays
            text = re.sub(r'}\s*{', '},{', text)
            text = re.sub(r']\s*\[', '],[', text)
            
            # Fix 3: Remove incomplete last item
            # If there's an error near the end, try truncating
            if e.pos > len(text) * 0.8:  # Error in last 20%
                logger.warning("Error near end of JSON, attempting truncation...")
                # Find last complete item
                last_complete = text[:e.pos].rfind('},')
                if last_complete > 0:
                    text = text[:last_complete+1] + ']}'
            
            # Try parsing again
            try:
                return json.loads(text)
            except json.JSONDecodeError as e2:
                logger.error(f"JSON repair failed: {e2}")
                logger.error(f"Problematic area: ...{text[max(0,e2.pos-100):e2.pos+100]}...")
                raise PipelineError(f"Failed to parse JSON from Gemini: {e2}")

gemini_client = GeminiClient()
