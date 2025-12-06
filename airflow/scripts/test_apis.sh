#!/bin/bash

echo "🔑 Testing API keys..."

# Test OpenAI
python3 - <<EOF
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    client.models.list()
    print('✅ OpenAI API key valid')
except Exception as e:
    print(f'❌ OpenAI API key invalid: {e}')
    exit(1)
EOF

# Test Gemini
python3 - <<EOF
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

try:
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    models = list(genai.list_models())
    print('✅ Gemini API key valid')
except Exception as e:
    print(f'❌ Gemini API key invalid: {e}')
    exit(1)
EOF

echo ""
echo "✅ All API keys validated"