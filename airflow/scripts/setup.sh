#!/bin/bash
set -e

echo "🚀 Setting up PitchQuest Airflow Pipeline..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required. Install from https://docker.com"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 required"; exit 1; }

# Create .env from template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  IMPORTANT: Edit .env and add your API keys!"
    echo "   - OPENAI_API_KEY"
    echo "   - GEMINI_API_KEY"
    echo ""
fi

# Create data directories
mkdir -p data/{input,temp,output}

# Set Airflow UID
export AIRFLOW_UID=$(id -u)

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: cd docker && docker-compose up airflow-init"
echo "3. Run: docker-compose up -d"
echo "4. Visit: http://localhost:8080 (user: airflow, pass: airflow)"