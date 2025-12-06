#!/bin/bash

echo "🔍 Validating environment..."

# Check RAM
TOTAL_RAM=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || sysctl -n hw.memsize | awk '{print int($1/1073741824)}')
echo "Total RAM: ${TOTAL_RAM}GB"

if [ "$TOTAL_RAM" -lt 4 ]; then
    echo "⚠️  Warning: Less than 4GB RAM. Pipeline may be slow."
fi

# Check Docker
if docker --version >/dev/null 2>&1; then
    echo "✅ Docker installed: $(docker --version)"
else
    echo "❌ Docker not installed"
    exit 1
fi

# Check disk space
FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo "✅ Free disk space: $FREE_SPACE"

# Check .env file
if [ -f .env ]; then
    if grep -q "your-key-here" .env; then
        echo "⚠️  Warning: .env contains placeholder values"
    else
        echo "✅ .env file configured"
    fi
else
    echo "❌ .env file missing. Run setup.sh first"
    exit 1
fi

echo ""
echo "✅ Environment validation complete"