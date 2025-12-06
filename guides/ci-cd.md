# CI/CD Pipeline - Compact Guide

**Owner:** Person 3 | **Timeline:** Day 2 (2-3 hours)

---

## Goal
Automated testing and deployment on every push to `main`.

---

## Step 1: GitHub Actions Workflow (45 min)

**File: `.github/workflows/main.yml`**

```yaml
name: PitchQuest CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # ==================== LINTING ====================
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Lint Backend
        run: |
          pip install black flake8
          black --check backend/
          flake8 backend/ --max-line-length=100 --exclude=venv
      
      - name: Lint Airflow
        run: |
          black --check airflow/pipeline/ airflow/utils/
          flake8 airflow/ --max-line-length=100 --exclude=venv,logs

  # ==================== BACKEND TESTS ====================
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt pytest pytest-cov
      
      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend/coverage.xml
          flags: backend

  # ==================== AIRFLOW TESTS ====================
  test-airflow:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install Airflow
        run: |
          cd airflow
          pip install -r requirements.txt pytest
      
      - name: Validate DAG
        run: |
          cd airflow
          python dags/pitch_analysis_dag.py
      
      - name: Run pipeline tests
        run: |
          cd airflow
          pytest tests/ -v

  # ==================== FRONTEND TESTS ====================
  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: |
          cd frontend
          npm ci
      
      - name: Run linter
        run: |
          cd frontend
          npm run lint
      
      - name: Build
        run: |
          cd frontend
          npm run build

  # ==================== DOCKER BUILD ====================
  build-images:
    needs: [lint, test-backend, test-airflow, test-frontend]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Build backend image
        run: |
          cd backend
          docker build -t pitchquest-backend:latest .
      
      - name: Build frontend image
        run: |
          cd frontend
          docker build -t pitchquest-frontend:latest .
      
      - name: Build airflow image
        run: |
          cd airflow
          docker build -t pitchquest-airflow:latest .
      
      - name: Test images
        run: |
          docker-compose -f docker-compose.yml config

  # ==================== DEPLOY ====================
  deploy:
    needs: [build-images]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy notification
        run: |
          echo "🚀 Deployment would happen here"
          # Add actual deployment commands:
          # - SSH to server
          # - Pull latest images
          # - Run docker-compose up -d
```

---

## Step 2: Pre-commit Hooks (30 min)

**File: `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.10
        exclude: ^(venv|.venv|env)/

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
        exclude: ^(venv|.venv|env|logs)/

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: [--maxkb=10000]
```

**Install:**
```bash
pip install pre-commit
pre-commit install
```

---

## Step 3: Backend Tests (30 min)

**File: `backend/tests/test_api.py`**

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_endpoint_exists():
    # Test endpoint exists (will fail without file)
    response = client.post("/api/upload")
    assert response.status_code in [400, 422]  # Bad request, not 404

def test_cors_headers():
    response = client.get("/health")
    assert "access-control-allow-origin" in response.headers
```

Run: `pytest backend/tests/ -v`

---

## Step 4: Airflow DAG Validation Test (15 min)

**File: `airflow/tests/test_dag_validation.py`**

```python
import pytest
from airflow.models import DagBag

def test_dag_loads():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"

def test_dag_structure():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.dags.get('pitchquest_video_analysis')
    
    assert dag is not None
    assert len(dag.tasks) >= 5  # Has multiple tasks
    assert dag.default_args['retries'] == 2

def test_no_cycles():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    for dag_id, dag in dagbag.dags.items():
        assert not dag.test_cycle()
```

---

## Step 5: Deployment Script (30 min)

**File: `scripts/deploy.sh`**

```bash
#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# Pull latest code
git pull origin main

# Build images
docker-compose build --no-cache

# Stop old containers
docker-compose down

# Start new containers
docker-compose up -d

# Wait for services
sleep 10

# Health checks
curl -f http://localhost:8000/health || exit 1
curl -f http://localhost:3000 || exit 1
curl -f http://localhost:8080/health || exit 1

echo "✅ Deployment complete!"

# Show logs
docker-compose logs --tail=50
```

**Make executable:**
```bash
chmod +x scripts/deploy.sh
```

---

## Step 6: Secrets Management (15 min)

**GitHub Secrets:**

Go to: `Repo → Settings → Secrets → Actions`

Add:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DATABASE_URL`

**Use in workflow:**

```yaml
test-backend:
  env:
    SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
    SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
```

---

## Step 7: Status Badges (15 min)

**Add to `README.md`:**

```markdown
![CI/CD](https://github.com/USERNAME/REPO/workflows/PitchQuest%20CI%2FCD/badge.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Coverage](https://img.shields.io/codecov/c/github/USERNAME/REPO)
```

---

## Testing the Pipeline

**Local test:**
```bash
# Run what CI runs
black --check backend/ airflow/
flake8 backend/ airflow/ --max-line-length=100
pytest backend/tests/ -v
pytest airflow/tests/ -v
docker-compose build
```

**Trigger CI:**
```bash
git add .
git commit -m "test: trigger CI/CD"
git push origin main
```

**Monitor:** Go to GitHub → Actions tab

---

## Common Issues

| Issue | Fix |
|-------|-----|
| Tests fail locally but pass in CI | Check Python version matches (3.10) |
| Docker build fails | Check `.dockerignore` excludes `venv/` |
| Secrets not working | Check secret names match workflow |
| Coverage upload fails | Install codecov token as secret |

---

## Advanced: Auto-Deploy to Cloud

**For Google Cloud Run:**

```yaml
deploy:
  steps:
    - name: Auth to GCP
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_SA_KEY }}
    
    - name: Deploy backend
      run: |
        gcloud run deploy pitchquest-backend \
          --image gcr.io/$PROJECT_ID/backend \
          --region us-central1 \
          --allow-unauthenticated
```

---

## Success Checklist

- [ ] CI runs on every push
- [ ] All tests pass
- [ ] Linting enforced
- [ ] Docker images build
- [ ] Pre-commit hooks installed
- [ ] Secrets configured
- [ ] Status badge shows passing
- [ ] Deploy script works

**CI/CD operational! ✅**