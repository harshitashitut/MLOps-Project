# Docker Compose Setup - Compact Guide

**Owner:** Person 3 | **Timeline:** Day 1 (2-3 hours)

---

## Goal
Single command deployment: `docker-compose up` runs entire stack.

---

## File Structure

```
MLOps-Project/
├── docker-compose.yml           # Main orchestration
├── backend/Dockerfile
├── frontend/Dockerfile
├── airflow/Dockerfile
└── .env                         # Shared secrets
```

---

## Step 1: Root docker-compose.yml (30 min)

```yaml
version: '3.8'

services:
  # PostgreSQL for Airflow metadata
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5

  # Redis for Airflow Celery
  redis:
    image: redis:latest
    expose:
      - 6379
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 30s
      retries: 50

  # Airflow Webserver
  airflow-webserver:
    build: ./airflow
    command: webserver
    ports:
      - "8080:8080"
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CORE__FERNET_KEY=''
      - AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
      - AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.basic_auth
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/pipeline:/opt/airflow/pipeline
      - ./airflow/utils:/opt/airflow/utils
      - ./airflow/data:/opt/airflow/data
      - ./airflow/logs:/opt/airflow/logs
    depends_on:
      postgres:
        condition: service_healthy

  # Airflow Scheduler
  airflow-scheduler:
    build: ./airflow
    command: scheduler
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
      - AIRFLOW__CORE__FERNET_KEY=''
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/pipeline:/opt/airflow/pipeline
      - ./airflow/utils:/opt/airflow/utils
      - ./airflow/data:/opt/airflow/data
      - ./airflow/logs:/opt/airflow/logs
    depends_on:
      postgres:
        condition: service_healthy

  # Backend API
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - DATABASE_URL=${DATABASE_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - AIRFLOW_URL=http://airflow-webserver:8080
      - AIRFLOW_USERNAME=admin
      - AIRFLOW_PASSWORD=admin
    volumes:
      - ./backend:/app
      - upload-volume:/tmp/pitchquest_uploads
    depends_on:
      - airflow-webserver

  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

volumes:
  postgres-db-volume:
  upload-volume:
```

---

## Step 2: Backend Dockerfile (15 min)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./app/

# Upload directory
RUN mkdir -p /tmp/pitchquest_uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 3: Frontend Dockerfile (15 min)

```dockerfile
FROM node:18-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM node:18-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]
```

**Update `next.config.js`:**
```javascript
module.exports = {
  output: 'standalone',
}
```

---

## Step 4: Airflow Dockerfile (15 min)

```dockerfile
FROM apache/airflow:2.7.3-python3.10

USER root
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Python deps
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# DAGs and pipeline code
COPY dags/ ${AIRFLOW_HOME}/dags/
COPY pipeline/ ${AIRFLOW_HOME}/pipeline/
COPY utils/ ${AIRFLOW_HOME}/utils/

# Data directories
RUN mkdir -p ${AIRFLOW_HOME}/data/input ${AIRFLOW_HOME}/data/temp ${AIRFLOW_HOME}/data/output
```

---

## Step 5: Initialize Airflow (15 min)

**File: `airflow/init.sh`**

```bash
#!/bin/bash
set -e

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "✅ Airflow initialized"
```

Add to `docker-compose.yml`:

```yaml
airflow-init:
  build: ./airflow
  entrypoint: /bin/bash
  command: -c "chmod +x /opt/airflow/init.sh && /opt/airflow/init.sh"
  environment:
    - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
  volumes:
    - ./airflow/init.sh:/opt/airflow/init.sh
  depends_on:
    postgres:
      condition: service_healthy
```

Update other Airflow services:
```yaml
depends_on:
  airflow-init:
    condition: service_completed_successfully
```

---

## Step 6: .env File (5 min)

**Create `.env` in root:**

```env
# Supabase
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
DATABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres

# API Keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

---

## Step 7: Test Deployment (30 min)

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Verify services
curl http://localhost:8000/health    # Backend
curl http://localhost:3000           # Frontend
curl http://localhost:8080/health    # Airflow

# Full test
# 1. Upload video via frontend (http://localhost:3000)
# 2. Check Airflow UI (http://localhost:8080)
# 3. Verify results appear

# Stop services
docker-compose down
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Port conflicts | Change ports: `"8081:8080"` |
| Airflow can't reach backend | Use service name: `http://backend:8000` |
| Frontend can't reach backend | Check `NEXT_PUBLIC_API_URL` |
| Volume permission errors | Add to service: `user: "${UID}:${GID}"` |
| Database connection fails | Check `DATABASE_URL` format |

---

## Production Optimizations

**For production, add:**

```yaml
# docker-compose.prod.yml
services:
  backend:
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  
  frontend:
    restart: always
    environment:
      - NEXT_PUBLIC_API_URL=https://api.yourdomain.com
  
  airflow-webserver:
    restart: always
```

**Deploy:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Success Checklist

- [ ] `docker-compose up` starts all services
- [ ] All health checks pass
- [ ] Frontend accessible at :3000
- [ ] Backend accessible at :8000
- [ ] Airflow UI accessible at :8080
- [ ] Can upload video and get results
- [ ] Logs show no errors
- [ ] Can restart services without issues

**Deployment ready! 🐳**