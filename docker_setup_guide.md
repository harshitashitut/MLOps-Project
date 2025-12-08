# 🐳 Docker Setup Guide - PitchQuest Multimodal

Complete guide to containerize and run the PitchQuest application.

## 📋 Prerequisites

- Docker Desktop installed (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- Git
- At least 8GB RAM available
- 20GB free disk space

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/harshitashitut/MLOps-Project.git
cd MLOps-Project
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

**Required variables:**
- `OPENAI_API_KEY` - For Whisper API
- `SUPABASE_URL` and `SUPABASE_KEY` - For database

### 3. Build and Run

**Development Mode (with hot reload):**
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

**Production Mode:**
```bash
docker-compose up -d --build
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## 📁 Project Structure

```
MLOps-Project/
├── backend/
│   ├── Dockerfile              # Production backend
│   ├── Dockerfile.dev          # Development backend
│   ├── .dockerignore
│   ├── app.py                  # Main FastAPI app
│   ├── requirements.txt
│   ├── outputs/               # Analysis results
│   ├── pipeline/              # ML pipeline
│   ├── prompts/               # LLM prompts
│   └── utils/                 # Utility functions
├── frontend/
│   ├── Dockerfile              # Production frontend
│   ├── Dockerfile.dev          # Development frontend
│   ├── .dockerignore
│   ├── nginx.conf              # Nginx config for prod
│   ├── package.json
│   ├── vite.config.js
│   ├── src/                   # React source
│   └── public/                # Static assets
├── docker-compose.yml          # Production compose
├── docker-compose.dev.yml      # Development compose
└── .env                        # Environment variables
```

## 🔧 Development Workflow

### Starting Development Environment

```bash
# Start all services with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or start specific services
docker-compose up backend
docker-compose up frontend
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend

# Last 50 lines
docker-compose logs --tail=50 backend
```

### Executing Commands Inside Containers

```bash
# Backend shell
docker-compose exec backend bash

# Install new Python package
docker-compose exec backend pip install <package>

# Frontend shell
docker-compose exec frontend sh

# Install new npm package
docker-compose exec frontend npm install <package>

# Database access
docker-compose exec postgres psql -U postgres -d pitchquest
```

### Rebuilding After Changes

```bash
# Rebuild specific service
docker-compose build backend
docker-compose up -d backend

# Rebuild all
docker-compose up -d --build
```

## 🏭 Production Deployment

### Build Production Images

```bash
# Build all services
docker-compose build

# Build and start
docker-compose up -d
```

### Check Health

```bash
# View running containers
docker-compose ps

# Check container health
docker inspect pitchquest-backend | grep -A 10 Health
```

### Resource Monitoring

```bash
# Real-time resource usage
docker stats

# Disk usage
docker system df

# Container logs
docker-compose logs --since 1h
```

## 🛠️ Common Commands

### Container Management

```bash
# Stop all containers
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v

# Restart specific service
docker-compose restart backend

# Remove and recreate container
docker-compose up -d --force-recreate backend
```

### Image Management

```bash
# List images
docker images

# Remove unused images
docker image prune -a

# Tag image for registry
docker tag pitchquest-backend:latest yourusername/pitchquest-backend:v1.0
```

### Database Management

```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres pitchquest > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres pitchquest < backup.sql

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # Mac/Linux
```

### Permission Denied

```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER .
```

### Out of Disk Space

```bash
# Clean up everything
docker system prune -a --volumes

# Remove specific items
docker image prune -a
docker volume prune
docker container prune
```

### Container Won't Start

```bash
# Check logs
docker-compose logs backend

# Check container exit code
docker-compose ps

# Inspect container
docker inspect pitchquest-backend

# Try recreating
docker-compose down
docker-compose up -d --force-recreate
```

### Model Loading Issues

```bash
# Clear model cache
docker-compose down -v
docker volume rm mlops-project_model_cache
docker-compose up -d

# Increase Docker memory (Docker Desktop Settings)
# Recommended: 8GB RAM minimum
```

## 📊 Health Checks

### Backend Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### Database Connection

```bash
docker-compose exec backend python -c "from app import db; print(db.ping())"
```

### Frontend Build

```bash
docker-compose exec frontend ls -la /usr/share/nginx/html
```

## 🔒 Security Best Practices

1. **Never commit .env file**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use secrets for production**
   ```bash
   # Use Docker secrets or environment variables from CI/CD
   ```

3. **Run as non-root user** (already configured in Dockerfiles)

4. **Keep images updated**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

## 📈 Performance Optimization

### Reduce Build Time

```bash
# Use BuildKit
export DOCKER_BUILDKIT=1
docker-compose build
```

### Multi-stage Builds (already implemented)

Frontend uses multi-stage builds:
1. Build stage (Node 18)
2. Production stage (Nginx Alpine)

### Layer Caching

- Copy `requirements.txt` / `package.json` before code
- Use `.dockerignore` to exclude unnecessary files

## 🚢 CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build images
        run: |
          docker-compose build
          
      - name: Run tests
        run: |
          docker-compose run backend pytest
          
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker-compose push
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify environment variables: `docker-compose config`
3. Try clean restart: `docker-compose down -v && docker-compose up -d`
4. Create issue on GitHub with logs

## 🎯 Next Steps

1. ✅ Containerize application
2. ⏭️ Set up CI/CD pipeline
3. ⏭️ Deploy to cloud (AWS ECS / Google Cloud Run)
4. ⏭️ Add monitoring (Prometheus + Grafana)
5. ⏭️ Implement auto-scaling

---

**Happy Containerizing! 🐳**