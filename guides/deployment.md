# Cloud Deployment Guide - Compact

**Owner:** Person 3 | **Timeline:** Day 3 (2-4 hours)

---

## Goal
Deploy full stack to cloud with one command. Options: GCP, Railway, or VM.

---

## Deployment Options

| Platform | Cost | Complexity | Best For |
|----------|------|------------|----------|
| **Railway** | ~$20/mo | Low ⭐ | Quick demo, no DevOps experience |
| **Google Cloud Run** | ~$30/mo | Medium ⭐⭐ | Scalable, auto-scaling |
| **VM (GCP/AWS)** | ~$40/mo | High ⭐⭐⭐ | Full control, team has DevOps skills |

**Recommendation for 3-day deadline: Railway** (simplest, fastest)

---

## Option 1: Railway Deployment (EASIEST) ⭐

### Step 1: Prepare for Railway (30 min)

**Create `railway.toml`:**
```toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "docker-compose up"
healthcheckPath = "/health"
healthcheckTimeout = 300
```

**Update `docker-compose.yml` for Railway:**
```yaml
# Add to each service:
services:
  backend:
    environment:
      - PORT=${PORT:-8000}  # Railway assigns PORT
    # Remove 'ports' section, Railway handles it
```

### Step 2: Deploy to Railway (30 min)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Add services
railway add  # Select: PostgreSQL, Redis

# Set environment variables
railway variables set SUPABASE_URL="https://..."
railway variables set SUPABASE_KEY="eyJ..."
railway variables set OPENAI_API_KEY="sk-..."
railway variables set GEMINI_API_KEY="AIza..."
railway variables set DATABASE_URL="postgresql://..."

# Deploy
railway up

# Get URL
railway open
```

**Custom domain (optional):**
```bash
railway domain  # Follow prompts
```

---

## Option 2: Google Cloud Run (RECOMMENDED) ⭐⭐

### Step 1: Prepare Images (1 hour)

**Build and push to Google Container Registry:**

```bash
# Set project
export PROJECT_ID="pitchquest-prod"
gcloud config set project $PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build images
docker build -t gcr.io/$PROJECT_ID/backend ./backend
docker build -t gcr.io/$PROJECT_ID/frontend ./frontend
docker build -t gcr.io/$PROJECT_ID/airflow ./airflow

# Push to GCR
docker push gcr.io/$PROJECT_ID/backend
docker push gcr.io/$PROJECT_ID/frontend
docker push gcr.io/$PROJECT_ID/airflow
```

### Step 2: Deploy Services (1 hour)

**Deploy Backend:**
```bash
gcloud run deploy pitchquest-backend \
  --image gcr.io/$PROJECT_ID/backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "SUPABASE_URL=$SUPABASE_URL,SUPABASE_KEY=$SUPABASE_KEY" \
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,GEMINI_API_KEY=$GEMINI_API_KEY" \
  --set-env-vars "AIRFLOW_URL=https://pitchquest-airflow-xxx.run.app" \
  --memory 2Gi \
  --timeout 300
```

**Deploy Frontend:**
```bash
gcloud run deploy pitchquest-frontend \
  --image gcr.io/$PROJECT_ID/frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "NEXT_PUBLIC_API_URL=https://pitchquest-backend-xxx.run.app" \
  --memory 1Gi
```

**Deploy Airflow (on Compute Engine VM):**
```bash
# Create VM
gcloud compute instances create pitchquest-airflow \
  --machine-type=e2-standard-2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# SSH and setup
gcloud compute ssh pitchquest-airflow

# On VM:
sudo apt update
sudo apt install -y docker.io docker-compose
git clone YOUR_REPO
cd MLOps-Project
# Add .env file with secrets
docker-compose up -d airflow-webserver airflow-scheduler
```

**Get URLs:**
```bash
# Backend URL
gcloud run services describe pitchquest-backend --format='value(status.url)'

# Frontend URL
gcloud run services describe pitchquest-frontend --format='value(status.url)'

# Airflow URL (VM external IP)
gcloud compute instances describe pitchquest-airflow --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
# Access at: http://EXTERNAL_IP:8080
```

### Step 3: Setup Cloud SQL for Airflow (optional)

```bash
# Create PostgreSQL instance
gcloud sql instances create pitchquest-db \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=us-central1

# Create database
gcloud sql databases create airflow --instance=pitchquest-db

# Get connection string
gcloud sql instances describe pitchquest-db --format='value(connectionName)'

# Update Airflow env to use Cloud SQL
# In docker-compose.yml:
# AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://user:pass@/airflow?host=/cloudsql/PROJECT:REGION:INSTANCE
```

---

## Option 3: VM Deployment (Full Control) ⭐⭐⭐

### Step 1: Setup VM (30 min)

**On GCP:**
```bash
gcloud compute instances create pitchquest-vm \
  --machine-type=e2-standard-4 \
  --boot-disk-size=100GB \
  --image-family=ubuntu-2204-lts \
  --tags=http-server,https-server

gcloud compute ssh pitchquest-vm
```

**On VM - Install Docker:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repo
git clone YOUR_REPO_URL
cd MLOps-Project
```

### Step 2: Setup Environment (15 min)

```bash
# Create .env
nano .env
# Paste all secrets

# Setup directories
mkdir -p data/input data/temp data/output logs
```

### Step 3: Deploy with Docker Compose (30 min)

```bash
# Pull images (if using pre-built)
docker-compose pull

# Or build locally
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 4: Setup Reverse Proxy (Nginx) (30 min)

**Install Nginx:**
```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

**Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/pitchquest
```

```nginx
# Frontend
server {
    listen 80;
    server_name pitchquest.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Backend API
server {
    listen 80;
    server_name api.pitchquest.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Airflow UI
server {
    listen 80;
    server_name airflow.pitchquest.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/pitchquest /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**Setup SSL (Let's Encrypt):**
```bash
sudo certbot --nginx -d pitchquest.com -d api.pitchquest.com -d airflow.pitchquest.com
```

### Step 5: Setup Auto-restart (15 min)

```bash
# Create systemd service
sudo nano /etc/systemd/system/pitchquest.service
```

```ini
[Unit]
Description=PitchQuest Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/USER/MLOps-Project
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable pitchquest
sudo systemctl start pitchquest
```

---

## Automated Deployment Script

**File: `scripts/deploy-production.sh`**

```bash
#!/bin/bash
set -e

echo "🚀 Deploying to production..."

# Configuration
SERVER="your-vm-ip-or-domain"
USER="your-username"
REPO_DIR="/home/$USER/MLOps-Project"

# SSH and deploy
ssh $USER@$SERVER << 'EOF'
cd /home/your-username/MLOps-Project

# Pull latest code
git pull origin main

# Rebuild images
docker-compose build --no-cache

# Restart services with zero downtime
docker-compose up -d --force-recreate

# Health checks
sleep 15
curl -f http://localhost:8000/health || exit 1
curl -f http://localhost:3000 || exit 1
curl -f http://localhost:8080/health || exit 1

# Show status
docker-compose ps

echo "✅ Deployment complete!"
EOF

echo "🎉 Production updated successfully"
```

**Usage:**
```bash
chmod +x scripts/deploy-production.sh
./scripts/deploy-production.sh
```

---

## CI/CD Integration

**Update `.github/workflows/main.yml`:**

```yaml
deploy:
  needs: [build-images]
  if: github.ref == 'refs/heads/main'
  runs-on: ubuntu-latest
  
  steps:
    - uses: actions/checkout@v3
    
    # Option 1: Deploy to Railway
    - name: Deploy to Railway
      run: |
        npm i -g @railway/cli
        railway up --service backend
        railway up --service frontend
      env:
        RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
    
    # Option 2: Deploy to GCP
    - name: Deploy to Cloud Run
      uses: google-github-actions/deploy-cloudrun@v1
      with:
        service: pitchquest-backend
        image: gcr.io/${{ secrets.GCP_PROJECT }}/backend
        credentials: ${{ secrets.GCP_SA_KEY }}
    
    # Option 3: Deploy to VM
    - name: Deploy to VM
      run: |
        echo "${{ secrets.SSH_PRIVATE_KEY }}" > key.pem
        chmod 600 key.pem
        ssh -i key.pem -o StrictHostKeyChecking=no ${{ secrets.VM_USER }}@${{ secrets.VM_HOST }} \
          "cd MLOps-Project && git pull && docker-compose up -d --build"
```

---

## Environment Variables for Production

**Create `.env.production`:**

```env
# Supabase (production project)
SUPABASE_URL=https://prod-xxx.supabase.co
SUPABASE_KEY=eyJprod...
DATABASE_URL=postgresql://postgres:pwd@db.prod.supabase.co:5432/postgres

# API Keys
OPENAI_API_KEY=sk-prod-...
GEMINI_API_KEY=AIzaprod...

# Service URLs (update after deployment)
NEXT_PUBLIC_API_URL=https://api.pitchquest.com
AIRFLOW_URL=https://airflow.pitchquest.com

# Production settings
NODE_ENV=production
DEBUG=false
LOG_LEVEL=info
```

---

## Post-Deployment Checklist

### Immediately After Deploy:
- [ ] All services respond to health checks
- [ ] Frontend loads at domain
- [ ] Backend API accessible
- [ ] Airflow UI accessible
- [ ] Can upload video end-to-end
- [ ] Results appear in database

### Within 24 Hours:
- [ ] Setup monitoring alerts
- [ ] Configure backups (database, volumes)
- [ ] Setup log aggregation
- [ ] Test auto-scaling (if applicable)
- [ ] Load testing with multiple concurrent uploads

### Domain Setup:
```bash
# Point domains to your deployment
# For Railway: Automatic
# For GCP: Add custom domain in Cloud Run console
# For VM: Update DNS A records to VM IP
```

---

## Cost Estimates (Monthly)

**Railway (Easiest):**
- Hobby Plan: $5/mo + usage (~$15/mo)
- Total: ~$20/mo

**GCP Cloud Run (Recommended):**
- Backend: ~$10/mo (100 requests/day)
- Frontend: ~$5/mo
- Airflow VM (e2-standard-2): ~$30/mo
- Cloud SQL (optional): ~$10/mo
- Total: ~$45/mo

**Self-Hosted VM:**
- e2-standard-4 VM: ~$40/mo
- Storage: ~$5/mo
- Total: ~$45/mo

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Services can't reach each other | Use service names, not localhost |
| Frontend can't reach backend | Update `NEXT_PUBLIC_API_URL` |
| SSL certificate errors | Run `certbot renew` |
| Out of memory | Increase VM size or container limits |
| Slow performance | Add Redis caching, CDN |

---

## Scaling Considerations

**If you need to scale (>100 users):**

1. **Horizontal scaling:** Add more backend/frontend instances
2. **Database:** Use connection pooling, read replicas
3. **Caching:** Add Redis for API responses
4. **CDN:** Use Cloudflare for static assets
5. **Load balancer:** Distribute traffic across instances

---

## Quick Reference

**Deploy to Railway:**
```bash
railway up
```

**Deploy to GCP:**
```bash
gcloud run deploy pitchquest-backend --image gcr.io/$PROJECT_ID/backend
```

**Deploy to VM:**
```bash
ssh user@vm "cd repo && git pull && docker-compose up -d"
```

**Rollback:**
```bash
# Railway: Use Railway dashboard
# GCP: gcloud run services update --image=PREVIOUS_IMAGE
# VM: git checkout PREVIOUS_COMMIT && docker-compose up -d
```

---

## Success Criteria

- [ ] Entire stack accessible via public URL
- [ ] One-command deployment works
- [ ] Automated via CI/CD
- [ ] SSL/HTTPS enabled
- [ ] Health checks passing
- [ ] Can demo live to professor
- [ ] Documentation includes deployment URL

**Production deployment complete! 🌐**