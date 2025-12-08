# PitchQuest - AI Video Analysis

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- API Keys:
  - OpenAI API key (for Whisper transcription)
  - Google Gemini API key (for video analysis)
  - Supabase project credentials

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Set up environment variables
```bash
cp .env.example .env
```

Then edit `.env` and fill in your API keys.

### 3. Run the application
```bash
docker-compose up --build
```

First build takes 3-5 minutes. Subsequent starts are faster.

### 4. Access the app
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### 5. Stop the application
```bash
docker-compose down
```

## Troubleshooting

**Container won't start?**
```bash
docker-compose logs backend
docker-compose logs frontend
```

**API keys not working?**
```bash
docker exec pitchquest-backend env | grep -E "OPENAI|GEMINI"
```

**Rebuild from scratch:**
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```
```

---

## Step 3: What Teammates Do

| Step | Command |
|------|---------|
| 1. Install Docker Desktop | Download from docker.com |
| 2. Clone repo | `git clone <repo-url>` |
| 3. Copy env template | `cp .env.example .env` |
| 4. Add their API keys | Edit `.env` file |
| 5. Run | `docker-compose up --build` |
| 6. Open browser | http://localhost:3000 |

---

## Optional: Share API Keys Securely

If you want teammates to use the **same** API keys (shared Supabase project, etc.):

1. **Don't commit keys to Git**
2. Share `.env` file via:
   - Slack DM (delete after they copy it)
   - Password manager (1Password, LastPass)
   - Encrypted file share

---

## Summary of Files to Commit
```
✅ Commit these:
├── docker-compose.yml
├── .env.example          # Template (no real keys)
├── .gitignore            # Must include .env
├── README.md
├── backend/
│   ├── Dockerfile
│   └── ... (all backend code)
└── frontend/
    ├── Dockerfile
    ├── nginx.conf
    └── ... (all frontend code)

❌ Do NOT commit:
├── .env                  # Contains real API keys
└── node_modules/         # Should already be ignored