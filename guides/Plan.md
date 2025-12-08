# PitchQuest MLOps Project - Master Game Plan

**Deadline:** December 9, 2025 (End of Day)  
**Team Size:** 3 people  
**Current Status:** Airflow pipeline 80% complete, everything else at 0%

---

## Executive Summary

### What We're Building
A production-grade MLOps system that analyzes startup pitch videos using AI, with:
- Next.js frontend for video upload and results visualization
- FastAPI backend for orchestration
- Apache Airflow for ML pipeline execution
- Supabase PostgreSQL for data persistence
- Automated deployment, monitoring, and drift detection

### Current Situation
✅ **Have:** Working Airflow pipeline that processes videos and generates analysis JSON  
❌ **Need:** Frontend, Backend, Database integration, Deployment, Monitoring, Documentation

### Success Criteria (Grading Rubric)
1. ✅ **Automated Deployment** - `docker-compose up` works without manual steps
2. ✅ **Clear Documentation** - README with setup, architecture diagram, video demo
3. ✅ **CI/CD Integration** - Automated testing and deployment pipeline
4. ✅ **Logging & Monitoring** - Dashboard showing drift, system health
5. ✅ **Graceful Error Handling** - No 500 errors exposed to users
6. ✅ **Professional UI** - Colorful, responsive, polished interface

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 USER (Web Browser)                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              NEXT.JS FRONTEND (:3000)                    │
│  • /upload - Video upload with drag-and-drop            │
│  • /results/[id] - Analysis dashboard with charts       │
│  • /report - Monitoring dashboard (drift, health)       │
└─────────────────────────────────────────────────────────┘
                         ↓ REST API
┌─────────────────────────────────────────────────────────┐
│              FASTAPI BACKEND (:8000)                     │
│  • POST /api/upload - Handle video upload               │
│  • POST /api/analyze/{id} - Trigger Airflow DAG         │
│  • GET /api/status/{job_id} - Check processing status   │
│  • GET /api/results/{video_id} - Fetch analysis         │
│  • GET /api/monitoring - Drift & system metrics         │
└─────────────────────────────────────────────────────────┘
         ↓                    ↓                  ↓
┌──────────────┐   ┌──────────────────┐   ┌────────────┐
│   SUPABASE   │   │ AIRFLOW (:8080)  │   │ EVIDENTLY  │
│  PostgreSQL  │←──│ • Preprocess     │──→│ Drift Check│
│              │   │ • Visual Analysis│   │            │
│  • videos    │   │ • Audio Analysis │   └────────────┘
│  • results   │   │ • Content LLM    │
│  • metrics   │   │ • Aggregation    │
└──────────────┘   └──────────────────┘
```

---

## Team Structure & Roles

### Person 1: Backend Lead
**Primary Responsibility:** FastAPI server, API endpoints, Airflow integration  
**Guide:** `02-backend-implementation.md`

**Day 1:** API skeleton, upload endpoint, Airflow client  
**Day 2:** Status polling, results retrieval, error handling  
**Day 3:** Rate limiting, validation, Swagger docs, testing

### Person 2: Frontend + Database Lead
**Primary Responsibility:** Next.js UI, Supabase setup, monitoring dashboard  
**Guide:** `03-frontend-implementation.md`

**Day 1:** Supabase tables, upload form UI  
**Day 2:** Results dashboard, status polling, visualizations  
**Day 3:** Monitoring page, UI polish, responsiveness

### Person 3: MLOps + DevOps Lead
**Primary Responsibility:** Airflow→DB integration, CI/CD, deployment, monitoring  
**Guide:** `04-airflow-db-integration.md`, `05-docker-compose.md`, `06-cicd-pipeline.md`, `07-monitoring-setup.md`

**Day 1:** Modify Airflow to save to Supabase, Docker Compose setup  
**Day 2:** Evidently AI drift detection, CI/CD pipeline  
**Day 3:** Production deployment, final testing, documentation

---

## 3-Day Sprint Breakdown

### Day 1 (Dec 6) - FOUNDATION [10-12 hours]
**Goal:** All components exist and can communicate

**Morning (4 hours):**
- Person 1: FastAPI skeleton with /health, /upload endpoints
- Person 2: Supabase project + tables, Next.js project init
- Person 3: Modify `airflow/utils/db_helper.py` to use Supabase

**Afternoon (6 hours):**
- Person 1: Implement Airflow REST API client
- Person 2: Upload form UI, basic routing
- Person 3: Test Airflow→Supabase connection, Docker Compose draft

**End of Day Checkpoint:**
- [ ] Backend returns 200 on `/health` and can accept file upload
- [ ] Frontend shows upload form
- [ ] Supabase has all tables created
- [ ] Airflow can save to Supabase (test with one video)

**Merge to main:** Each person pushes branch, team reviews together

---

### Day 2 (Dec 7) - INTEGRATION [12-14 hours]
**Goal:** Full end-to-end flow works

**Morning (6 hours):**
- Person 1: Implement `/api/analyze`, `/api/status` with Airflow polling
- Person 2: Results page with data visualization
- Person 3: Add Evidently AI drift detection task to DAG

**Afternoon (6 hours):**
- Person 1: Error handling, validation, API tests
- Person 2: Status polling in frontend, loading states
- Person 3: GitHub Actions CI/CD pipeline, monitoring endpoint

**End of Day Checkpoint:**
- [ ] Upload video → Triggers Airflow → See results (FULL FLOW)
- [ ] Error handling works (try corrupt video, large file)
- [ ] CI/CD runs on push
- [ ] Drift detection executes after analysis

**Integration Testing:** Everyone tests the full system together

---

### Day 3 (Dec 8) - PRODUCTION POLISH [12-14 hours]
**Goal:** Production-ready system

**Morning (4 hours) - ALL HANDS:**
- Run full flow 10 times with different videos
- Document every bug found
- Fix critical bugs together

**Afternoon (8 hours):**
- Person 1: Rate limiting, health checks, optimize API
- Person 2: UI polish (colors, animations, responsive design)
- Person 3: Production Docker Compose, deployment testing

**Evening:**
- Test on fresh machine (Person 3's laptop)
- Fix deployment issues
- Prepare for demo day

**End of Day Checkpoint:**
- [ ] `docker-compose up` brings up entire system
- [ ] Works on fresh machine
- [ ] UI looks professional
- [ ] All edge cases handled

---

### Day 4 (Dec 9) - DEMO & DOCS [8-10 hours]
**Goal:** Submission ready

**Morning (4 hours):**
- ALL: Record 5-minute video demo showing:
  - Fresh deployment (`docker-compose up`)
  - Upload video
  - Show results dashboard
  - Show monitoring page
  - Error handling demo
- Person 2: Create architecture diagram (Mermaid or draw.io)
- Person 1 & 3: Write comprehensive README

**Afternoon (4 hours):**
- Final testing on completely clean environment
- Practice live demo
- Prepare answers to common questions
- Submit project

---

## Critical Path Items (Must Complete)

### Blockers (If These Fail, Project Fails)
1. **Airflow → Supabase Connection:** Must work by end of Day 1
2. **Backend → Airflow Integration:** Must trigger DAG successfully
3. **Frontend → Backend Communication:** CORS, API calls must work
4. **Docker Compose Networking:** All services must reach each other

### High Priority (Grading Impact)
- Error handling everywhere (no crashes)
- Working monitoring dashboard
- CI/CD pipeline functional
- Video demo clear and professional
- Architecture diagram in README

### Nice to Have (Cut If Behind Schedule)
- Advanced animations in UI
- Real-time progress bar (polling is fine)
- Multiple video simultaneous upload
- User authentication
- Advanced drift visualizations

---

## Daily Coordination

### Morning Standup (9:00 AM, 15 minutes)
- What did you complete yesterday?
- What are you working on today?
- Any blockers?
- Integration points today?

### Evening Sync (7:00 PM, 30 minutes)
- Demo what you built
- Integration testing
- Plan for tomorrow
- Merge branches

### Communication Channels
- **Slack/Discord:** Quick questions, updates
- **GitHub Issues:** Track bugs and tasks
- **Shared Google Doc:** API contracts, environment variables

---

## Git Workflow

### Branch Strategy
```
main (protected)
  ├── backend (Person 1)
  ├── frontend (Person 2)
  └── mlops (Person 3)
```

### Daily Flow
```bash
# Morning
git checkout -b backend  # or frontend/mlops
git pull origin main

# Work on your code...

# Evening (before standup)
git add .
git commit -m "feat: implement upload endpoint"
git push origin backend

# Create PR, team reviews together
# Merge to main after testing
```

### Merge Requirements
- [ ] Code runs without errors
- [ ] Tests pass (if applicable)
- [ ] Doesn't break existing features
- [ ] Team member reviews code

---

## Shared Resources

### Environment Variables (.env)
**Person 2 creates Supabase project, shares credentials with team:**
```env
# Supabase (shared)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
DATABASE_URL=postgresql://postgres:password@db.xxx.supabase.co:5432/postgres

# APIs (each person gets their own)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...

# Airflow
AIRFLOW_URL=http://localhost:8080
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin
```

**Share via:** Secure note in Slack, NOT in Git

### API Contract (Document First)
**Person 1 defines API endpoints in shared doc BEFORE implementation:**
```
POST /api/upload
  Request: multipart/form-data with video file
  Response: {video_id, status, message}

POST /api/analyze/{video_id}
  Response: {job_id, status}

GET /api/status/{job_id}
  Response: {status: "running"|"success"|"failed", progress: 0-100}

GET /api/results/{video_id}
  Response: {overall_score, content_analysis, delivery_analysis, ...}
```

Person 2 uses this contract to build frontend

---

## Risk Management

### Top Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Integration fails | High | Critical | Test integration on Day 1, not Day 3 |
| Docker networking issues | Medium | High | Use Docker Compose from Day 1 |
| Supabase connection errors | Medium | High | Test DB connection first thing Day 1 |
| Running out of time | High | Critical | Cut nice-to-haves, focus on core flow |
| Team member blocked | Medium | Medium | Daily standups, help each other |
| Deployment fails | Medium | High | Test on fresh VM on Day 3 afternoon |

### Emergency Scope Reduction
**If significantly behind by end of Day 2, cut in this order:**
1. ~~Advanced UI animations~~ → Basic but functional UI
2. ~~Real-time progress~~ → Just show "Processing..."
3. ~~Drift visualization~~ → Just log drift detection
4. ~~Cloud deployment~~ → Local Docker Compose only

**NEVER CUT:**
- Working end-to-end flow
- Error handling
- Basic monitoring
- Documentation
- Video demo

---

## Success Metrics

### Technical Metrics
- [ ] System runs on fresh machine with `docker-compose up`
- [ ] Full flow (upload → results) completes in <3 minutes
- [ ] Zero unhandled exceptions (all errors caught)
- [ ] API response time <500ms for all endpoints
- [ ] CI/CD pipeline runs in <5 minutes

### Quality Metrics
- [ ] Architecture diagram clear and accurate
- [ ] README has complete setup instructions
- [ ] Video demo shows entire flow without cuts
- [ ] UI works on mobile and desktop
- [ ] All edge cases handled gracefully

### Team Metrics
- [ ] All 3 members contribute equally
- [ ] Daily standups keep everyone aligned
- [ ] No major merge conflicts
- [ ] Code reviews completed same day
- [ ] Everyone can explain the full system

---

## Submission Checklist

### Code Repository
- [ ] All code in `main` branch
- [ ] `.env.example` with placeholders (NO real keys!)
- [ ] `.gitignore` properly configured
- [ ] All branches merged and cleaned up

### Documentation
- [ ] README.md with quick start
- [ ] ARCHITECTURE.md with system diagram
- [ ] API documentation (Swagger or markdown)
- [ ] Deployment guide
- [ ] Individual component guides (this folder!)

### Demo Materials
- [ ] 5-minute video demo uploaded
- [ ] Architecture diagram (PNG or Mermaid)
- [ ] Screenshots of UI
- [ ] Live demo prepared (backup plan if wifi fails)

### Functionality
- [ ] Video upload works
- [ ] Analysis completes successfully
- [ ] Results display correctly
- [ ] Monitoring dashboard accessible
- [ ] Error handling tested
- [ ] Docker Compose deployment tested

---

## Contact & Coordination

### Team Member Responsibilities

**Person 1 (Backend Lead):**
- Email: [your-email]
- Best time: [your-hours]
- Slack: @username

**Person 2 (Frontend + DB Lead):**
- Email: [your-email]
- Best time: [your-hours]
- Slack: @username

**Person 3 (MLOps + DevOps Lead):**
- Email: [your-email]
- Best time: [your-hours]
- Slack: @username

### Emergency Protocol
If someone is blocked for >1 hour:
1. Post in team chat immediately
2. Jump on quick call to debug together
3. Reassign task if necessary

---

## Motivation

**Remember:** You already have 80% of the ML pipeline working. This is about:
- Wrapping it in a professional application
- Making it production-ready
- Showcasing your MLOps skills

**This is achievable in 3 days because:**
- You have a strong foundation (Airflow pipeline)
- Team of 3 skilled people
- Clear plan and division of work
- Modern tools that speed up development

**You've got this! 🚀**

---

## Quick Reference

### Essential Commands
```bash
# Start entire system
docker-compose up --build

# Run tests
pytest tests/ -v

# Check API
curl http://localhost:8000/health

# View logs
docker-compose logs -f backend
```

### Essential Files
- `docker-compose.yml` - Full system orchestration
- `backend/main.py` - API entry point
- `frontend/app/page.tsx` - Upload UI
- `airflow/dags/pitch_analysis_dag.py` - ML pipeline

### Essential URLs
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Airflow UI: http://localhost:8080
- Supabase: https://app.supabase.com

---

**Now go to the individual guides for detailed implementation instructions!**