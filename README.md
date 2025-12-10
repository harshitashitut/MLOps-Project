# PitchQuest: AI-Powered Public Speaking Analysis System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED.svg)](https://github.com/harshitashitut/MLOps-Project/tree/feature/docker-containerization)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/harshitashitut/MLOps-Project/actions)

> **🚀 Latest Update: Full CI/CD Pipeline with Docker**  
> Automated testing and deployment using GitHub Actions. Check out our [`feature/docker-containerization`](https://github.com/harshitashitut/MLOps-Project/tree/feature/docker-containerization) branch for complete setup.

> **Note**: This project is under active development for MLOps coursework. Architecture and implementation details are subject to change.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [🐳 Docker Setup](#-docker-setup)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Analysis Pipeline](#analysis-pipeline)
- [CI/CD Pipeline](#cicd-pipeline)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

PitchQuest is an AI-powered platform that provides comprehensive feedback on public speaking, presentations, and pitch delivery. The system analyzes videos across three critical dimensions:

### 🎤 Content Analysis
- **Structure & Clarity**: Evaluates speech organization and message clarity
- **Engagement**: Measures audience connection and persuasiveness
- **Language Quality**: Assesses vocabulary, grammar, and articulation
- **Message Focus**: Analyzes coherence and key message delivery

### 🗣️ Delivery Analysis
- **Speaking Pace**: 155 WPM optimal range analysis
- **Vocal Confidence**: Emotion detection via Wav2Vec2
- **Filler Words**: Automated detection and counting
- **Articulation**: Speech clarity and pronunciation assessment

### 👁️ Visual Analysis
- **Body Language**: Posture and gesture analysis via Gemini Vision
- **Eye Contact**: Camera engagement measurement
- **Confidence Level**: Overall visual presence scoring
- **Professional Appearance**: Background and presentation quality

## ✨ Key Features

### 🤖 Advanced AI Analysis
- **Multimodal Processing**: Simultaneous audio, video, and visual analysis
- **Pre-trained Models**: OpenAI Whisper, Wav2Vec2, Gemini Vision, Gemini Pro
- **Fast Processing**: 3-5 minute analysis for typical videos
- **Comprehensive Feedback**: Detailed scores, strengths, and improvement areas

### 📊 User Experience
- **Real-time Upload**: Drag-and-drop video interface
- **Detailed Dashboard**: Category scores, metrics, and recommendations
- **Progress Tracking**: Historical performance visualization
- **Authentication**: Secure user accounts via Supabase

### 🔐 Privacy & Security
- **Secure Storage**: Encrypted video and data storage
- **User Control**: Easy data deletion
- **Authentication**: JWT-based secure access

## 🏗️ Architecture

<p align="center">
  <img src="Blank diagram.png" alt="PitchQuest Model Architecture">
</p>
<p align="center"><i>PitchQuest Model Architecture Overview</i></p>
## 🛠️ Technology Stack

### Frontend
- **Framework**: React 18 with Vite
- **Language**: JavaScript/JSX
- **Styling**: Tailwind CSS
- **UI Components**: Lucide React icons
- **Routing**: React Router
- **State Management**: React Hooks

### Backend
- **API Framework**: FastAPI 0.104+
- **Language**: Python 3.10
- **Video Processing**: FFmpeg, OpenCV
- **Authentication**: Supabase Auth
- **Background Tasks**: FastAPI BackgroundTasks

### AI/ML Models
- **Speech-to-Text**: OpenAI Whisper API
- **Emotion Recognition**: Wav2Vec2 (HuggingFace)
- **Visual Analysis**: Google Gemini Vision (2.5-flash)
- **Content Analysis**: Google Gemini Pro (2.5-pro)
- **Pose Estimation**: MediaPipe (placeholder, uses Gemini Vision)

### Infrastructure
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth
- **Storage**: Local filesystem (uploads/outputs)
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Deployment**: Google Cloud Platform (Cloud Run)

## 📁 Project Structure

```
MLOps-Project/
├── frontend/                   # React + Vite application
│   ├── src/
│   │   ├── api/               # API service layer
│   │   │   └── videoAnalysis.js
│   │   ├── components/        # React components
│   │   │   └── ui/           # Reusable UI components
│   │   ├── context/          # React context (Auth)
│   │   ├── lib/              # Utilities
│   │   │   └── supabase.js  # Supabase client
│   │   ├── pages/            # Page components
│   │   │   ├── Home.jsx
│   │   │   └── PublicSpeaking.jsx
│   │   ├── App.jsx           # Main app component
│   │   └── main.jsx          # Entry point
│   ├── public/               # Static assets
│   ├── Dockerfile            # Frontend container
│   ├── nginx.conf            # Production nginx config
│   ├── nginx.standalone.conf # CI testing config
│   ├── package.json
│   └── vite.config.js
│
├── backend/                   # FastAPI service
│   ├── pipeline/             # ML analysis pipeline
│   │   ├── preprocessing.py  # Frame/audio extraction
│   │   ├── visual_analysis.py # Gemini Vision analysis
│   │   ├── audio_analysis.py  # Whisper + Wav2Vec2
│   │   ├── content_analysis.py # Gemini content analysis
│   │   └── aggregation.py    # Final report generation
│   ├── services/             # Business logic
│   │   └── analysis_service.py
│   ├── utils/                # Helper modules
│   │   ├── config.py        # Configuration
│   │   ├── db_helper.py     # Database operations
│   │   ├── gemini_client.py # Gemini API client
│   │   ├── validators.py    # Input validation
│   │   └── error_handlers.py
│   ├── prompts/              # AI prompts
│   │   ├── visual_analysis.txt
│   │   ├── content_analysis.txt
│   │   └── aggregation.txt
│   ├── uploads/              # Temporary video storage
│   ├── outputs/              # Analysis results (JSON)
│   ├── app.py                # FastAPI application
│   ├── Dockerfile            # Backend container
│   └── requirements.txt      # Python dependencies
│
├── .github/
│   └── workflows/            # CI/CD pipelines
│       ├── backend-ci.yml   # Backend testing
│       └── frontend-ci.yml  # Frontend testing
│
├── docker-compose.yml        # Multi-container orchestration
├── .env.example              # Environment template
├── .gitignore
└── README.md
```

## 🐳 Docker Setup

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+

### Quick Start with Docker

```bash
# 1. Clone repository
git clone https://github.com/harshitashitut/MLOps-Project.git
cd MLOps-Project

# 2. Switch to Docker branch (recommended)
git checkout feature/docker-containerization

# 3. Create environment file
cat > .env << 'EOF'
# Frontend
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_URL=http://localhost:8000

# Backend
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
EOF

# 4. Build and start all services
docker compose up --build

# 5. Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 | React app with Nginx |
| Backend | 8000 | FastAPI server |

### Useful Docker Commands

```bash
# Start services in background
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild after changes
docker compose up --build

# Check service health
docker compose ps

# Access backend shell
docker compose exec backend bash
```

## 🚀 Getting Started

### Local Development (Without Docker)

#### Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SUPABASE_URL="your_url"
export SUPABASE_KEY="your_key"
export OPENAI_API_KEY="your_key"
export GEMINI_API_KEY="your_key"

# Run server
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env file
cat > .env.local << 'EOF'
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_API_URL=http://localhost:8000
EOF

# Start development server
npm run dev

# Access at http://localhost:5173
```

### Environment Variables

#### Required for Backend:
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIza...
```

#### Required for Frontend:
```bash
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key
VITE_API_URL=http://localhost:8000
```

## 🧠 Analysis Pipeline

### Pipeline Architecture

The video analysis pipeline consists of five sequential stages:

```
1. Preprocessing
   ├─ Extract frames (1 per 2 seconds, max 10)
   └─ Extract audio (MP3, 16kHz)

2. Visual Analysis
   ├─ MediaPipe pose detection (placeholder)
   └─ Gemini Vision multimodal analysis

3. Audio Analysis
   ├─ Whisper transcription (OpenAI API)
   ├─ Wav2Vec2 emotion detection
   └─ Vocal metrics calculation (WPM, fillers)

4. Content Analysis
   └─ Gemini Pro text analysis

5. Aggregation
   └─ Gemini Pro comprehensive report generation
```

### Processing Time

- **Short video (< 1 min)**: ~2-3 minutes
- **Medium video (1-5 min)**: ~3-5 minutes
- **Long video (5-10 min)**: ~5-8 minutes

### Output Structure

```json
{
  "overall_score": 47,
  "performance_level": "beginner",
  "category_scores": {
    "content": {"score": 30, "structure": 10, "clarity": 60},
    "vocal_delivery": {"score": 90, "pace": 95, "articulation": 63},
    "visual_presentation": {"score": 20, "posture": 0, "eye_contact": 80},
    "tone_emotion": {"score": 48, "dominant_emotion": "surprised"}
  },
  "key_metrics": {
    "speech_duration_seconds": 23.21,
    "words_per_minute": 155.1,
    "filler_words_count": 1
  },
  "strengths": [...],
  "improvements": [...],
  "detailed_feedback": {...}
}
```

## 🔄 CI/CD Pipeline

### Automated Testing Workflow

Our GitHub Actions CI/CD pipeline runs on every push:

#### Backend CI (`backend-ci.yml`)
```
Job 1: Build Docker Image (1-2 min)
  ✅ Dockerfile builds
  ✅ Python dependencies install
  ✅ System packages available

Job 2: Test Database Connection (2-3 min)
  ✅ Supabase connection works
  ✅ Health checks pass
  ✅ Container runs successfully

Job 3: CI Complete
```

#### Frontend CI (`frontend-ci.yml`)
```
Job 1: Build & Test (2-3 min)
  ✅ npm build succeeds
  ✅ Docker image builds
  ✅ Nginx serves content
  ✅ Assets load correctly

Job 2: CI Complete
```

### Triggering CI

CI automatically runs when:
- Code pushed to `main`, `develop`, or `feature/**` branches
- Pull request created to `main`
- Only if relevant files changed (`backend/**` or `frontend/**`)

### Viewing CI Results

1. Go to [Actions tab](https://github.com/harshitashitut/MLOps-Project/actions)
2. Click on latest workflow run
3. View detailed logs for each job

## 📖 API Documentation

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-app.run.app` (after deployment)

### Authentication

All endpoints except `/` and `/health` require authentication:

```bash
# Login
POST /api/auth/login
{
  "email": "user@example.com",
  "password": "password"
}

# Returns JWT token
# Use in subsequent requests:
Headers: {
  "Authorization": "Bearer YOUR_TOKEN"
}
```

### Key Endpoints

#### Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "database": "healthy",
  "storage": {"upload_dir": "True", "output_dir": "True"},
  "timestamp": "2025-12-06T..."
}
```

#### Upload Video for Analysis
```bash
POST /api/analyze-video
Content-Type: multipart/form-data

Parameters:
- video: file (required) - Video file (MP4, MOV, AVI, MKV, max 500MB)
- X-User-ID: header (optional) - User identifier

Response:
{
  "status": "processing",
  "video_id": "uuid",
  "message": "Video uploaded successfully. Analysis in progress.",
  "poll_url": "/api/status/uuid",
  "timestamp": "2025-12-06T..."
}
```

#### Check Analysis Status
```bash
GET /api/status/{video_id}

Response:
{
  "video_id": "uuid",
  "status": "completed",  # or "processing", "failed"
  "progress": 100,
  "current_step": "Aggregation complete",
  "overall_score": 47
}
```

#### Get Analysis Results
```bash
GET /api/results/{video_id}

Response: (Full analysis JSON - see Output Structure above)
```

#### Get User Videos
```bash
GET /api/videos
Headers: X-User-ID: your_user_id

Response:
{
  "user_id": "uuid",
  "count": 5,
  "videos": [
    {
      "video_id": "uuid",
      "filename": "presentation.mp4",
      "status": "completed",
      "overall_score": 47,
      "uploaded_at": "2025-12-06T..."
    }
  ]
}
```

**Interactive API Docs**: Visit `http://localhost:8000/docs` when backend is running

## 📊 Performance Metrics

### Current Performance

| Metric | Value | Status |
|--------|-------|--------|
| Processing Time (1-min video) | 2-3 minutes | 🟢 Good |
| Processing Time (5-min video) | 3-5 minutes | 🟢 Good |
| API Success Rate | ~95% | 🟡 Gemini rate limits |
| Docker Build Time | 2-3 minutes | 🟢 Good |
| CI Pipeline Duration | 5-8 minutes | 🟢 Good |

### Cost per Analysis

- **OpenAI Whisper**: ~$0.006 per minute
- **Gemini Vision**: ~$0.05 per request
- **Gemini Pro**: ~$0.03 per request
- **Wav2Vec2**: Free (local inference)

**Total**: ~$0.15-0.25 per analysis (depending on video length)

## 🚀 Deployment

### Current Status

- ✅ **Local Development**: Docker Compose
- ✅ **CI/CD**: GitHub Actions
- 🚧 **Production Deployment**: In progress (GCP Cloud Run)

### Deployment to GCP (Planned)

The application will be deployed to Google Cloud Platform:

```
GitHub (feature branch)
        ↓
CI Tests Pass
        ↓
Push Docker images to Google Container Registry
        ↓
Deploy to Cloud Run (Staging)
        ↓
Smoke tests pass
        ↓
Deploy to Cloud Run (Production)
```

**Services:**
- **Backend**: Cloud Run (serverless containers)
- **Frontend**: Cloud Run (static hosting with nginx)
- **Database**: Supabase (managed PostgreSQL)
- **Storage**: Cloud Storage (for video archives)

## 🤝 Contributing

We welcome contributions! Here's how:

### Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Make changes and test locally
4. Commit: `git commit -m 'Add your feature'`
5. Push: `git push origin feature/your-feature`
6. Create Pull Request

### Development Guidelines

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: Follow ESLint rules
- **Commits**: Use conventional commit messages
- **Tests**: Add tests for new features
- **Documentation**: Update README for major changes

### Running Tests

```bash
# Backend
cd backend
pytest tests/ -v

# Frontend
cd frontend
npm test

# Docker
docker compose up --build
```

## 📄 License

This project is developed for academic purposes as part of an MLOps course. License TBD.

## 🙏 Acknowledgments

### Technologies
- **OpenAI** - Whisper API
- **Google** - Gemini Vision & Pro models
- **HuggingFace** - Wav2Vec2 model hosting
- **Supabase** - Database and authentication
- **FastAPI** - Backend framework
- **React** - Frontend framework
- **Anthropic Claude** - Development assistance

### Team
- Built by students for MLOps coursework
- Special thanks to course instructors and TAs

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/harshitashitut/MLOps-Project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/harshitashitut/MLOps-Project/discussions)
- **Documentation**: See `docs/` folder

---

**Built with ❤️ for better public speaking**

*Last Updated: December 2025*
