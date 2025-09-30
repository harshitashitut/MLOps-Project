# PitchQuest Multimodal: AI-Powered Pitch Analysis System

A comprehensive multimodal AI platform that analyzes pitch presentations across content, delivery, and visual performance dimensions. This system provides automated feedback comparable to professional pitch coaches through advanced machine learning and MLOps practices.

## Team Members

- Harshita Shitut
- Mohit Kakda
- Muhammad Salman
- Sachin Muttu Baraddi
- Uttapreksha Patel

## Project Overview

PitchQuest Multimodal transforms pitch practice by analyzing video presentations through three specialized lenses:

- **Content Analysis**: Evaluates problem-solution clarity, market sizing, business model completeness, and competitive differentiation through speech transcription and natural language processing
- **Delivery Analysis**: Assesses speaking pace, vocal confidence, filler word usage, emotional tone, and articulation through audio processing and emotion detection
- **Visual Analysis**: Examines body language, posture, eye contact, hand gestures, facial expressions, and pitch deck slide quality through computer vision and finetuned vision-language models

The platform guides users through a structured five-phase workflow: mentor preparation, investor simulation with video recording, multimodal analysis, post-session mentor discussion, and progress tracking through a personal dashboard.

**Note**: This project is under active development. Architecture, file structure, and implementation details are subject to change as we refine the system based on testing and performance evaluation.

## Technical Architecture

**Note:** File structure and architecture details are tentative and subject to modification based on testing and performance requirements.

### Core Components

**Frontend**: Next.js web application with video upload, real-time analysis display, and dashboard visualization

**Backend**: FastAPI service handling video processing, model orchestration, and agent coordination

**ML Pipeline**: Multi-model inference system combining transcription, emotion detection, pose estimation, facial analysis, and vision-language understanding

**MLOps**: Automated training pipeline, model registry, drift monitoring, and A/B testing framework

**Database**: Supabase (PostgreSQL) for session data, analysis results, user feedback, and training metrics



## Key Features

### Multimodal Analysis Pipeline
- Parallel processing of audio, video frames, and slide images
- Integration of multiple pre-trained models (Whisper, Wav2Vec2, MediaPipe, DeepFace, SmolVLM)
- Finetuned SmolVLM-500M for pitch deck specific evaluation
- Sub-2-minute processing latency for 10-minute pitch videos


### Models

**Pre-trained Models** (used as-is):
- Whisper API: Speech-to-text transcription
- Wav2Vec2: Speech emotion recognition
- MediaPipe Pose: Body pose estimation
- DeepFace: Facial emotion detection
- SmolVLM-500M (base): Body language description

**Finetuned Models** (trained in this project):
- SmolVLM-500M: Pitch deck slide analysis (trained on 500-1,000 labeled slides)

## Repository Structure

```
pitchquest-multimodal/
├── src/
│   └── Main project code (API, ML models, agents)
├── data/
│   └── Datasets and data collection scripts
├── notebooks/
│   └── Exploratory analysis and experiments
├── tests/
│   └── Unit and integration tests
├── requirements.txt
└── README.md
```

**Detailed structure will be finalized during development and may include additional directories for MLOps infrastructure, deployment configurations, and documentation.**



## Model Training

### Finetuning SmolVLM for Slide Analysis

```bash
# Prepare dataset
python scripts/scrape_pitch_decks.py
python scripts/label_slides.py

# Run training
python mlops/training/finetune_smolvlm.py

# Validate model
python mlops/training/validate_model.py
```

Training uses LoRA (Low-Rank Adaptation) and takes approximately 2-4 hours on a T4 GPU (Google Colab free tier).

## MLOps Pipeline

### Automated Retraining

The system automatically retrains models when:
- 100 user feedback corrections accumulated
- Weekly scheduled runs (configurable)
- Manual trigger via GitHub Actions

```bash
# Check if retraining needed
python scripts/check_retraining_trigger.py

# Trigger manual retraining
gh workflow run retrain.yml
```

### Monitoring

Access monitoring dashboards:
- **Model Performance**: Weights & Biases project dashboard
- **System Metrics**: Grafana dashboard (requires setup)
- **Drift Detection**: Evidently AI reports


## Dataset Information

### Pitch Deck Slide Dataset
- **Size**: 500-1,000 labeled slides
- **Sources**: Y Combinator Demo Day, SlideShare, Sequoia Capital examples, team-created samples
- **Labels**: Quality scores for clarity, design, data visualization, readability
- **Format**: JPEG images (1920x1080) with JSON metadata

### User Feedback Dataset
- Continuously collected from production usage
- Triggers automated retraining pipeline
- Stored in Supabase with full audit trail

## Performance Targets

- **Processing Latency**: < 2 minutes per 10-minute video
- **Model Accuracy**: > 85% agreement with human evaluators
- **System Uptime**: > 99%
- **Cost per Analysis**: < $0.30
- **User Satisfaction**: > 4.0/5.0

## License

To be determined. This is an academic project developed for educational purposes.

## Acknowledgments

- Anthropic Claude for AI assistance
- HuggingFace for model hosting and transformers library
- OpenAI for Whisper API
- Google for MediaPipe
- Weights & Biases for experiment tracking
- Course instructors and teaching assistants

## Repository Link

[GitHub Repository](https://github.com/harshitashitut/MLOps-Project.git)

## Contact

For questions or issues, please contact team members through university email or create an issue in the GitHub repository.

---


