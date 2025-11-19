# MLOps Multi-Modal Analysis Pipeline

## Project Overview

A comprehensive MLOps system that analyzes professional presentations through three AI models, providing objective feedback on speech, content quality, and body language. Built for interview coaching and presentation training with bias detection and performance validation.

## Repository Structure

```
MLOps-Multi-Modal-Analysis/
├── README.md                     # This overview file
├── requirements.txt              # Master dependencies
├── Whisper_Pipeline/            
│   ├── README.md                # Detailed Whisper documentation
│   ├── whisper_validation.py
│   └── data/                    # 23 test videos + transcripts
├── LLM_Pipeline/               
│   ├── README.md                # Detailed LLM documentation
│   ├── llm_validation.py
│   ├── fine_tune_llm.py
│   └── Data/                    # 180 expert-labeled cases
└── Model_Pipeline/             
    ├── README.md                # Detailed MediaPipe documentation
    ├── body_language_analyzer.py
    ├── bias_testing_script.py
    └── data/                    # 9 demographic test videos
```

## System Architecture

```
Input Video/Audio → Multi-Modal Pipeline → Integrated Analysis Report
                           ↓
    ┌─────────────┬─────────────┬─────────────┐
    │   Whisper   │     LLM     │  MediaPipe  │
    │   Pipeline  │  Pipeline   │  Pipeline   │
    └─────────────┴─────────────┴─────────────┘
```

## Model Pipelines

### Whisper Speech Analysis
**Documentation**: [Whisper Details](./Model_Pipeline/whisper_analysis/readme.md)
- **Technology**: OpenAI Whisper ASR
- **Function**: Audio transcription and speech quality assessment
- **Performance**: 73.03% transcription accuracy (WER: 26.97%)
- **Dataset**: 23 videos with ground truth transcripts
- **Status**: Production ready with quality guidelines

### LLM Content Analysis
**Documentation**: [Model_Pipeline/LLM VALIDATION/README.md](./Model_Pipeline/LLM%20VALIDATION/README.md)
- **Technology**: FLAN-T5 (base/large) with fine-tuning
- **Function**: Interview answer evaluation and structured feedback
- **Performance**: 4.09 MAE (score error), requires model upgrade
- **Dataset**: 180 expert-labeled interview cases across ML/SE/Web domains
- **Status**: Needs cloud-based training for production deployment

### MediaPipe Body Language Analysis
**Documentation**: [MediaPipe Details](./Model_Pipeline/mediapipe_bias_detection/README.md)
- **Technology**: MediaPipe Pose Detection (33 landmarks)
- **Function**: Posture, gesture, and non-verbal communication analysis
- **Performance**: 91.1% posture accuracy, 100% detection success
- **Dataset**: 9 videos with demographic diversity testing
- **Status**: Functional with significant bias concerns requiring mitigation

## Performance Summary

| Pipeline | Accuracy | Key Strength | Main Challenge |
|----------|----------|--------------|----------------|
| Whisper | 73.03% | Reliable transcription | Audio quality sensitivity |
| LLM | 8.1/10 content score | Expert-level feedback | Model size limitations |
| MediaPipe | 91.1% posture detection | Fast processing | Demographic bias |

## Critical Findings

### Bias Detection Results
- **MediaPipe VLM**: 46.8% performance gap across ethnicities
- **Whisper**: No significant demographic bias detected
- **LLM**: Insufficient demographic data for bias assessment

### Production Readiness Assessment
- **Whisper**: Ready for deployment with quality guidelines
- **LLM**: Requires larger model (cloud training needed)
- **MediaPipe**: Usable with bias mitigation strategies

## Quick Start

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Run Individual Pipelines
```bash
# Speech analysis
cd Whisper_Pipeline && python whisper_validation.py

# Content analysis  
cd LLM_Pipeline && python llm_validation.py

# Body language analysis
cd Model_Pipeline && python body_language_analyzer.py
```

## Key Contributions

### Technical Achievements
- **Multi-Modal Integration**: Unified pipeline processing speech, text, and vision
- **Bias Detection Framework**: Systematic evaluation across demographic groups
- **MLOps Best Practices**: Validation, fine-tuning, and monitoring pipelines

### Research Insights
- **AI Bias Documentation**: Quantified performance gaps in computer vision models
- **Model Validation Framework**: Rigorous testing against expert-labeled ground truth
- **Hardware Constraints Analysis**: Consumer vs cloud deployment trade-offs

## Detailed Documentation Navigation

| Component | Documentation Path |
|-----------|-------------------|
| **Speech Recognition Setup** | [Whisper_Pipeline/README.md](./Whisper_Pipeline/README.md) |
| **Content Analysis Training** | [LLM_Pipeline/README.md](./LLM_Pipeline/README.md) |
| **Body Language Analysis & Bias Testing** | [Model_Pipeline/README.md](./Model_Pipeline/README.md) |

## Implementation Notes

### Bias Mitigation Recommendations
1. **MediaPipe**: Display confidence scores, implement quality thresholds
2. **Multi-Modal Validation**: Cross-reference results across pipelines  
3. **User Transparency**: Inform users of potential bias in visual analysis

### Hardware Requirements
- **Development Environment**: 8GB RAM, consumer laptop sufficient
- **Production LLM Deployment**: Cloud GPU required for larger models
- **MediaPipe Processing**: CPU sufficient, GPU optional for performance

## Contributing

1. Fork repository and create feature branch
2. Run bias testing on any new models
3. Update performance benchmarks in relevant README
4. Submit pull request with validation results

## License

MIT License - Educational and Research Use

---

**Documentation Links**: [Whisper Details](./Model_Pipeline/whisper_analysis/readme.md) | [LLM Details](./Model_Pipeline/LLM VALIDATION/Readme.md) | [MediaPipe Details](./Model_Pipeline/mediapipe_bias_detection/README.md)
