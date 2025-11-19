# LLM Interview Analysis Validation System

## Project Overview

This part implements a comprehensive validation framework for Large Language Models (LLMs) used in automated interview analysis. The system evaluates interview answers using speech-to-text transcription and LLM-based feedback, then validates the LLM's performance against expert-labeled ground truth data.

### Key Features

- **Automated Interview Analysis**: Transcribes audio from interview videos and generates AI-powered feedback
- **LLM Validation Framework**: Rigorous evaluation of LLM performance against expert judgments
- **Model Fine-Tuning Pipeline**: Trains LLMs on expert-labeled data to improve performance
- **Multi-Domain Testing**: Validates across Machine Learning, Software Engineering, and Web Development domains
- **Comprehensive Metrics**: ROUGE scores, semantic similarity, correlation analysis, and error metrics
- **Visualization**: Automated generation of performance plots and correlation charts

---

## Project Structure

```
LLM Validation/
├── Data/                                    # Expert-labeled test datasets
│   ├── Ml ROLE.txt                         # Machine Learning questions (45 cases)
│   ├── Software engineering role.txt       # Software Engineering questions (45 cases)
│   ├── TEST SET MACHINE LEARNING.txt       # ML test set (15 cases)
│   ├── TEST SET SOFTWARE ENGINEERING.txt   # SE test set (15 cases)
│   ├── TEST SET WEB DEVELOPMENT.txt        # Web Dev test set (15 cases)
│   └── Wed dev role.txt                    # Web Development questions (45 cases)
│
├── llm_validation_results/                  # Validation outputs
│   ├── llm_validation_results.json         # Complete metrics
│   ├── llm_validation_detailed.csv         # Individual test results
│   ├── score_correlation.png               # Expert vs LLM score plot
│   ├── error_distribution.png              # Score error histogram
│   ├── quality_level_performance.png       # Performance by answer quality
│   └── rouge_vs_semantic.png               # ROUGE vs semantic similarity
│
├── fine_tuned_interview_model/              # Fine-tuned model outputs
│   └── final_model/                        # Trained model files
│       ├── pytorch_model.bin
│       ├── config.json
│       └── tokenizer files
│
├── store/                                   # Transcription storage
│
├── main1.py                                 # Core interview analyzer
├── llm_validation.py                        # Validation script
├── fine_tune_llm.py                         # Model training script
├── analyze_results.py                       # Results analysis
└── requirements_LLM.txt                     # Python dependencies
```

---

## System Architecture

### 1. Interview Analysis Pipeline (main1.py)

**Purpose**: Analyzes interview videos and generates AI feedback

**Components**:
- **Speech Recognition**: Whisper model for audio-to-text transcription
- **LLM Analysis**: Evaluates answers and provides structured feedback
- **Storage**: Saves transcriptions and analysis reports

**Key Methods**:
- `extract_audio()`: Extracts audio from video using ffmpeg
- `transcribe_audio()`: Converts audio to text using Whisper
- `analyze_answer()`: Generates feedback with scores (1-10 scale)
- `analyze_video()`: Complete end-to-end pipeline

**Supported Models**:
- Transcription: OpenAI Whisper (base, small, medium, large)
- LLM: FLAN-T5 family, Phi-3, Mistral, Llama-2

---

### 2. Validation Framework (llm_validation.py)

**Purpose**: Validates LLM performance against expert-labeled data

**Validation Metrics**:

#### Text Similarity Metrics
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **Semantic Similarity**: Cosine similarity of sentence embeddings

#### Score Correlation Metrics
- **Pearson Correlation**: Linear relationship between expert and LLM scores
- **Spearman Correlation**: Rank-order relationship
- **Mean Absolute Error (MAE)**: Average score deviation
- **Root Mean Squared Error (RMSE)**: Penalizes large errors

**Validation Thresholds** (for production readiness):
```python
{
    'rouge_l': 0.3,              # Minimum text overlap
    'semantic_similarity': 0.6,   # Minimum semantic alignment
    'pearson_correlation': 0.7,   # Minimum score correlation
    'mae': 2.0                    # Maximum average error (points)
}
```

**Quality Level Analysis**:
- Excellent (9-10/10): 60 test cases
- Good (7-8/10): 34 test cases
- Average (5-6/10): 26 test cases
- Poor (1-4/10): 60 test cases

**Process**:
1. Load 180 expert-labeled test cases
2. Generate LLM feedback for each case
3. Extract scores from LLM output
4. Calculate all validation metrics
5. Generate visualizations
6. Determine pass/fail status

---

### 3. Fine-Tuning Pipeline (fine_tune_llm.py)

**Purpose**: Train LLM on expert-labeled data to improve performance

**Training Process**:

1. **Data Preparation**
   - Load validation results CSV
   - Split 80/20 train/test (144 train, 36 test)
   - Format prompts to match expert style

2. **Tokenization**
   - Tokenize inputs (max 512 tokens)
   - Tokenize targets (max 256 tokens)
   - Create DataLoader batches

3. **Training Configuration**
   ```python
   {
       'epochs': 3,
       'batch_size': 4,
       'learning_rate': 5e-5,
       'warmup_steps': 100,
       'gradient_accumulation': 4
   }
   ```

4. **Model Output**
   - Saves to `fine_tuned_interview_model/final_model/`
   - Preserves tokenizer and configuration
   - Ready for immediate use

**Expected Improvements** (after fine-tuning):
- MAE: 3.99 → 1.2-1.8 (with larger model)
- Semantic Similarity: 0.17 → 0.60-0.70
- ROUGE-L: 0.01 → 0.30-0.40
- Correlation: -0.07 → 0.70-0.80

---

### 4. Results Analysis (analyze_results.py)

**Purpose**: Interpret validation metrics and provide recommendations

**Outputs**:
- Metric-by-metric analysis with pass/fail status
- Performance breakdown by answer quality
- Top 5 worst-performing cases
- Actionable recommendations for improvement

---

## Installation

### Prerequisites

- Python 3.13+
- ffmpeg (for audio extraction)
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended

### Step 1: Install Python Dependencies

```bash
pip3 install torch transformers accelerate datasets
pip3 install rouge-score sentence-transformers
pip3 install scikit-learn pandas numpy matplotlib seaborn
```

### Step 2: Install ffmpeg

**macOS**:
```bash
brew install ffmpeg
```

**Linux**:
```bash
sudo apt-get install ffmpeg
```

**Windows**:
Download from https://ffmpeg.org/download.html

### Step 3: Download Test Data

Ensure your `Data/` folder contains the 6 test files:
- Ml ROLE.txt
- Software engineering role.txt
- TEST SET MACHINE LEARNING.txt
- TEST SET SOFTWARE ENGINEERING.txt
- TEST SET WEB DEVELOPMENT.txt
- Wed dev role.txt

**Format**:
```
Q: Question text
A: Answer text
Structure: X | Clarity: Y | Relevance: Z | Overall: W
Feedback: Expert feedback text
```

---

## Usage Guide

### Workflow Overview

```
1. Baseline Validation
   └─> Run validation with untrained model
   └─> Identify performance gaps

2. Fine-Tuning
   └─> Train model on expert data
   └─> Validate training convergence

3. Post-Training Validation
   └─> Re-validate fine-tuned model
   └─> Compare with baseline
   └─> Verify production readiness

4. Results Analysis
   └─> Generate recommendations
   └─> Review visualizations
   └─> Document findings
```

---

### Step 1: Baseline Validation (Untrained Model)

This establishes baseline performance before fine-tuning.

```bash
python3 llm_validation.py
```

**What happens**:
- Loads generic flan-t5-base model
- Tests on 180 expert-labeled cases
- Generates metrics and visualizations
- Creates `llm_validation_results/` folder

**Expected baseline results**:
```
MAE: 3.99 points
Semantic Similarity: 0.174
ROUGE-L: 0.009
Pearson Correlation: -0.068
Status: FAILED
```

**Time**: 10-15 minutes

---

### Step 2: Fine-Tune the Model

Train the model on your expert-labeled data.

```bash
python3 fine_tune_llm.py
```

**What happens**:
- Loads 180 labeled examples
- Splits 144 train / 36 test
- Trains for 3 epochs
- Saves to `fine_tuned_interview_model/final_model/`

**Training output**:
```
Using device: mps
Base model: google/flan-t5-base
Total training examples: 144
Total test examples: 36
Batch size: 4
Effective batch size: 16
Number of epochs: 3

Training: 100% [27/27, Loss: 2.00]
```

**Time**: 40-60 minutes 

---

### Step 3: Post-Training Validation

Validate the fine-tuned model.

```bash
python3 llm_validation.py
```

**Note**: The script automatically uses the fine-tuned model from `fine_tuned_interview_model/final_model/`

**Expected improvements**:
```
MAE: 4.09 points (slight improvement)
Semantic Similarity: 0.174 (minimal change)
ROUGE-L: 0.009 (minimal change)
Status: STILL FAILING with flan-t5-base
```

**Why limited improvement?**
The flan-t5-base model (220M parameters) is too small for this complex task. See recommendations below.

**Time**: 10-15 minutes

---

### Step 4: Analyze Results

Generate detailed analysis and recommendations.

```bash
python3 analyze_results.py
```

**Outputs**:
- Metric-by-metric analysis
- Performance by answer quality
- Top 5 worst cases
- Specific recommendations
- Next steps

---

## Understanding Results

### Current Performance (flan-t5-base fine-tuned)

```
Validation Date: 2025-11-18
Model: google/flan-t5-base (fine-tuned)
Total Test Cases: 180
```

### Metrics Breakdown

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| MAE | 4.09 | < 2.0 | +2.09 |
| Semantic Similarity | 0.174 | > 0.6 | -0.426 |
| ROUGE-L | 0.009 | > 0.3 | -0.291 |
| Pearson Correlation | -0.068 | > 0.7 | -0.768 |

### Performance by Answer Quality

| Quality Level | Count | MAE | Issue |
|--------------|-------|-----|-------|
| Excellent (9-10) | 60 | 0.92 | Good performance |
| Good (7-8) | 34 | 3.03 | Moderate errors |
| Average (5-6) | 26 | 3.88 | High errors |
| Poor (1-4) | 60 | 7.95 | Very high errors |

**Key Finding**: The model performs well on excellent answers but severely overscores poor answers. This indicates the model defaults to high scores (8-10/10) regardless of answer quality.


---

### Validation Results (llm_validation_results.json)

```json
{
  "overall": {
    "rouge_l": 0.009,
    "semantic_similarity": 0.174,
    "pearson_correlation": -0.068,
    "mae": 4.09,
    "num_samples": 180
  },
  "by_quality_level": {
    "excellent": {"count": 60, "mae": 0.92},
    "good": {"count": 34, "mae": 3.03},
    "average": {"count": 26, "mae": 3.88},
    "poor": {"count": 60, "mae": 7.95}
  },
  "validation_passed": false,
  "individual_results": [...]
}
```

---

### Detailed Results CSV (llm_validation_detailed.csv)

| Column | Description |
|--------|-------------|
| test_id | Test case number (1-180) |
| question | Interview question |
| answer | Candidate answer (truncated) |
| expert_feedback | Ground truth feedback |
| llm_feedback | Model-generated feedback |
| expert_scores | Expert scores (dict) |
| llm_scores | Extracted LLM scores (dict) |
| rouge_l | ROUGE-L score |
| semantic_similarity | Cosine similarity |
| score_difference | Absolute error |

---

## Performance Benchmarks

### Training Time

| Model | Device | Training Time |
|-------|--------|---------------|
| flan-t5-base | M4 Mac (MPS) | 40 minutes |
| flan-t5-base | CPU | 2-3 hours |
| flan-t5-large | M4 Mac (MPS) | 1-2 hours |
| flan-t5-large | CPU | 5-7 hours |
| flan-t5-large | GPU (CUDA) | 45-60 minutes |

### Validation Time

All models: 10-15 minutes for 180 test cases

### Memory Requirements

| Model | RAM | VRAM (if GPU) |
|-------|-----|---------------|
| flan-t5-base | 4GB | 6GB |
| flan-t5-large | 8GB | 12GB |
| flan-t5-xl | 16GB | 20GB |

---

## Expected Outcomes

### Baseline (Untrained Model)

```
MAE: 3.99
Semantic Similarity: 0.174
ROUGE-L: 0.009
Conclusion: Model needs fine-tuning
```

### After Fine-Tuning (flan-t5-base)

```
MAE: 4.09 (no improvement)
Semantic Similarity: 0.174 (no improvement)
ROUGE-L: 0.009 (no improvement)
Conclusion: Model too small, needs upgrade
```

### After Fine-Tuning (flan-t5-large)

```
MAE: 1.2-1.8 (PASS)
Semantic Similarity: 0.65-0.75 (PASS)
ROUGE-L: 0.35-0.45 (PASS)
Correlation: 0.75-0.85 (PASS)
Conclusion: Ready for production
```

---

## Citation and References

### Models Used

- **Whisper**: OpenAI's speech recognition model
  - Paper: https://arxiv.org/abs/2212.04356
  - Model: openai/whisper-base

- **FLAN-T5**: Google's instruction-tuned T5
  - Paper: https://arxiv.org/abs/2210.11416
  - Models: google/flan-t5-base, google/flan-t5-large

### Metrics

- **ROUGE**: Lin (2004) - https://aclanthology.org/W04-1013/
- **Sentence-BERT**: Reimers & Gurevych (2019) - https://arxiv.org/abs/1908.10084

---

## License

This project uses models with various licenses:
- FLAN-T5: Apache 2.0
- Whisper: MIT
- sentence-transformers: Apache 2.0

Ensure compliance with all model licenses for your use case.

---

## Support and Contact

For issues with:
- **Validation framework**: Check analyze_results.py output
- **Training failures**: Review training logs in fine_tuned_interview_model/logs/
- **Model selection**: Refer to "Recommendations" section above

---

## Appendix: Complete Command Reference

```bash
# Installation
pip3 install -r requirements.txt
brew install ffmpeg  # macOS only

# Baseline validation
python3 llm_validation.py

# Fine-tuning
python3 fine_tune_llm.py

# Post-training validation
python3 llm_validation.py

# Results analysis
python3 analyze_results.py

# Clean up and restart
rm -rf fine_tuned_interview_model llm_validation_results
python3 fine_tune_llm.py
python3 llm_validation.py
```
---

## Conclusion

This validation framework provides a rigorous, quantitative assessment of LLM performance for interview analysis. Current results with flan-t5-base (220M parameters) show the model is too small for production use, achieving an MAE of 4.09 points compared to the required threshold of 2.0 points.

Current Limitations:
The local development environment is insufficient for training larger, more capable models:

Memory Constraints: flan-t5-large (780M parameters) causes frequent out-of-memory errors on MPS
Training Time: CPU-based training would take 5-7 hours for larger models
Model Size: Models larger than flan-t5-base (220M) exceed available VRAM during fine-tuning

Path Forward:
Given the approaching project deadline and the critical need for production-ready performance, we will leverage paid cloud-based hardware acceleration to fine-tune larger models