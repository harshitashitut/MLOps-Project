# Model Bias Detection - MediaPipe Analysis

## Overview
This folder contains bias detection analysis for MediaPipe Pose model used in the PitchQuest project. We test whether the pose detection model performs equally across different demographic groups.

## Purpose
As part of our MLOps project requirements, we evaluate potential biases in pre-trained models before deployment. MediaPipe is used for body language analysis in pitch videos, so we must ensure it works fairly across diverse founder demographics.

## Folder Structure
```
Project_ml_model_bias/
├── README.md                      # This file
├── body_language_analyzer.py      # MediaPipe wrapper for posture analysis
├── bias_testing_script.py         # Automated bias testing pipeline
├── video_labels.csv               # Manual demographic labels for test videos
├── bias_test_results.csv          # Analysis results with confidence scores
└── data/                          # Test videos (not committed to repo)
    ├── vid1.mp4
    ├── vid2.mp4
    └── ...
```

## Setup

### Requirements
```bash
pip install opencv-python mediapipe numpy pandas
```

### Test Data
- 9 test videos downloaded from Pexels (free stock videos)
- Manually labeled by demographics: skin tone, gender, age group, lighting
- Videos represent diverse presenters to test model fairness

## How to Run

1. **Prepare test videos:**
   - Download videos and place in `data/` folder
   - Create `video_labels.csv` with demographic labels

2. **Run bias test:**
```bash
   python3 bias_testing_script.py
```

3. **Review results:**
   - Check console output for bias analysis
   - Open `bias_test_results.csv` for detailed metrics

## Metrics Measured

For each video, we extract:
- **Detection Confidence:** How confident MediaPipe is in detecting the person (0-1)
- **Detection Rate:** % of frames where pose was detected
- **Keypoints Detected:** Number of body landmarks found (out of 33)
- **Posture Score:** Quality of posture (0-1)

## Results Summary

### Key Findings
| Skin Tone | Avg Confidence | Avg Keypoints | Detection Rate |
|-----------|----------------|---------------|----------------|
| White (6) | 0.70           | 24.0          | 100%           |
| Brown (2) | 0.57           | 17.6          | 93%            |
| Black (1) | 0.47           | 14.0          | 71%            |

### Bias Detected ⚠️
- **33% lower confidence** on darker skin tones vs lighter skin tones
- **42% fewer keypoints detected** on black skin (14 vs 25)
- **29% lower detection rate** on black skin (71% vs 100%)

### Root Cause Analysis
- MediaPipe likely trained on dataset with over-representation of lighter skin tones
- Lower contrast between darker skin and backgrounds affects detection
- Lighting conditions compound the issue

## Mitigation Strategies

### Implemented
1. **Confidence Thresholds:** Flag low-confidence predictions (<0.6) for manual review
2. **Weighted Scoring:** Reduce importance of visual analysis when confidence is low
3. **User Warnings:** Notify users when lighting/video quality may affect results

### Recommended (Future Work)
1. Fine-tune MediaPipe on diverse dataset if possible
2. Add pre-processing (lighting normalization)
3. Provide alternative: allow users to skip visual analysis
4. Collect diverse training data for custom pose model

## Limitations

- **Small sample size:** Only 9 videos (1 black, 2 brown, 6 white)
- **Confounding variables:** Can't fully isolate skin tone from lighting/video quality
- **No ground truth:** Comparing relative performance, not absolute accuracy
- **Single model:** Only tested MediaPipe, not alternative pose detection models

**Note:** This is preliminary analysis for course project. Production deployment would require larger, more controlled study (30+ videos per group).

## Integration with PitchQuest

This bias detection informs our deployment decisions:
- ✅ Use MediaPipe for body language BUT with caveats
- ✅ Display confidence scores to users
- ✅ Weight content analysis higher than delivery for low-confidence cases
- ❌ Consider removing visual analysis entirely if bias cannot be mitigated

## References

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [Fairlearn - Bias Detection Tools](https://fairlearn.org/)
- [Gender Shades: Facial Recognition Bias Study](http://gendershades.org/)

## Authors
Team PitchQuest - Fall 2025 MLOps Project

---
*Last Updated: November 2025*