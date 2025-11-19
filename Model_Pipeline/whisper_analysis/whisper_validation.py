# """
# Whisper Transcription Validation Script
# Validates speech-to-text accuracy and detects bias across demographic groups
# """

# import sys
# sys.path.append("/home/mohit/.local/lib/python3.13/site-packages")

# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Tuple
# from jiwer import wer, cer, mer, wil

# # Import your analyzer
# from main import InterviewAnalyzer


# class WhisperValidator:
#     def __init__(self, output_dir="whisper_validation_results"):
#         """
#         Initialize Whisper validator
        
#         Args:
#             output_dir: Directory to save validation results
#         """
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(exist_ok=True)
        
#         print("Initializing Whisper Validator...")
#         print(f"Results will be saved to: {self.output_dir.absolute()}\n")
    
#     def auto_detect_test_files(self, video_dir: str, transcript_dir: str) -> List[Dict]:
#         """
#         Automatically detect matching video and transcript files
        
#         Args:
#             video_dir: Directory containing video files (e.g., "data/video")
#             transcript_dir: Directory containing transcript files (e.g., "data/transcript")
            
#         Returns:
#             List of test cases with matched files
#         """
#         video_dir = Path(video_dir)
#         transcript_dir = Path(transcript_dir)
        
#         print(f"Scanning for test files...")
#         print(f"  Video directory: {video_dir}")
#         print(f"  Transcript directory: {transcript_dir}")
        
#         test_cases = []
        
#         # Find all video files
#         video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv']
#         video_files = []
#         for ext in video_extensions:
#             video_files.extend(video_dir.glob(f'*{ext}'))
        
#         print(f"\nFound {len(video_files)} video files")
        
#         for video_path in sorted(video_files):
#             video_id = video_path.stem  # filename without extension
            
#             # Look for matching transcript
#             transcript_path = transcript_dir / f"{video_id}.txt"
            
#             if not transcript_path.exists():
#                 print(f"  ⚠ Warning: No transcript found for {video_id}, skipping...")
#                 continue
            
#             # Add to test cases
#             test_case = {
#                 "video_id": video_id,
#                 "video_path": str(video_path),
#                 "transcript_path": str(transcript_path)
#             }
            
#             test_cases.append(test_case)
#             print(f"  ✓ Matched: {video_id}")
        
#         print(f"\nTotal matched pairs: {len(test_cases)}\n")
#         return test_cases
    
#     def add_demographics_interactive(self, test_cases: List[Dict]) -> List[Dict]:
#         """
#         Optionally add demographic info to test cases interactively
#         """
#         print("="*80)
#         print("DEMOGRAPHIC INFORMATION (Optional - for bias detection)")
#         print("="*80)
#         print("You can add demographic info for bias analysis.")
#         print("Press Enter to skip any field.\n")
        
#         add_demographics = input("Add demographic information? (y/n) [n]: ").lower() == 'y'
        
#         if not add_demographics:
#             # Add default values
#             for test_case in test_cases:
#                 test_case['speaker'] = test_case['video_id']
#                 test_case['demographics'] = {
#                     'gender': 'unknown',
#                     'age_group': 'unknown',
#                     'accent': 'unknown',
#                     'ethnicity': 'unknown'
#                 }
#                 test_case['audio_quality'] = 'unknown'
#                 test_case['speaking_speed'] = 'unknown'
#             return test_cases
        
#         print("\nEnter demographics for each video:")
#         print("(Press Enter to use defaults)\n")
        
#         for test_case in test_cases:
#             print(f"\n--- Video: {test_case['video_id']} ---")
            
#             test_case['speaker'] = input(f"  Speaker name [{test_case['video_id']}]: ") or test_case['video_id']
#             test_case['demographics'] = {
#                 'gender': input("  Gender (male/female/other) [unknown]: ") or "unknown",
#                 'age_group': input("  Age group (18-25/25-35/35-50/50+) [unknown]: ") or "unknown",
#                 'accent': input("  Accent (american/british/indian/etc) [unknown]: ") or "unknown",
#                 'ethnicity': input("  Ethnicity [unknown]: ") or "unknown"
#             }
#             test_case['audio_quality'] = input("  Audio quality (clear/moderate/poor) [unknown]: ") or "unknown"
#             test_case['speaking_speed'] = input("  Speaking speed (fast/normal/slow) [unknown]: ") or "unknown"
        
#         return test_cases
    
#     def validate_transcription(
#         self,
#         analyzer: InterviewAnalyzer,
#         test_cases: List[Dict]
#     ) -> Dict:
#         """
#         Validate Whisper transcription model
        
#         Args:
#             analyzer: InterviewAnalyzer with loaded transcription model
#             test_cases: List of test cases
            
#         Returns:
#             Dictionary with validation metrics and bias analysis
#         """
#         print("\n" + "="*80)
#         print("VALIDATING WHISPER TRANSCRIPTION MODEL")
#         print("="*80)
#         print(f"Total test cases: {len(test_cases)}\n")
        
#         results = []
#         wer_scores = []
#         cer_scores = []
#         mer_scores = []  # Match Error Rate
#         wil_scores = []  # Word Information Lost
        
#         # For bias detection - group by demographics
#         slices = {
#             'gender': {},
#             'age_group': {},
#             'accent': {},
#             'ethnicity': {},
#             'audio_quality': {},
#             'speaking_speed': {}
#         }
        
#         for idx, test_case in enumerate(test_cases, 1):
#             video_path = test_case['video_path']
#             transcript_path = test_case['transcript_path']
            
#             print(f"Processing {idx}/{len(test_cases)}: {test_case['video_id']}")
            
#             # Load ground truth
#             with open(transcript_path, 'r', encoding='utf-8') as f:
#                 ground_truth = f.read().strip()
            
#             if not ground_truth:
#                 print(f"  Warning: Empty transcript, skipping...")
#                 continue
            
#             # Extract audio and transcribe
#             try:
#                 audio_path = analyzer.extract_audio(video_path)
#                 prediction = analyzer.transcribe_audio(audio_path)
                
#                 # Clean up audio file
#                 if os.path.exists(audio_path):
#                     os.remove(audio_path)
                
#             except Exception as e:
#                 print(f"  Error processing video: {e}")
#                 continue
            
#             # Calculate metrics
#             try:
#                 wer_score = wer(ground_truth, prediction)
#                 cer_score = cer(ground_truth, prediction)
#                 mer_score = mer(ground_truth, prediction)
#                 wil_score = wil(ground_truth, prediction)
#             except Exception as e:
#                 print(f"  Error calculating metrics: {e}")
#                 continue
            
#             wer_scores.append(wer_score)
#             cer_scores.append(cer_score)
#             mer_scores.append(mer_score)
#             wil_scores.append(wil_score)
            
#             print(f"  WER: {wer_score:.2%} | CER: {cer_score:.2%} | Accuracy: {(1-wer_score):.2%}")
            
#             # Store result
#             result = {
#                 'video_id': test_case['video_id'],
#                 'speaker': test_case.get('speaker', test_case['video_id']),
#                 'ground_truth': ground_truth,
#                 'prediction': prediction,
#                 'ground_truth_length': len(ground_truth),
#                 'prediction_length': len(prediction),
#                 'word_count': len(ground_truth.split()),
#                 'wer': wer_score,
#                 'cer': cer_score,
#                 'mer': mer_score,
#                 'wil': wil_score,
#                 'accuracy': 1 - wer_score,
#                 'demographics': test_case.get('demographics', {}),
#                 'audio_quality': test_case.get('audio_quality', 'unknown'),
#                 'speaking_speed': test_case.get('speaking_speed', 'unknown')
#             }
#             results.append(result)
            
#             # Collect metrics by slice for bias detection
#             demographics = test_case.get('demographics', {})
            
#             for slice_key in ['gender', 'age_group', 'accent', 'ethnicity']:
#                 slice_value = demographics.get(slice_key, 'unknown')
#                 if slice_value != 'unknown':
#                     if slice_value not in slices[slice_key]:
#                         slices[slice_key][slice_value] = []
#                     slices[slice_key][slice_value].append(wer_score)
            
#             # Collect by audio quality and speaking speed
#             audio_quality = test_case.get('audio_quality', 'unknown')
#             if audio_quality != 'unknown':
#                 if audio_quality not in slices['audio_quality']:
#                     slices['audio_quality'][audio_quality] = []
#                 slices['audio_quality'][audio_quality].append(wer_score)
            
#             speaking_speed = test_case.get('speaking_speed', 'unknown')
#             if speaking_speed != 'unknown':
#                 if speaking_speed not in slices['speaking_speed']:
#                     slices['speaking_speed'][speaking_speed] = []
#                 slices['speaking_speed'][speaking_speed].append(wer_score)
        
#         if len(results) == 0:
#             print("Error: No results to analyze!")
#             return None
        
#         # Calculate overall metrics
#         print("\n" + "="*80)
#         print("OVERALL METRICS")
#         print("="*80)
        
#         avg_wer = np.mean(wer_scores)
#         avg_cer = np.mean(cer_scores)
#         avg_mer = np.mean(mer_scores)
#         avg_wil = np.mean(wil_scores)
#         avg_accuracy = 1 - avg_wer
        
#         std_wer = np.std(wer_scores)
#         median_wer = np.median(wer_scores)
        
#         print(f"\nTranscription Accuracy Metrics:")
#         print(f"  Average WER (Word Error Rate): {avg_wer:.2%}")
#         print(f"  Average CER (Character Error Rate): {avg_cer:.2%}")
#         print(f"  Average MER (Match Error Rate): {avg_mer:.2%}")
#         print(f"  Average WIL (Word Info Lost): {avg_wil:.2%}")
#         print(f"  Average Accuracy: {avg_accuracy:.2%}")
#         print(f"  Median WER: {median_wer:.2%}")
#         print(f"  Std Dev WER: {std_wer:.2%}")
        
#         # Calculate metrics by slice (bias detection)
#         print(f"\n" + "="*80)
#         print("BIAS DETECTION - PERFORMANCE BY DEMOGRAPHIC SLICE")
#         print("="*80)
        
#         slice_metrics = {}
#         bias_detected = {}
        
#         for slice_name, slice_groups in slices.items():
#             if len(slice_groups) == 0 or all(len(v) == 0 for v in slice_groups.values()):
#                 continue
            
#             print(f"\n{slice_name.upper().replace('_', ' ')}:")
#             slice_metrics[slice_name] = {}
            
#             for group, scores in slice_groups.items():
#                 if len(scores) > 0:
#                     avg_score = np.mean(scores)
#                     slice_metrics[slice_name][group] = {
#                         'avg_wer': avg_score,
#                         'avg_accuracy': 1 - avg_score,
#                         'count': len(scores),
#                         'std_dev': np.std(scores)
#                     }
#                     print(f"  {group}: WER={avg_score:.2%}, Accuracy={1-avg_score:.2%} (n={len(scores)})")
            
#             # Check for bias (>10% difference between groups)
#             if len(slice_metrics[slice_name]) > 1:
#                 wer_values = [m['avg_wer'] for m in slice_metrics[slice_name].values()]
#                 max_wer = max(wer_values)
#                 min_wer = min(wer_values)
#                 difference = max_wer - min_wer
                
#                 if difference > 0.10:  # 10% threshold
#                     bias_detected[slice_name] = {
#                         'difference': difference,
#                         'max_wer': max_wer,
#                         'min_wer': min_wer,
#                         'severity': 'HIGH' if difference > 0.20 else 'MODERATE'
#                     }
#                     print(f"  ⚠ BIAS DETECTED: {difference:.2%} difference ({bias_detected[slice_name]['severity']})")
        
#         # Determine validation pass/fail
#         validation_passed = avg_wer < 0.25  # 25% WER threshold
        
#         print(f"\n" + "="*80)
#         print(f"VALIDATION: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
#         print(f"  Threshold: WER < 25%")
#         print(f"  Your WER: {avg_wer:.2%}")
#         if bias_detected:
#             print(f"  ⚠ Bias detected in {len(bias_detected)} demographic slice(s)")
#         print("="*80)
        
#         # Compile results
#         validation_results = {
#             'overall': {
#                 'average_wer': avg_wer,
#                 'average_cer': avg_cer,
#                 'average_mer': avg_mer,
#                 'average_wil': avg_wil,
#                 'average_accuracy': avg_accuracy,
#                 'median_wer': median_wer,
#                 'std_dev_wer': std_wer,
#                 'num_samples': len(results)
#             },
#             'by_slice': slice_metrics,
#             'bias_detected': bias_detected,
#             'individual_results': results,
#             'validation_passed': validation_passed,
#             'validation_threshold': 0.25,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         # Generate visualizations
#         print("\nGenerating visualizations...")
#         self._plot_wer_distribution(wer_scores)
#         self._plot_slice_comparison(slice_metrics)
#         self._plot_accuracy_by_video(results)
        
#         if len(wer_scores) > 5:
#             self._plot_error_heatmap(results)
        
#         # Save results
#         output_file = self.output_dir / "whisper_validation_results.json"
#         with open(output_file, 'w') as f:
#             json.dump(validation_results, f, indent=2)
        
#         # Save detailed CSV
#         df = pd.DataFrame(results)
#         csv_file = self.output_dir / "whisper_validation_detailed.csv"
#         df.to_csv(csv_file, index=False)
        
#         print(f"\n✓ Results saved to: {output_file}")
#         print(f"✓ Detailed CSV saved to: {csv_file}")
#         print(f"✓ Visualizations saved to: {self.output_dir}\n")
        
#         return validation_results
    
#     def _plot_wer_distribution(self, wer_scores):
#         """Plot WER distribution histogram"""
#         plt.figure(figsize=(10, 6))
#         plt.hist(wer_scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
#         plt.axvline(np.mean(wer_scores), color='red', linestyle='--', 
#                    linewidth=2, label=f'Mean: {np.mean(wer_scores):.2%}')
#         plt.axvline(np.median(wer_scores), color='green', linestyle='--', 
#                    linewidth=2, label=f'Median: {np.median(wer_scores):.2%}')
#         plt.xlabel('Word Error Rate (WER)', fontsize=12)
#         plt.ylabel('Frequency', fontsize=12)
#         plt.title('Transcription Error Distribution', fontsize=14, fontweight='bold')
#         plt.legend(fontsize=10)
#         plt.grid(True, alpha=0.3, axis='y')
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'wer_distribution.png', dpi=300, bbox_inches='tight')
#         plt.close()
    
#     def _plot_slice_comparison(self, slice_metrics):
#         """Plot comparison across demographic slices"""
#         for slice_name, groups in slice_metrics.items():
#             if len(groups) == 0:
#                 continue
            
#             group_names = list(groups.keys())
#             wer_values = [groups[g]['avg_wer'] for g in group_names]
#             counts = [groups[g]['count'] for g in group_names]
            
#             fig, ax = plt.subplots(figsize=(10, 6))
#             bars = ax.bar(group_names, wer_values, alpha=0.7, color='coral')
            
#             # Add count labels on bars
#             for bar, count in zip(bars, counts):
#                 height = bar.get_height()
#                 ax.text(bar.get_x() + bar.get_width()/2., height,
#                        f'n={count}',
#                        ha='center', va='bottom', fontsize=9)
            
#             ax.set_xlabel(slice_name.replace('_', ' ').title(), fontsize=12)
#             ax.set_ylabel('Average WER', fontsize=12)
#             ax.set_title(f'WER by {slice_name.replace("_", " ").title()}', 
#                         fontsize=14, fontweight='bold')
#             ax.set_ylim(0, max(wer_values) * 1.2)
#             plt.xticks(rotation=45, ha='right')
#             plt.grid(True, alpha=0.3, axis='y')
#             plt.tight_layout()
#             plt.savefig(self.output_dir / f'wer_by_{slice_name}.png', dpi=300, bbox_inches='tight')
#             plt.close()
    
#     def _plot_accuracy_by_video(self, results):
#         """Plot accuracy for each video"""
#         video_ids = [r['video_id'] for r in results]
#         accuracies = [r['accuracy'] for r in results]
        
#         plt.figure(figsize=(12, 6))
#         bars = plt.bar(range(len(video_ids)), accuracies, alpha=0.7, color='lightgreen')
        
#         # Color code by performance
#         for i, (bar, acc) in enumerate(zip(bars, accuracies)):
#             if acc >= 0.85:
#                 bar.set_color('green')
#             elif acc >= 0.75:
#                 bar.set_color('yellow')
#             else:
#                 bar.set_color('red')
        
#         plt.xlabel('Video ID', fontsize=12)
#         plt.ylabel('Transcription Accuracy', fontsize=12)
#         plt.title('Accuracy by Video', fontsize=14, fontweight='bold')
#         plt.xticks(range(len(video_ids)), video_ids, rotation=45, ha='right')
#         plt.ylim(0, 1.1)
#         plt.axhline(y=0.75, color='orange', linestyle='--', linewidth=1, label='75% threshold')
#         plt.legend()
#         plt.grid(True, alpha=0.3, axis='y')
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'accuracy_by_video.png', dpi=300, bbox_inches='tight')
#         plt.close()
    
#     def _plot_error_heatmap(self, results):
#         """Plot heatmap of WER vs video characteristics"""
#         # Create dataframe
#         df = pd.DataFrame([{
#             'video_id': r['video_id'],
#             'WER': r['wer'],
#             'word_count': r['word_count']
#         } for r in results])
        
#         # Sort by WER
#         df = df.sort_values('WER', ascending=False)
        
#         plt.figure(figsize=(10, max(6, len(results) * 0.3)))
        
#         # Create heatmap data
#         heatmap_data = df[['WER']].T
        
#         sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn_r',
#                    xticklabels=df['video_id'].values, yticklabels=['WER'],
#                    cbar_kws={'label': 'Error Rate'})
        
#         plt.title('Error Rate Heatmap by Video', fontsize=14, fontweight='bold')
#         plt.xlabel('Video ID', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(self.output_dir / 'error_heatmap.png', dpi=300, bbox_inches='tight')
#         plt.close()


# # Example usage
# if __name__ == "__main__":
#     print("="*80)
#     print("WHISPER TRANSCRIPTION VALIDATION")
#     print("="*80)
    
#     # Initialize validator
#     validator = WhisperValidator(output_dir="whisper_validation_results")
    
#     # Initialize analyzer
#     print("\nLoading Whisper model...")
#     analyzer = InterviewAnalyzer(use_gpu=True, storage_dir="store")
#     analyzer.load_transcription_model("openai/whisper-base")
    
#     # Auto-detect video and transcript files
#     print("\n" + "="*80)
#     test_cases = validator.auto_detect_test_files(
#         video_dir="/home/mohit/Downloads/data/wisper_data/video",      # Your video folder
#         transcript_dir="/home/mohit/Downloads/data/wisper_data/transcript"  # Your transcript folder
#     )
    
#     if len(test_cases) == 0:
#         print("Error: No matching video/transcript pairs found!")
#         print("Please check your folder structure:")
#         print("  data/video/1.mp4")
#         print("  data/transcript/1.txt")
#         sys.exit(1)
    
#     # Optional: Add demographic information for bias detection
#     test_cases = validator.add_demographics_interactive(test_cases)
    
#     # Run validation
#     results = validator.validate_transcription(analyzer, test_cases)
    
#     if results:
#         print("\n" + "="*80)
#         print("VALIDATION COMPLETE!")
#         print("="*80)
#         print(f"\nCheck the 'whisper_validation_results' folder for:")
#         print("  - whisper_validation_results.json (full metrics)")
#         print("  - whisper_validation_detailed.csv (individual results)")
#         print("  - Visualization plots (PNG files)")
#         print("\nNext: Analyze bias patterns and document results for submission")


"""
Whisper Transcription Validation Script
Validates speech-to-text accuracy and detects bias across demographic groups
"""

import sys
sys.path.append("/home/mohit/.local/lib/python3.13/site-packages")

import os
import json
import numpy as np
import pandas as pd

# IMPORTANT: Set matplotlib backend to non-GUI before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server/headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from jiwer import wer, cer, mer, wil

# Import your analyzer
from main import InterviewAnalyzer


class WhisperValidator:
    def __init__(self, output_dir="whisper_validation_results"):
        """
        Initialize Whisper validator
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("Initializing Whisper Validator...")
        print(f"Results will be saved to: {self.output_dir.absolute()}\n")
    
    def auto_detect_test_files(self, video_dir: str, transcript_dir: str) -> List[Dict]:
        """
        Automatically detect matching video and transcript files
        
        Args:
            video_dir: Directory containing video files (e.g., "data/video")
            transcript_dir: Directory containing transcript files (e.g., "data/transcript")
            
        Returns:
            List of test cases with matched files
        """
        video_dir = Path(video_dir)
        transcript_dir = Path(transcript_dir)
        
        print(f"Scanning for test files...")
        print(f"  Video directory: {video_dir}")
        print(f"  Transcript directory: {transcript_dir}")
        
        test_cases = []
        
        # Find all video files
        video_extensions = ['.mp4', '.webm', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
        
        print(f"\nFound {len(video_files)} video files")
        
        for video_path in sorted(video_files):
            video_id = video_path.stem  # filename without extension
            
            # Look for matching transcript
            transcript_path = transcript_dir / f"{video_id}.txt"
            
            if not transcript_path.exists():
                print(f"  ⚠ Warning: No transcript found for {video_id}, skipping...")
                continue
            
            # Add to test cases
            test_case = {
                "video_id": video_id,
                "video_path": str(video_path),
                "transcript_path": str(transcript_path)
            }
            
            test_cases.append(test_case)
            print(f"  ✓ Matched: {video_id}")
        
        print(f"\nTotal matched pairs: {len(test_cases)}\n")
        return test_cases
    
    def add_demographics_interactive(self, test_cases: List[Dict]) -> List[Dict]:
        """
        Optionally add demographic info to test cases interactively
        """
        print("="*80)
        print("DEMOGRAPHIC INFORMATION (Optional - for bias detection)")
        print("="*80)
        print("You can add demographic info for bias analysis.")
        print("Press Enter to skip any field.\n")
        
        add_demographics = input("Add demographic information? (y/n) [n]: ").lower() == 'y'
        
        if not add_demographics:
            # Add default values
            for test_case in test_cases:
                test_case['speaker'] = test_case['video_id']
                test_case['demographics'] = {
                    'gender': 'unknown',
                    'age_group': 'unknown',
                    'accent': 'unknown',
                    'ethnicity': 'unknown'
                }
                test_case['audio_quality'] = 'unknown'
                test_case['speaking_speed'] = 'unknown'
            return test_cases
        
        print("\nEnter demographics for each video:")
        print("(Press Enter to use defaults)\n")
        
        for test_case in test_cases:
            print(f"\n--- Video: {test_case['video_id']} ---")
            
            test_case['speaker'] = input(f"  Speaker name [{test_case['video_id']}]: ") or test_case['video_id']
            test_case['demographics'] = {
                'gender': input("  Gender (male/female/other) [unknown]: ") or "unknown",
                'age_group': input("  Age group (18-25/25-35/35-50/50+) [unknown]: ") or "unknown",
                'accent': input("  Accent (american/british/indian/etc) [unknown]: ") or "unknown",
                'ethnicity': input("  Ethnicity [unknown]: ") or "unknown"
            }
            test_case['audio_quality'] = input("  Audio quality (clear/moderate/poor) [unknown]: ") or "unknown"
            test_case['speaking_speed'] = input("  Speaking speed (fast/normal/slow) [unknown]: ") or "unknown"
        
        return test_cases
    
    def validate_transcription(
        self,
        analyzer: InterviewAnalyzer,
        test_cases: List[Dict]
    ) -> Dict:
        """
        Validate Whisper transcription model
        
        Args:
            analyzer: InterviewAnalyzer with loaded transcription model
            test_cases: List of test cases
            
        Returns:
            Dictionary with validation metrics and bias analysis
        """
        print("\n" + "="*80)
        print("VALIDATING WHISPER TRANSCRIPTION MODEL")
        print("="*80)
        print(f"Total test cases: {len(test_cases)}\n")
        
        results = []
        wer_scores = []
        cer_scores = []
        mer_scores = []  # Match Error Rate
        wil_scores = []  # Word Information Lost
        
        # For bias detection - group by demographics
        slices = {
            'gender': {},
            'age_group': {},
            'accent': {},
            'ethnicity': {},
            'audio_quality': {},
            'speaking_speed': {}
        }
        
        for idx, test_case in enumerate(test_cases, 1):
            video_path = test_case['video_path']
            transcript_path = test_case['transcript_path']
            
            print(f"Processing {idx}/{len(test_cases)}: {test_case['video_id']}")
            
            # Load ground truth
            with open(transcript_path, 'r', encoding='utf-8') as f:
                ground_truth = f.read().strip()
            
            if not ground_truth:
                print(f"  Warning: Empty transcript, skipping...")
                continue
            
            # Extract audio and transcribe
            try:
                audio_path = analyzer.extract_audio(video_path)
                prediction = analyzer.transcribe_audio(audio_path)
                
                # Clean up audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                
            except Exception as e:
                print(f"  Error processing video: {e}")
                continue
            
            # Calculate metrics
            try:
                wer_score = wer(ground_truth, prediction)
                cer_score = cer(ground_truth, prediction)
                mer_score = mer(ground_truth, prediction)
                wil_score = wil(ground_truth, prediction)
            except Exception as e:
                print(f"  Error calculating metrics: {e}")
                continue
            
            wer_scores.append(wer_score)
            cer_scores.append(cer_score)
            mer_scores.append(mer_score)
            wil_scores.append(wil_score)
            
            print(f"  WER: {wer_score:.2%} | CER: {cer_score:.2%} | Accuracy: {(1-wer_score):.2%}")
            
            # Store result
            result = {
                'video_id': test_case['video_id'],
                'speaker': test_case.get('speaker', test_case['video_id']),
                'ground_truth': ground_truth,
                'prediction': prediction,
                'ground_truth_length': len(ground_truth),
                'prediction_length': len(prediction),
                'word_count': len(ground_truth.split()),
                'wer': wer_score,
                'cer': cer_score,
                'mer': mer_score,
                'wil': wil_score,
                'accuracy': 1 - wer_score,
                'demographics': test_case.get('demographics', {}),
                'audio_quality': test_case.get('audio_quality', 'unknown'),
                'speaking_speed': test_case.get('speaking_speed', 'unknown')
            }
            results.append(result)
            
            # Collect metrics by slice for bias detection
            demographics = test_case.get('demographics', {})
            
            for slice_key in ['gender', 'age_group', 'accent', 'ethnicity']:
                slice_value = demographics.get(slice_key, 'unknown')
                if slice_value != 'unknown':
                    if slice_value not in slices[slice_key]:
                        slices[slice_key][slice_value] = []
                    slices[slice_key][slice_value].append(wer_score)
            
            # Collect by audio quality and speaking speed
            audio_quality = test_case.get('audio_quality', 'unknown')
            if audio_quality != 'unknown':
                if audio_quality not in slices['audio_quality']:
                    slices['audio_quality'][audio_quality] = []
                slices['audio_quality'][audio_quality].append(wer_score)
            
            speaking_speed = test_case.get('speaking_speed', 'unknown')
            if speaking_speed != 'unknown':
                if speaking_speed not in slices['speaking_speed']:
                    slices['speaking_speed'][speaking_speed] = []
                slices['speaking_speed'][speaking_speed].append(wer_score)
        
        if len(results) == 0:
            print("Error: No results to analyze!")
            return None
        
        # Calculate overall metrics
        print("\n" + "="*80)
        print("OVERALL METRICS")
        print("="*80)
        
        avg_wer = np.mean(wer_scores)
        avg_cer = np.mean(cer_scores)
        avg_mer = np.mean(mer_scores)
        avg_wil = np.mean(wil_scores)
        avg_accuracy = 1 - avg_wer
        
        std_wer = np.std(wer_scores)
        median_wer = np.median(wer_scores)
        
        print(f"\nTranscription Accuracy Metrics:")
        print(f"  Average WER (Word Error Rate): {avg_wer:.2%}")
        print(f"  Average CER (Character Error Rate): {avg_cer:.2%}")
        print(f"  Average MER (Match Error Rate): {avg_mer:.2%}")
        print(f"  Average WIL (Word Info Lost): {avg_wil:.2%}")
        print(f"  Average Accuracy: {avg_accuracy:.2%}")
        print(f"  Median WER: {median_wer:.2%}")
        print(f"  Std Dev WER: {std_wer:.2%}")
        
        # Calculate metrics by slice (bias detection)
        print(f"\n" + "="*80)
        print("BIAS DETECTION - PERFORMANCE BY DEMOGRAPHIC SLICE")
        print("="*80)
        
        slice_metrics = {}
        bias_detected = {}
        
        for slice_name, slice_groups in slices.items():
            if len(slice_groups) == 0 or all(len(v) == 0 for v in slice_groups.values()):
                continue
            
            print(f"\n{slice_name.upper().replace('_', ' ')}:")
            slice_metrics[slice_name] = {}
            
            for group, scores in slice_groups.items():
                if len(scores) > 0:
                    avg_score = np.mean(scores)
                    slice_metrics[slice_name][group] = {
                        'avg_wer': avg_score,
                        'avg_accuracy': 1 - avg_score,
                        'count': len(scores),
                        'std_dev': np.std(scores)
                    }
                    print(f"  {group}: WER={avg_score:.2%}, Accuracy={1-avg_score:.2%} (n={len(scores)})")
            
            # Check for bias (>10% difference between groups)
            if len(slice_metrics[slice_name]) > 1:
                wer_values = [m['avg_wer'] for m in slice_metrics[slice_name].values()]
                max_wer = max(wer_values)
                min_wer = min(wer_values)
                difference = max_wer - min_wer
                
                if difference > 0.10:  # 10% threshold
                    bias_detected[slice_name] = {
                        'difference': difference,
                        'max_wer': max_wer,
                        'min_wer': min_wer,
                        'severity': 'HIGH' if difference > 0.20 else 'MODERATE'
                    }
                    print(f"  ⚠ BIAS DETECTED: {difference:.2%} difference ({bias_detected[slice_name]['severity']})")
        
        # Determine validation pass/fail
        validation_passed = avg_wer < 0.30  # 25% WER threshold
        
        print(f"\n" + "="*80)
        print(f"VALIDATION: {'✓ PASSED' if validation_passed else '✗ FAILED'}")
        print(f"  Threshold: WER < 30%")
        print(f"  Your WER: {avg_wer:.2%}")
        if bias_detected:
            print(f"  ⚠ Bias detected in {len(bias_detected)} demographic slice(s)")
        print("="*80)
        
        # Compile results
        validation_results = {
            'overall': {
                'average_wer': avg_wer,
                'average_cer': avg_cer,
                'average_mer': avg_mer,
                'average_wil': avg_wil,
                'average_accuracy': avg_accuracy,
                'median_wer': median_wer,
                'std_dev_wer': std_wer,
                'num_samples': len(results)
            },
            'by_slice': slice_metrics,
            'bias_detected': bias_detected,
            'individual_results': results,
            'validation_passed': validation_passed,
            'validation_threshold': 0.25,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self._plot_wer_distribution(wer_scores)
        self._plot_slice_comparison(slice_metrics)
        self._plot_accuracy_by_video(results)
        
        if len(wer_scores) > 5:
            self._plot_error_heatmap(results)
        
        # Save results (convert numpy types to Python native types for JSON)
        output_file = self.output_dir / "whisper_validation_results.json"
        
        # Helper function to convert numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        validation_results = convert_numpy_types(validation_results)
        
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Save detailed CSV
        df = pd.DataFrame(results)
        csv_file = self.output_dir / "whisper_validation_detailed.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n✓ Results saved to: {output_file}")
        print(f"✓ Detailed CSV saved to: {csv_file}")
        print(f"✓ Visualizations saved to: {self.output_dir}\n")
        
        return validation_results
    
    def _plot_wer_distribution(self, wer_scores):
        """Plot WER distribution histogram"""
        plt.figure(figsize=(10, 6))
        plt.hist(wer_scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(np.mean(wer_scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(wer_scores):.2%}')
        plt.axvline(np.median(wer_scores), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(wer_scores):.2%}')
        plt.xlabel('Word Error Rate (WER)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Transcription Error Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'wer_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_slice_comparison(self, slice_metrics):
        """Plot comparison across demographic slices"""
        for slice_name, groups in slice_metrics.items():
            if len(groups) == 0:
                continue
            
            group_names = list(groups.keys())
            wer_values = [groups[g]['avg_wer'] for g in group_names]
            counts = [groups[g]['count'] for g in group_names]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(group_names, wer_values, alpha=0.7, color='coral')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'n={count}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel(slice_name.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Average WER', fontsize=12)
            ax.set_title(f'WER by {slice_name.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(wer_values) * 1.2)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'wer_by_{slice_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_accuracy_by_video(self, results):
        """Plot accuracy for each video"""
        video_ids = [r['video_id'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(video_ids)), accuracies, alpha=0.7, color='lightgreen')
        
        # Color code by performance
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc >= 0.85:
                bar.set_color('green')
            elif acc >= 0.75:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        plt.xlabel('Video ID', fontsize=12)
        plt.ylabel('Transcription Accuracy', fontsize=12)
        plt.title('Accuracy by Video', fontsize=14, fontweight='bold')
        plt.xticks(range(len(video_ids)), video_ids, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.axhline(y=0.75, color='orange', linestyle='--', linewidth=1, label='75% threshold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_video.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_heatmap(self, results):
        """Plot heatmap of WER vs video characteristics"""
        # Create dataframe
        df = pd.DataFrame([{
            'video_id': r['video_id'],
            'WER': r['wer'],
            'word_count': r['word_count']
        } for r in results])
        
        # Sort by WER
        df = df.sort_values('WER', ascending=False)
        
        plt.figure(figsize=(10, max(6, len(results) * 0.3)))
        
        # Create heatmap data
        heatmap_data = df[['WER']].T
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn_r',
                   xticklabels=df['video_id'].values, yticklabels=['WER'],
                   cbar_kws={'label': 'Error Rate'})
        
        plt.title('Error Rate Heatmap by Video', fontsize=14, fontweight='bold')
        plt.xlabel('Video ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("WHISPER TRANSCRIPTION VALIDATION")
    print("="*80)
    
    # Initialize validator
    validator = WhisperValidator(output_dir="whisper_validation_results")
    
    # Initialize analyzer
    print("\nLoading Whisper model...")
    analyzer = InterviewAnalyzer(use_gpu=True, storage_dir="store")
    analyzer.load_transcription_model("openai/whisper-base")
    
    # Auto-detect video and transcript files
    print("\n" + "="*80)
    test_cases = validator.auto_detect_test_files(
        video_dir="/home/mohit/Downloads/data/wisper_data/video",      # Your video folder
        transcript_dir="/home/mohit/Downloads/data/wisper_data/transcript"  # Your transcript folder
    )
    
    if len(test_cases) == 0:
        print("Error: No matching video/transcript pairs found!")
        print("Please check your folder structure:")
        print("  data/video/1.mp4")
        print("  data/transcript/1.txt")
        sys.exit(1)
    
    # Optional: Add demographic information for bias detection
    test_cases = validator.add_demographics_interactive(test_cases)
    
    # Run validation
    results = validator.validate_transcription(analyzer, test_cases)
    
    if results:
        print("\n" + "="*80)
        print("VALIDATION COMPLETE!")
        print("="*80)
        print(f"\nCheck the 'whisper_validation_results' folder for:")
        print("  - whisper_validation_results.json (full metrics)")
        print("  - whisper_validation_detailed.csv (individual results)")
        print("  - Visualization plots (PNG files)")
        print("\nNext: Analyze bias patterns and document results for submission")