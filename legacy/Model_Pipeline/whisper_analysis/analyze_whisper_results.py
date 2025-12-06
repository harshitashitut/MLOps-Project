"""
Analyze Whisper Validation Results
Provides insights and recommendations
"""

import json
import pandas as pd
from pathlib import Path


def analyze_whisper_results(results_path="whisper_validation_results/whisper_validation_results.json"):
    """
    Analyze Whisper validation results and provide recommendations
    """
    # Load results
    if not Path(results_path).exists():
        print(f"Error: Results file not found: {results_path}")
        print("Please run whisper_validation.py first!")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    overall = results['overall']
    by_slice = results.get('by_slice', {})
    bias_detected = results.get('bias_detected', {})
    
    print("="*80)
    print("WHISPER TRANSCRIPTION ANALYSIS REPORT")
    print("="*80)
    print(f"\nTest Date: {results['timestamp']}")
    print(f"Total Samples: {overall['num_samples']}")
    print(f"\nValidation Status: {'✓ PASSED' if results['validation_passed'] else '✗ FAILED'}")
    
    # Overall performance
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    avg_wer = overall['average_wer']
    avg_accuracy = overall['average_accuracy']
    
    print(f"\nWord Error Rate (WER): {avg_wer:.2%}")
    print(f"Transcription Accuracy: {avg_accuracy:.2%}")
    
    # Performance rating
    if avg_wer < 0.10:
        rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        feedback = "Near-perfect transcription quality"
    elif avg_wer < 0.15:
        rating = "VERY GOOD ⭐⭐⭐⭐"
        feedback = "High quality with minor errors"
    elif avg_wer < 0.25:
        rating = "GOOD ⭐⭐⭐"
        feedback = "Acceptable quality, some errors present"
    else:
        rating = "NEEDS IMPROVEMENT ⭐⭐"
        feedback = "Too many errors, requires attention"
    
    print(f"\nPerformance Rating: {rating}")
    print(f"Assessment: {feedback}")
    
    # Detailed metrics
    print(f"\nDetailed Metrics:")
    print(f"  Character Error Rate (CER): {overall['average_cer']:.2%}")
    print(f"  Match Error Rate (MER): {overall['average_mer']:.2%}")
    print(f"  Word Information Lost (WIL): {overall['average_wil']:.2%}")
    print(f"  Median WER: {overall['median_wer']:.2%}")
    print(f"  Standard Deviation: {overall['std_dev_wer']:.2%}")
    
    # Consistency analysis
    std_wer = overall['std_dev_wer']
    if std_wer < 0.05:
        consistency = "Very consistent performance across videos"
    elif std_wer < 0.10:
        consistency = "Reasonably consistent"
    else:
        consistency = "Inconsistent - check for data quality issues"
    
    print(f"\nConsistency: {consistency}")
    
    # Bias analysis
    if bias_detected:
        print("\n" + "="*80)
        print("⚠ BIAS DETECTION ALERT")
        print("="*80)
        
        for slice_name, bias_info in bias_detected.items():
            print(f"\n{slice_name.upper().replace('_', ' ')}:")
            print(f"  Severity: {bias_info['severity']}")
            print(f"  Performance Gap: {bias_info['difference']:.2%}")
            print(f"  Best Group WER: {bias_info['min_wer']:.2%}")
            print(f"  Worst Group WER: {bias_info['max_wer']:.2%}")
            
            # Recommendations
            if slice_name == 'accent':
                print(f"\n  Recommendations:")
                print(f"    - Fine-tune Whisper on underrepresented accents")
                print(f"    - Consider using whisper-large for better accent handling")
                print(f"    - Collect more training data from diverse accents")
            elif slice_name == 'gender':
                print(f"\n  Recommendations:")
                print(f"    - Balance training data across genders")
                print(f"    - Check audio quality consistency")
            elif slice_name == 'audio_quality':
                print(f"\n  Recommendations:")
                print(f"    - Implement audio preprocessing (noise reduction)")
                print(f"    - Use higher quality recording equipment")
                print(f"    - Filter out low-quality samples before transcription")
    else:
        print("\n" + "="*80)
        print("✓ NO SIGNIFICANT BIAS DETECTED")
        print("="*80)
        print("Model performs fairly across demographic groups")
    
    # Performance by demographic slice
    if by_slice:
        print("\n" + "="*80)
        print("PERFORMANCE BY DEMOGRAPHIC GROUP")
        print("="*80)
        
        for slice_name, groups in by_slice.items():
            if len(groups) > 0:
                print(f"\n{slice_name.upper().replace('_', ' ')}:")
                
                # Sort by WER
                sorted_groups = sorted(groups.items(), key=lambda x: x[1]['avg_wer'])
                
                for group, metrics in sorted_groups:
                    status = "✓" if metrics['avg_wer'] < 0.15 else "⚠" if metrics['avg_wer'] < 0.25 else "✗"
                    print(f"  {status} {group}: {metrics['avg_accuracy']:.2%} accuracy (WER: {metrics['avg_wer']:.2%}, n={metrics['count']})")
    
    # Find worst performing videos
    print("\n" + "="*80)
    print("TOP 5 VIDEOS WITH HIGHEST ERROR RATES")
    print("="*80)
    
    individual = results['individual_results']
    sorted_by_wer = sorted(individual, key=lambda x: x['wer'], reverse=True)
    
    for i, video in enumerate(sorted_by_wer[:5], 1):
        print(f"\n{i}. Video: {video['video_id']}")
        print(f"   WER: {video['wer']:.2%} | Accuracy: {video['accuracy']:.2%}")
        print(f"   Words: {video['word_count']} | Audio Quality: {video.get('audio_quality', 'unknown')}")
        
        # Show snippets if available
        if 'ground_truth' in video and 'prediction' in video:
            gt_snippet = video['ground_truth'][:80] + "..." if len(video['ground_truth']) > 80 else video['ground_truth']
            pred_snippet = video['prediction'][:80] + "..." if len(video['prediction']) > 80 else video['prediction']
            print(f"   Ground Truth: {gt_snippet}")
            print(f"   Prediction:   {pred_snippet}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if avg_wer < 0.15:
        print("\n✓ Performance is strong. Continue with current setup.")
        print("\nNext steps:")
        print("  1. Document these results in your project report")
        print("  2. Include all visualization plots")
        print("  3. Proceed to VLM validation")
    else:
        print("\nSuggested improvements:\n")
        
        if avg_wer > 0.25:
            print("🎯 CRITICAL - Model Performance:")
            print("  - Use whisper-medium or whisper-large instead of whisper-base")
            print("  - Fine-tune Whisper on your specific domain/accent")
            print("  - Verify ground truth transcripts are accurate")
        
        if std_wer > 0.10:
            print("\n📊 Data Quality:")
            print("  - Ensure consistent audio quality across all videos")
            print("  - Remove background noise using audio preprocessing")
            print("  - Standardize recording conditions")
        
        if bias_detected:
            print("\n⚖️ Bias Mitigation:")
            print("  - Collect more data from underrepresented groups")
            print("  - Use data augmentation for minority groups")
            print("  - Consider ensemble models with accent-specific fine-tuning")
        
        print("\n🔧 Technical Improvements:")
        print("  - Try different Whisper model sizes:")
        print("    • whisper-small (better than base, ~2GB RAM)")
        print("    • whisper-medium (best balance, ~5GB RAM)")
        print("    • whisper-large-v2 (highest quality, ~10GB RAM)")
        print("  - Adjust chunk_length_s parameter for longer utterances")
        print("  - Experiment with language-specific fine-tuned models")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY FOR PROJECT DOCUMENTATION")
    print("="*80)
    
    print(f"""
Whisper Transcription Model Validation

Dataset Composition:
- Total samples: {overall['num_samples']}
- Average transcript length: {np.mean([r['word_count'] for r in individual]):.0f} words

Performance Metrics:
- Word Error Rate: {avg_wer:.2%}
- Transcription Accuracy: {avg_accuracy:.2%}
- Character Error Rate: {overall['average_cer']:.2%}
- Validation Status: {'PASSED ✓' if results['validation_passed'] else 'FAILED ✗'}

Bias Analysis:
- Demographic slices evaluated: {len(by_slice)}
- Bias detected: {'Yes (' + str(len(bias_detected)) + ' slices)' if bias_detected else 'No'}

Model Configuration:
- Model: openai/whisper-base
- Device: {'GPU' if 'cuda' in str(individual[0]) else 'CPU'}

Conclusion:
{feedback}
{'No significant bias across demographic groups.' if not bias_detected else 'Bias mitigation strategies recommended for ' + ', '.join(bias_detected.keys()) + '.'}
""")
    
    print("="*80)
    print("\nDetailed results available in:")
    print(f"  - {Path(results_path).parent / 'whisper_validation_detailed.csv'}")
    print(f"  - Visualization plots in {Path(results_path).parent}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    import numpy as np
    
    results_file = "whisper_validation_results/whisper_validation_results.json"
    analyze_whisper_results(results_file)