"""
Analyze and interpret LLM validation results
Provides insights and recommendations based on validation metrics
"""

import json
import pandas as pd
from pathlib import Path


def analyze_validation_results(results_path="llm_validation_results/llm_validation_results.json"):
    """
    Analyze validation results and provide recommendations
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    overall = results['overall']
    by_quality = results['by_quality_level']
    thresholds = results['validation_thresholds']
    
    print("="*80)
    print("LLM VALIDATION ANALYSIS REPORT")
    print("="*80)
    print(f"\nTest Date: {results['timestamp']}")
    print(f"Total Samples: {overall['num_samples']}")
    print(f"\nValidation Status: {'✓ PASSED' if results['validation_passed'] else '✗ FAILED'}")
    
    # Analyze each metric
    print("\n" + "="*80)
    print("METRIC ANALYSIS")
    print("="*80)
    
    # ROUGE-L Analysis
    print(f"\n1. ROUGE-L Score: {overall['rouge_l']:.3f}")
    print(f"   Threshold: {thresholds['rouge_l']}")
    if overall['rouge_l'] >= thresholds['rouge_l']:
        print("   Status: GOOD - Text overlap with expert feedback is adequate")
    else:
        print("   Status: NEEDS IMPROVEMENT")
        print("   Recommendation: LLM feedback doesn't match expert wording enough")
        print("   - Try fine-tuning with more examples")
        print("   - Improve prompt engineering to match expert style")
    
    # Semantic Similarity Analysis
    print(f"\n2. Semantic Similarity: {overall['semantic_similarity']:.3f}")
    print(f"   Threshold: {thresholds['semantic_similarity']}")
    if overall['semantic_similarity'] >= thresholds['semantic_similarity']:
        print("   Status: GOOD - LLM captures similar meaning to experts")
    else:
        print("   Status: NEEDS IMPROVEMENT")
        print("   Recommendation: LLM feedback doesn't convey expert concepts well")
        print("   - Review prompt to emphasize key evaluation criteria")
        print("   - Consider using a larger/better model")
    
    # Correlation Analysis
    print(f"\n3. Score Correlation (Pearson): {overall['pearson_correlation']:.3f}")
    print(f"   Threshold: {thresholds['pearson_correlation']}")
    if overall['pearson_correlation'] >= thresholds['pearson_correlation']:
        print("   Status: GOOD - LLM scores track expert scores well")
    else:
        print("   Status: NEEDS IMPROVEMENT")
        print("   Recommendation: LLM scoring doesn't align with expert judgment")
        print("   - Add more explicit scoring criteria to prompt")
        print("   - Train model to output structured scores")
    
    # MAE Analysis
    print(f"\n4. Mean Absolute Error: {overall['mae']:.2f} points")
    print(f"   Threshold: {thresholds['mae']}")
    if overall['mae'] <= thresholds['mae']:
        print("   Status: GOOD - Scores are close to expert ratings")
    else:
        print("   Status: NEEDS IMPROVEMENT")
        print("   Recommendation: LLM scoring is off by too much")
        print("   - Review cases with largest errors")
        print("   - Calibrate scoring scale in prompt")
    
    # Quality Level Analysis
    print("\n" + "="*80)
    print("PERFORMANCE BY ANSWER QUALITY")
    print("="*80)
    
    for level in ['excellent', 'good', 'average', 'poor']:
        if level in by_quality:
            metrics = by_quality[level]
            print(f"\n{level.upper()} Answers (n={metrics['count']}):")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  ROUGE-L: {metrics['avg_rouge_l']:.3f}")
            print(f"  Semantic Sim: {metrics['avg_semantic_similarity']:.3f}")
            
            # Identify issues
            if metrics['mae'] > 2.0:
                print(f"  Warning: High error on {level} answers")
            if metrics['avg_semantic_similarity'] < 0.5:
                print(f"  Warning: Poor semantic match on {level} answers")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    issues = []
    if overall['rouge_l'] < thresholds['rouge_l']:
        issues.append("low_rouge")
    if overall['semantic_similarity'] < thresholds['semantic_similarity']:
        issues.append("low_semantic")
    if overall['pearson_correlation'] < thresholds['pearson_correlation']:
        issues.append("low_correlation")
    if overall['mae'] > thresholds['mae']:
        issues.append("high_error")
    
    if not issues:
        print("\n✓ All metrics passed! Your LLM is performing well.")
        print("\nNext steps:")
        print("  1. Document these results in your project submission")
        print("  2. Proceed to VLM validation")
        print("  3. Set up CI/CD pipeline for automated validation")
    else:
        print("\nIssues detected. Suggested fixes:\n")
        
        if "low_rouge" in issues or "low_semantic" in issues:
            print("Improve Prompt Engineering:")
            print("  - Add more specific instructions about feedback format")
            print("  - Include examples of good feedback in the prompt")
            print("  - Specify desired feedback length and structure")
            
        if "low_correlation" in issues or "high_error" in issues:
            print("\nImprove Scoring Calibration:")
            print("  - Add explicit scoring rubric to prompt")
            print("  - Use few-shot examples with scores")
            print("  - Consider structured output format (JSON)")
            
        print("\nModel Improvements:")
        print("  - Try a larger model (e.g., flan-t5-xl or flan-t5-xxl)")
        print("  - Fine-tune on your labeled data")
        print("  - Experiment with temperature/sampling parameters")
        
        print("\nData Improvements:")
        print("  - Collect more diverse training examples")
        print("  - Ensure expert feedback is consistent")
        print("  - Balance dataset across quality levels")
    
    # Find worst performing cases
    print("\n" + "="*80)
    print("TOP 5 CASES WITH LARGEST ERRORS")
    print("="*80)
    
    individual = results['individual_results']
    sorted_by_error = sorted(individual, key=lambda x: x['score_difference'], reverse=True)
    
    for i, case in enumerate(sorted_by_error[:5], 1):
        print(f"\n{i}. Test ID: {case['test_id']} | Error: {case['score_difference']:.1f} points")
        print(f"   Question: {case['question'][:70]}...")
        print(f"   Expert Score: {case['expert_scores']['overall']} | LLM Score: {case['llm_scores']['overall']:.1f}")
        print(f"   ROUGE-L: {case['rouge_l']:.3f} | Semantic Sim: {case['semantic_similarity']:.3f}")
    
    print("\n" + "="*80)
    print("For detailed results, see:")
    print(f"  - {Path(results_path).parent / 'llm_validation_detailed.csv'}")
    print(f"  - Visualization plots in {Path(results_path).parent}")
    print("="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    # Check if results file exists
    results_file = "llm_validation_results/llm_validation_results.json"
    
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run the validation script first!")
        sys.exit(1)
    
    analyze_validation_results(results_file)
