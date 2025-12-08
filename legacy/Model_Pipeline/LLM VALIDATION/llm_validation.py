"""
LLM Validation Script for Interview Analysis System
Validates LLM feedback quality against expert-labeled data
"""

import sys
sys.path.append("/home/mohit/.local/lib/python3.13/site-packages")

import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Metrics libraries
from rouge_score import rouge_scorer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer

# Import your analyzer
from main1 import InterviewAnalyzer


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(i) for i in obj)
    return obj


class LLMValidator:
    def __init__(self, output_dir="llm_validation_results"):
        """
        Initialize LLM validator
        
        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print("Initializing LLM Validator...")
        print(f"Results will be saved to: {self.output_dir.absolute()}")
        
        # Load semantic similarity model
        print("Loading semantic similarity model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        print("Initialization complete!\n")
    
    def parse_test_file(self, file_path: str) -> List[Dict]:
        """
        Parse test data files (Software Engineering, Web Development, etc.)
        
        Expected format in text file:
        Q: Question text
        A: Answer text
        Structure: X | Clarity: Y | Relevance: Z | Overall: W
        Feedback: Expert feedback text
        
        Returns:
            List of test cases with questions, answers, scores, and feedback
        """
        test_cases = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by question blocks (look for "Q:" pattern)
        blocks = re.split(r'\n\n+(?=Q:|\d+\.)', content)
        
        for block in blocks:
            if not block.strip() or 'TEST SET' in block:
                continue
            
            # Extract question
            q_match = re.search(r'Q:\s*(.+?)(?=\nA:)', block, re.DOTALL)
            if not q_match:
                continue
            question = q_match.group(1).strip()
            
            # Extract answer
            a_match = re.search(r'A:\s*(.+?)(?=\nStructure:)', block, re.DOTALL)
            if not a_match:
                continue
            answer = a_match.group(1).strip()
            
            # Extract scores
            scores = {}
            score_pattern = r'Structure:\s*(\d+)\s*\|\s*Clarity:\s*(\d+)\s*\|\s*Relevance:\s*(\d+)\s*\|\s*Overall:\s*(\d+)'
            score_match = re.search(score_pattern, block)
            
            if score_match:
                scores = {
                    'structure': int(score_match.group(1)),
                    'clarity': int(score_match.group(2)),
                    'relevance': int(score_match.group(3)),
                    'overall': int(score_match.group(4))
                }
            else:
                continue
            
            # Extract expert feedback
            feedback_match = re.search(r'Feedback:\s*(.+?)(?=\n\n|\Z)', block, re.DOTALL)
            expert_feedback = feedback_match.group(1).strip() if feedback_match else ""
            
            test_cases.append({
                'question': question,
                'answer': answer,
                'expert_scores': scores,
                'expert_feedback': expert_feedback
            })
        
        return test_cases
    
    def extract_score_from_llm_feedback(self, feedback: str) -> Dict[str, float]:
        """
        Extract numeric scores from LLM feedback text
        Looks for patterns like "8/10", "Score: 8", "Rating: 8"
        Also tries to extract individual component scores
        """
        scores = {
            'structure': None,
            'clarity': None,
            'relevance': None,
            'overall': None
        }
        
        # Patterns to search for
        patterns = {
            'overall': [
                r'Overall[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
                r'Rating[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
                r'Score[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?',
                r'(\d+(?:\.\d+)?)\s*/\s*10',
                r'(\d+(?:\.\d+)?)\s+out of 10'
            ],
            'structure': [
                r'Structure[:\s]+(\d+(?:\.\d+)?)',
            ],
            'clarity': [
                r'Clarity[:\s]+(\d+(?:\.\d+)?)',
            ],
            'relevance': [
                r'Relevance[:\s]+(\d+(?:\.\d+)?)',
            ]
        }
        
        # Try to find scores
        for score_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, feedback, re.IGNORECASE)
                if match:
                    try:
                        scores[score_type] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        
        # If we couldn't find overall score, try to average what we found
        if scores['overall'] is None:
            component_scores = [s for s in [scores['structure'], scores['clarity'], scores['relevance']] if s is not None]
            if component_scores:
                scores['overall'] = np.mean(component_scores)
            else:
                # Last resort: look for any number in range 1-10
                number_match = re.search(r'\b([1-9]|10)(?:\.\d+)?\b', feedback)
                if number_match:
                    scores['overall'] = float(number_match.group(1))
                else:
                    scores['overall'] = 5.0  # Default if nothing found
        
        return scores
    
    def validate_llm(
        self,
        analyzer: InterviewAnalyzer,
        test_data: List[Dict],
        save_individual_results: bool = True
    ) -> Dict:
        """
        Validate LLM model against test data
        
        Args:
            analyzer: InterviewAnalyzer instance with loaded LLM
            test_data: List of test cases from parse_test_file()
            save_individual_results: Whether to save individual predictions
            
        Returns:
            Dictionary with comprehensive validation metrics
        """
        print("\n" + "="*60)
        print("VALIDATING LLM FEEDBACK MODEL")
        print("="*60)
        print(f"Total test cases: {len(test_data)}\n")
        
        results = []
        rouge_scores = []
        semantic_similarities = []
        
        expert_overall_scores = []
        llm_overall_scores = []
        
        # Component-wise scores
        expert_structure_scores = []
        llm_structure_scores = []
        expert_clarity_scores = []
        llm_clarity_scores = []
        expert_relevance_scores = []
        llm_relevance_scores = []
        
        # Categorize by quality level
        quality_levels = {
            'excellent': [],  # 9-10
            'good': [],       # 7-8
            'average': [],    # 5-6
            'poor': []        # 1-4
        }
        
        for idx, test_case in enumerate(test_data, 1):
            print(f"Processing {idx}/{len(test_data)}: {test_case['question'][:50]}...")
            
            question = test_case['question']
            answer = test_case['answer']
            expert_feedback = test_case['expert_feedback']
            expert_scores = test_case['expert_scores']
            
            # Get LLM feedback
            try:
                llm_feedback = analyzer.analyze_answer(question, answer)
            except Exception as e:
                print(f"  Error getting LLM feedback: {e}")
                llm_feedback = "Error generating feedback"
            
            # Extract scores from LLM feedback
            llm_scores = self.extract_score_from_llm_feedback(llm_feedback)
            
            # Calculate ROUGE score
            try:
                rouge = self.rouge_scorer.score(expert_feedback, llm_feedback)
                rouge_l = rouge['rougeL'].fmeasure
                rouge_1 = rouge['rouge1'].fmeasure
                rouge_2 = rouge['rouge2'].fmeasure
            except Exception as e:
                print(f"  Error calculating ROUGE: {e}")
                rouge_l = rouge_1 = rouge_2 = 0.0
            
            rouge_scores.append(rouge_l)
            
            # Calculate semantic similarity
            try:
                expert_emb = self.semantic_model.encode(expert_feedback)
                llm_emb = self.semantic_model.encode(llm_feedback)
                similarity = np.dot(expert_emb, llm_emb) / (
                    np.linalg.norm(expert_emb) * np.linalg.norm(llm_emb)
                )
            except Exception as e:
                print(f"  Error calculating similarity: {e}")
                similarity = 0.0
            
            semantic_similarities.append(similarity)
            
            # Store overall scores
            expert_overall = expert_scores['overall']
            llm_overall = llm_scores['overall']
            
            expert_overall_scores.append(expert_overall)
            llm_overall_scores.append(llm_overall)
            
            # Store component scores if available
            if llm_scores['structure'] is not None:
                expert_structure_scores.append(expert_scores.get('structure', expert_overall))
                llm_structure_scores.append(llm_scores['structure'])
            
            if llm_scores['clarity'] is not None:
                expert_clarity_scores.append(expert_scores.get('clarity', expert_overall))
                llm_clarity_scores.append(llm_scores['clarity'])
            
            if llm_scores['relevance'] is not None:
                expert_relevance_scores.append(expert_scores.get('relevance', expert_overall))
                llm_relevance_scores.append(llm_scores['relevance'])
            
            # Categorize by quality level
            if expert_overall >= 9:
                quality_levels['excellent'].append(idx - 1)
            elif expert_overall >= 7:
                quality_levels['good'].append(idx - 1)
            elif expert_overall >= 5:
                quality_levels['average'].append(idx - 1)
            else:
                quality_levels['poor'].append(idx - 1)
            
            # Store individual result
            result = {
                'test_id': idx,
                'question': question,
                'answer': answer[:200] + "..." if len(answer) > 200 else answer,
                'expert_feedback': expert_feedback,
                'llm_feedback': llm_feedback,
                'expert_scores': expert_scores,
                'llm_scores': llm_scores,
                'rouge_1': rouge_1,
                'rouge_2': rouge_2,
                'rouge_l': rouge_l,
                'semantic_similarity': similarity,
                'score_difference': abs(expert_overall - llm_overall)
            }
            results.append(result)
        
        print("\nCalculating overall metrics...")
        
        # Calculate correlation metrics for overall scores
        pearson_corr, pearson_p = pearsonr(expert_overall_scores, llm_overall_scores)
        spearman_corr, spearman_p = spearmanr(expert_overall_scores, llm_overall_scores)
        
        # Calculate error metrics
        mae = mean_absolute_error(expert_overall_scores, llm_overall_scores)
        rmse = np.sqrt(mean_squared_error(expert_overall_scores, llm_overall_scores))
        
        # Calculate metrics by quality level
        quality_metrics = {}
        for level, indices in quality_levels.items():
            if len(indices) > 0:
                level_expert = [expert_overall_scores[i] for i in indices]
                level_llm = [llm_overall_scores[i] for i in indices]
                level_mae = mean_absolute_error(level_expert, level_llm)
                level_rouge = np.mean([rouge_scores[i] for i in indices])
                level_semantic = np.mean([semantic_similarities[i] for i in indices])
                
                quality_metrics[level] = {
                    'count': len(indices),
                    'mae': level_mae,
                    'avg_rouge_l': level_rouge,
                    'avg_semantic_similarity': level_semantic
                }
        
        # Overall metrics
        avg_rouge_1 = np.mean([r['rouge_1'] for r in results])
        avg_rouge_2 = np.mean([r['rouge_2'] for r in results])
        avg_rouge_l = np.mean(rouge_scores)
        avg_semantic = np.mean(semantic_similarities)
        
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"\nText Similarity Metrics:")
        print(f"  Average ROUGE-1: {avg_rouge_1:.3f}")
        print(f"  Average ROUGE-2: {avg_rouge_2:.3f}")
        print(f"  Average ROUGE-L: {avg_rouge_l:.3f}")
        print(f"  Average Semantic Similarity: {avg_semantic:.3f}")
        
        print(f"\nScore Correlation Metrics:")
        print(f"  Pearson Correlation: {pearson_corr:.3f} (p={pearson_p:.4f})")
        print(f"  Spearman Correlation: {spearman_corr:.3f} (p={spearman_p:.4f})")
        
        print(f"\nScore Error Metrics:")
        print(f"  Mean Absolute Error (MAE): {mae:.2f} points")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} points")
        
        print(f"\nPerformance by Answer Quality:")
        for level, metrics in quality_metrics.items():
            print(f"  {level.upper()} (n={metrics['count']}):")
            print(f"    MAE: {metrics['mae']:.2f}")
            print(f"    ROUGE-L: {metrics['avg_rouge_l']:.3f}")
            print(f"    Semantic Similarity: {metrics['avg_semantic_similarity']:.3f}")
        
        # Determine validation pass/fail
        validation_passed = (
            avg_rouge_l > 0.3 and 
            avg_semantic > 0.6 and 
            pearson_corr > 0.7 and
            mae < 2.0
        )
        
        validation_results = {
            'overall': {
                'rouge_1': avg_rouge_1,
                'rouge_2': avg_rouge_2,
                'rouge_l': avg_rouge_l,
                'semantic_similarity': avg_semantic,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'mae': mae,
                'rmse': rmse,
                'num_samples': len(test_data)
            },
            'by_quality_level': quality_metrics,
            'validation_passed': validation_passed,
            'validation_thresholds': {
                'rouge_l': 0.3,
                'semantic_similarity': 0.6,
                'pearson_correlation': 0.7,
                'mae': 2.0
            },
            'individual_results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate visualizations
        self._plot_score_correlation(
            expert_overall_scores, 
            llm_overall_scores,
            "Overall Score Correlation"
        )
        
        self._plot_error_distribution(
            [r['score_difference'] for r in results],
            "Score Difference Distribution"
        )
        
        self._plot_quality_level_performance(quality_metrics)
        
        self._plot_rouge_semantic_scatter(rouge_scores, semantic_similarities)
        
        # Save results to JSON with numpy type conversion
        output_file = self.output_dir / "llm_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(convert_numpy_types(validation_results), f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}\n")
        
        # Save individual results to CSV for easy analysis
        if save_individual_results:
            df = pd.DataFrame(results)
            csv_file = self.output_dir / "llm_validation_detailed.csv"
            df.to_csv(csv_file, index=False)
            print(f"Detailed results saved to: {csv_file}\n")
        
        return validation_results
    
    def _plot_score_correlation(self, expert_scores, llm_scores, title):
        """Plot correlation between expert and LLM scores"""
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(expert_scores, llm_scores, alpha=0.6, s=100)
        
        # Perfect correlation line
        min_score = min(min(expert_scores), min(llm_scores))
        max_score = max(max(expert_scores), max(llm_scores))
        plt.plot([min_score, max_score], [min_score, max_score], 
                'r--', linewidth=2, label='Perfect correlation')
        
        # Best fit line
        z = np.polyfit(expert_scores, llm_scores, 1)
        p = np.poly1d(z)
        plt.plot(expert_scores, p(expert_scores), 
                'b-', linewidth=2, alpha=0.7, label=f'Best fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        plt.xlabel('Expert Scores', fontsize=12)
        plt.ylabel('LLM Scores', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'score_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, errors, title):
        """Plot distribution of score errors"""
        plt.figure(figsize=(10, 6))
        
        plt.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(np.mean(errors), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        plt.axvline(np.median(errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(errors):.2f}')
        
        plt.xlabel('Absolute Score Difference', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_level_performance(self, quality_metrics):
        """Plot performance metrics by answer quality level"""
        levels = list(quality_metrics.keys())
        mae_values = [quality_metrics[l]['mae'] for l in levels]
        rouge_values = [quality_metrics[l]['avg_rouge_l'] for l in levels]
        semantic_values = [quality_metrics[l]['avg_semantic_similarity'] for l in levels]
        
        x = np.arange(len(levels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width, mae_values, width, label='MAE', alpha=0.8)
        ax.bar(x, rouge_values, width, label='ROUGE-L', alpha=0.8)
        ax.bar(x + width, semantic_values, width, label='Semantic Sim.', alpha=0.8)
        
        ax.set_xlabel('Answer Quality Level', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Performance by Answer Quality Level', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([l.capitalize() for l in levels])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'quality_level_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rouge_semantic_scatter(self, rouge_scores, semantic_scores):
        """Plot relationship between ROUGE and semantic similarity"""
        plt.figure(figsize=(10, 8))
        
        plt.scatter(rouge_scores, semantic_scores, alpha=0.6, s=100, color='purple')
        
        # Add trend line
        z = np.polyfit(rouge_scores, semantic_scores, 1)
        p = np.poly1d(z)
        plt.plot(rouge_scores, p(rouge_scores), 
                'r-', linewidth=2, alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        
        plt.xlabel('ROUGE-L Score', fontsize=12)
        plt.ylabel('Semantic Similarity', fontsize=12)
        plt.title('ROUGE vs Semantic Similarity', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'rouge_vs_semantic.png', dpi=300, bbox_inches='tight')
        plt.close()


# Example usage
if __name__ == "__main__":
    print("LLM Validation Script")
    print("="*60)
    
    # Initialize validator
    validator = LLMValidator(output_dir="llm_validation_results")
    
    # Initialize your analyzer
    print("\nLoading analyzer and models...")
    analyzer = InterviewAnalyzer(use_gpu=True, storage_dir="store")
    analyzer.load_transcription_model("openai/whisper-base")
    
    # UPDATED: Use fine-tuned model trained on expert feedback
    analyzer.load_llm_model("fine_tuned_interview_model/final_model")
    
    # Parse test data files
    print("\nParsing test data files...")
    test_files = [
        "Data/Ml ROLE.txt",
        "Data/Software engineering role.txt",
        "Data/TEST SET MACHINE LEARNING.txt",
        "Data/TEST SET SOFTWARE ENGINEERING.txt",
        "Data/TEST SET WEB DEVELOPMENT.txt",
        "Data/Wed dev role.txt"
    ]
    
    all_test_data = []
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"  Loading: {file_path}")
            test_cases = validator.parse_test_file(file_path)
            all_test_data.extend(test_cases)
            print(f"    Found {len(test_cases)} test cases")
        else:
            print(f"  Warning: {file_path} not found, skipping...")
    
    print(f"\nTotal test cases loaded: {len(all_test_data)}")
    
    if len(all_test_data) == 0:
        print("Error: No test data found!")
        sys.exit(1)
    
    # Run validation
    print("\nStarting LLM validation...")
    results = validator.validate_llm(
        analyzer,
        all_test_data,
        save_individual_results=True
    )
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE!")
    print("="*60)
    print(f"\nCheck the '{validator.output_dir}' folder for:")
    print("  - llm_validation_results.json (full metrics)")
    print("  - llm_validation_detailed.csv (individual results)")
    print("  - Visualization plots (PNG files)")