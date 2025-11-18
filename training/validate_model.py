"""
Model Validation Script
Validates the finetuned model against quality thresholds and human agreement.

Usage: python training/validate_model.py --model_path ./outputs
"""

import sys
from pathlib import Path
import yaml
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
from loguru import logger
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).parent.parent))
from data_pipeline.dataset import PitchDeckDataset


class ModelValidator:
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/training_config.yaml"
    ):
        """
        Initialize the model validator.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to training configuration
        """
        self.model_path = Path(model_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.threshold = self.config['registry']['min_accuracy_threshold']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Model Validator initialized for {model_path}")
    
    def load_model(self):
        """Load the finetuned model."""
        logger.info("Loading model...")
        
        # Load processor
        processor_path = self.model_path / "processor"
        if processor_path.exists():
            self.processor = AutoProcessor.from_pretrained(processor_path)
        else:
            # Fallback to base model processor
            self.processor = AutoProcessor.from_pretrained(
                self.config['model']['name']
            )
        
        # Load base model
        base_model = AutoModelForVision2Seq.from_pretrained(
            self.config['model']['name'],
            load_in_8bit=False,
            torch_dtype=torch.float16
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            base_model,
            self.model_path,
            torch_dtype=torch.float16
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("✅ Model loaded successfully")
    
    def predict(self, image_path: str, prompt: str) -> str:
        """
        Generate prediction for a single image.
        
        Args:
            image_path: Path to the image
            prompt: Input prompt
            
        Returns:
            Generated text response
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        # Decode
        generated_text = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return generated_text
    
    def extract_scores_from_text(self, text: str) -> dict:
        """
        Extract numeric scores from generated text.
        
        Args:
            text: Generated text containing scores
            
        Returns:
            Dictionary of extracted scores
        """
        scores = {
            'clarity': None,
            'design': None,
            'data_visualization': None,
            'readability': None,
            'content_quality': None
        }
        
        # Simple regex-based extraction
        import re
        
        for key in scores.keys():
            # Look for patterns like "Clarity: 4/5" or "Clarity: 4"
            pattern = rf"{key.replace('_', ' ')}:?\s*(\d)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scores[key] = int(match.group(1))
        
        return scores
    
    def validate_on_dataset(self, test_dataset: PitchDeckDataset) -> dict:
        """
        Validate model on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Validation metrics
        """
        logger.info(f"Validating on {len(test_dataset)} test samples...")
        
        predictions = []
        ground_truths = []
        
        for i in tqdm(range(len(test_dataset)), desc="Validating"):
            sample = test_dataset.samples[i]
            
            # Get prediction
            pred_text = self.predict(
                sample['image_path'],
                sample['prompt']
            )
            
            # Extract scores
            pred_scores = self.extract_scores_from_text(pred_text)
            
            # Get ground truth
            gt_scores = sample['labels']
            
            predictions.append(pred_scores)
            ground_truths.append(gt_scores)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truths)
        
        return metrics
    
    def _calculate_metrics(self, predictions: list, ground_truths: list) -> dict:
        """Calculate validation metrics."""
        metrics = {}
        
        categories = ['clarity', 'design', 'data_visualization', 'readability', 'content_quality']
        
        # Per-category metrics
        for cat in categories:
            pred_vals = [p[cat] for p in predictions if p[cat] is not None]
            gt_vals = [gt[cat] for gt, p in zip(ground_truths, predictions) if p[cat] is not None]
            
            if len(pred_vals) > 0:
                # Mean Absolute Error
                mae = mean_absolute_error(gt_vals, pred_vals)
                
                # Accuracy (within 1 point)
                within_1 = sum(abs(p - g) <= 1 for p, g in zip(pred_vals, gt_vals)) / len(pred_vals)
                
                metrics[f'{cat}_mae'] = mae
                metrics[f'{cat}_within_1'] = within_1
        
        # Overall metrics
        all_preds = []
        all_gts = []
        
        for pred, gt in zip(predictions, ground_truths):
            for cat in categories:
                if pred[cat] is not None:
                    all_preds.append(pred[cat])
                    all_gts.append(gt[cat])
        
        if len(all_preds) > 0:
            metrics['overall_mae'] = mean_absolute_error(all_gts, all_preds)
            metrics['overall_within_1'] = sum(abs(p - g) <= 1 for p, g in zip(all_preds, all_gts)) / len(all_preds)
            
            # Agreement with humans (within 1 point)
            metrics['human_agreement'] = metrics['overall_within_1']
        
        return metrics
    
    def validate(self) -> dict:
        """
        Run complete validation pipeline.
        
        Returns:
            Validation results including pass/fail status
        """
        # Load model
        self.load_model()
        
        # Load test dataset
        test_labels = "../data/labeled/splits/labels_test.json"
        test_dataset = PitchDeckDataset(
            data_dir="../data/raw",
            labels_file=test_labels,
            processor=self.processor,
            split='test'
        )
        
        # Validate
        metrics = self.validate_on_dataset(test_dataset)
        
        # Check if model passes threshold
        human_agreement = metrics.get('human_agreement', 0)
        passes_threshold = human_agreement >= self.threshold
        
        results = {
            'metrics': metrics,
            'threshold': self.threshold,
            'passes_threshold': passes_threshold,
            'human_agreement': human_agreement
        }
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("VALIDATION RESULTS")
        logger.info("="*60)
        logger.info(f"Human Agreement: {human_agreement:.2%}")
        logger.info(f"Threshold: {self.threshold:.2%}")
        logger.info(f"Status: {'PASS' if passes_threshold else 'FAIL'}")
        logger.info("="*60)
        
        for key, value in metrics.items():
            if 'mae' in key:
                logger.info(f"  {key}: {value:.3f}")
            elif 'within_1' in key:
                logger.info(f"  {key}: {value:.2%}")
        
        # Save results
        results_file = self.model_path / "validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n Results saved to {results_file}")
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Validate finetuned model")
    parser.add_argument(
        '--model_path',
        type=str,
        default='./outputs',
        help='Path to the trained model'
    )
    
    args = parser.parse_args()
    
    validator = ModelValidator(model_path=args.model_path)
    results = validator.validate()
    
    # Print summary
    print("\n" + "="*60)
    if results['passes_threshold']:
        print(" MODEL VALIDATION PASSED!")
        print(f"   Human agreement: {results['human_agreement']:.1%}")
        print("   Ready for registration and deployment")
    else:
        print(" MODEL VALIDATION FAILED")
        print(f"   Human agreement: {results['human_agreement']:.1%}")
        print(f"   Required: {results['threshold']:.1%}")
        print("   Consider: More training, more data, or adjust hyperparameters")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if results['passes_threshold'] else 1)


if __name__ == "__main__":
    main()