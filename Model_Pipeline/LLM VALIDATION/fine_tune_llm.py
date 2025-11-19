"""
Fine-tune LLM on Interview Evaluation Task
Trains the model on your labeled expert feedback data
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

class LLMFineTuner:
    def __init__(
        self,
        base_model: str = "google/flan-t5-large",
        output_dir: str = "fine_tuned_model",
        use_gpu: bool = True
    ):
        """
        Initialize fine-tuning setup
        
        Args:
            base_model: Base model to fine-tune
            output_dir: Directory to save fine-tuned model
            use_gpu: Whether to use GPU if available
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        elif use_gpu and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Base model: {base_model}")
        print(f"Output directory: {self.output_dir.absolute()}")
        
    def prepare_data_from_validation(
        self,
        validation_csv: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare training data from validation results CSV
        
        Args:
            validation_csv: Path to llm_validation_detailed.csv
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            train_dataset, test_dataset
        """
        print("\nPreparing training data...")
        
        # Load validation results
        df = pd.read_csv(validation_csv)
        print(f"Loaded {len(df)} examples")
        
        # Create training examples
        training_examples = []
        
        for idx, row in df.iterrows():
            # Create input text
            input_text = self._format_input(
                row['question'],
                row['answer']
            )
            
            # Create target text (expert feedback + scores)
            target_text = self._format_target(
                row['expert_feedback'],
                eval(row['expert_scores'])  # Convert string to dict
            )
            
            training_examples.append({
                'input': input_text,
                'target': target_text,
                'question': row['question'],
                'answer': row['answer']
            })
        
        # Split into train and test
        train_examples, test_examples = train_test_split(
            training_examples,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"Split: {len(train_examples)} training, {len(test_examples)} testing")
        
        # Convert to Hugging Face Dataset
        train_dataset = Dataset.from_dict({
            'input': [ex['input'] for ex in train_examples],
            'target': [ex['target'] for ex in train_examples]
        })
        
        test_dataset = Dataset.from_dict({
            'input': [ex['input'] for ex in test_examples],
            'target': [ex['target'] for ex in test_examples]
        })
        
        return train_dataset, test_dataset
    
    def _format_input(self, question: str, answer: str) -> str:
        """Format input for the model"""
        prompt = f"""Evaluate this interview answer and provide detailed feedback with numerical scores from 1 to 10.

Question: {question}

Answer: {answer}

Provide your evaluation with detailed feedback explaining the strengths and weaknesses, followed by numerical scores.
Format your scores exactly like this example: Structure: 8/10 | Clarity: 7/10 | Relevance: 9/10 | Overall: 8/10
"""
        return prompt
    
    def _format_target(self, feedback: str, scores: Dict) -> str:
        """Format target output (expert feedback + scores)"""
        structure = scores.get('structure', scores['overall'])
        clarity = scores.get('clarity', scores['overall'])
        relevance = scores.get('relevance', scores['overall'])
        overall = scores['overall']
        
        target = f"""{feedback}

Structure: {structure}/10 | Clarity: {clarity}/10 | Relevance: {relevance}/10 | Overall: {overall}/10"""
        
        return target
    
    def tokenize_data(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        max_input_length: int = 512,
        max_target_length: int = 256
    ) -> Tuple[Dataset, Dataset]:
        """
        Tokenize the datasets
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            
        Returns:
            Tokenized train and test datasets
        """
        print("\nTokenizing data...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        def tokenize_function(examples):
            # Tokenize inputs
            model_inputs = self.tokenizer(
                examples['input'],
                max_length=max_input_length,
                truncation=True,
                padding=False
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                examples['target'],
                max_length=max_target_length,
                truncation=True,
                padding=False
            )
            
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        
        # Tokenize datasets
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=test_dataset.column_names
        )
        
        print(f"Tokenization complete!")
        return tokenized_train, tokenized_test
    
    def train(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        gradient_accumulation_steps: int = 4
    ):
        """
        Fine-tune the model
        
        Args:
            train_dataset: Tokenized training dataset
            test_dataset: Tokenized test dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            logging_steps: Log metrics every N steps
            gradient_accumulation_steps: Accumulate gradients over N steps
        """
        print("\n" + "="*60)
        print("STARTING FINE-TUNING")
        print("="*60)
        
        # Load base model
        print(f"\nLoading base model: {self.base_model}")
        model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
        
        # Move to device
        if self.device == "cuda":
            model = model.to(self.device)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            eval_strategy="steps",  # Fixed: was evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            fp16=self.device == "cuda",  # Use mixed precision on GPU
            report_to="none",  # Disable wandb
            logging_dir=str(self.output_dir / "logs"),
            remove_unused_columns=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train
        print("\nStarting training...")
        print(f"Total training examples: {len(train_dataset)}")
        print(f"Total test examples: {len(test_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Warmup steps: {warmup_steps}")
        print()
        
        trainer.train()
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model(str(self.output_dir / "final_model"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final_model"))
        
        print(f"\n{'='*60}")
        print("FINE-TUNING COMPLETE!")
        print(f"{'='*60}")
        print(f"\nModel saved to: {self.output_dir / 'final_model'}")
        print("\nTo use the fine-tuned model:")
        print(f"  analyzer.load_llm_model('{self.output_dir / 'final_model'}')")
        
    def evaluate_sample(self, model_path: str, question: str, answer: str):
        """
        Test the fine-tuned model on a sample
        
        Args:
            model_path: Path to fine-tuned model
            question: Interview question
            answer: Candidate answer
        """
        print("\n" + "="*60)
        print("TESTING FINE-TUNED MODEL")
        print("="*60)
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        if self.device == "cuda":
            model = model.to(self.device)
        
        # Format input
        input_text = self._format_input(question, answer)
        
        # Generate feedback
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
        
        feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nQuestion: {question}")
        print(f"\nAnswer: {answer[:200]}...")
        print(f"\nGenerated Feedback:\n{feedback}")
        print("\n" + "="*60)


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("LLM FINE-TUNING SCRIPT")
    print("="*60)
    
    # Initialize fine-tuner
    finetuner = LLMFineTuner(
        base_model="google/flan-t5-base",  # Using smaller model to avoid memory issues
        output_dir="fine_tuned_interview_model",
        use_gpu=True
    )
    
    # Prepare data from validation results
    print("\nStep 1: Preparing training data...")
    train_dataset, test_dataset = finetuner.prepare_data_from_validation(
        validation_csv="llm_validation_results/llm_validation_detailed.csv",
        test_size=0.2,  # 20% for testing
        random_state=42
    )
    
    # Tokenize data
    print("\nStep 2: Tokenizing data...")
    train_tokenized, test_tokenized = finetuner.tokenize_data(
        train_dataset,
        test_dataset,
        max_input_length=512,
        max_target_length=256
    )
    
    # Train model
    print("\nStep 3: Fine-tuning model...")
    finetuner.train(
        train_tokenized,
        test_tokenized,
        num_epochs=3,              # Adjust based on dataset size
        batch_size=4,              # Reduce if out of memory
        learning_rate=5e-5,        # Standard for fine-tuning
        warmup_steps=100,
        save_steps=100,
        eval_steps=100,
        logging_steps=50,
        gradient_accumulation_steps=4  # Effective batch size = 16
    )
    
    # Test on a sample
    print("\nStep 4: Testing on sample...")
    test_question = "What is regularization in machine learning?"
    test_answer = "Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function."
    
    finetuner.evaluate_sample(
        model_path="fine_tuned_interview_model/final_model",
        question=test_question,
        answer=test_answer
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update main1.py to use fine-tuned model")
    print("2. Run validation again: python3 llm_validation.py")
    print("3. Compare results with baseline")
    print("\nTo use fine-tuned model in your code:")
    print("  analyzer.load_llm_model('fine_tuned_interview_model/final_model')")


if __name__ == "__main__":
    main()
