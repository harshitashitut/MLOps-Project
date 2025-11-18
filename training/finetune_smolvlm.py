"""
SmolVLM Finetuning Script
Finetunes SmolVLM-500M on labeled pitch deck slides using LoRA.

Usage: python training/finetune_smolvlm.py
"""

import os
import sys
from pathlib import Path
import yaml
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
import wandb
from loguru import logger

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from data_pipeline.dataset import PitchDeckDataset, create_data_splits, collate_fn


class SmolVLMTrainer:
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """Initialize the trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.lora_config = self.config['lora']
        self.training_config = self.config['training']
        self.wandb_config = self.config['wandb']
        
        logger.info("SmolVLM Trainer initialized")
    
    def setup_wandb(self):
        """Initialize Weights & Biases tracking."""
        if self.wandb_config.get('entity'):
            wandb.init(
                project=self.wandb_config['project'],
                entity=self.wandb_config['entity'],
                name=self.wandb_config.get('name'),
                tags=self.wandb_config['tags'],
                config=self.config
            )
            logger.info("W&B initialized")
        else:
            logger.warning("W&B entity not set. Skipping W&B initialization.")
    
    def load_model_and_processor(self):
        """Load the base model and processor."""
        logger.info(f"Loading model: {self.model_config['name']}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_config['name'],
            cache_dir=self.model_config['cache_dir']
        )
        
        # Load model with 8-bit quantization (saves memory)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_config['name'],
            load_in_8bit=self.model_config['load_in_8bit'],
            device_map=self.model_config['device_map'],
            cache_dir=self.model_config['cache_dir'],
            torch_dtype=torch.float16
        )
        
        logger.info("Model loaded successfully")
    
    def setup_lora(self):
        """Configure and apply LoRA to the model."""
        logger.info("Setting up LoRA...")
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_config['r'],  # Rank - lower = fewer parameters
            lora_alpha=self.lora_config['lora_alpha'],  # Scaling factor
            target_modules=self.lora_config['target_modules'],  # Which layers to apply LoRA
            lora_dropout=self.lora_config['lora_dropout'],
            bias=self.lora_config['bias'],
            task_type=self.lora_config['task_type']
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters (should be ~1% of total)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configured successfully")
    
    def prepare_datasets(self):
        """Prepare train, validation, and test datasets."""
        logger.info("Preparing datasets...")
        
        # Create data splits if needed
        labels_file = "../data/labeled/labels.json"
        splits_dir = Path("../data/labeled/splits")
        
        if not splits_dir.exists() or not list(splits_dir.glob("labels_*.json")):
            logger.info("Creating data splits...")
            split_files = create_data_splits(
                labels_file=labels_file,
                train_ratio=self.config['data']['train_split'],
                val_ratio=self.config['data']['val_split'],
                test_ratio=self.config['data']['test_split']
            )
        else:
            split_files = {
                'train': str(splits_dir / "labels_train.json"),
                'val': str(splits_dir / "labels_val.json"),
                'test': str(splits_dir / "labels_test.json")
            }
        
        # Create datasets
        self.train_dataset = PitchDeckDataset(
            data_dir="../data/raw",
            labels_file=split_files['train'],
            processor=self.processor,
            split='train'
        )
        
        self.val_dataset = PitchDeckDataset(
            data_dir="../data/raw",
            labels_file=split_files['val'],
            processor=self.processor,
            split='val'
        )
        
        self.test_dataset = PitchDeckDataset(
            data_dir="../data/raw",
            labels_file=split_files['test'],
            processor=self.processor,
            split='test'
        )
        
        logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def setup_training_args(self) -> TrainingArguments:
        """Configure training arguments."""
        args = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            warmup_steps=self.training_config['warmup_steps'],
            logging_steps=self.training_config['logging_steps'],
            save_steps=self.training_config['save_steps'],
            eval_steps=self.training_config['eval_steps'],
            save_total_limit=self.training_config['save_total_limit'],
            fp16=self.training_config['fp16'],
            gradient_checkpointing=self.training_config['gradient_checkpointing'],
            optim=self.training_config['optim'],
            lr_scheduler_type=self.training_config['lr_scheduler_type'],
            max_grad_norm=self.training_config['max_grad_norm'],
            seed=self.training_config['seed'],
            
            # Evaluation
            eval_strategy="steps",
            
            # Logging
            logging_dir=f"{self.training_config['output_dir']}/logs",
            report_to=["wandb"] if self.wandb_config.get('entity') else ["none"],
            
            # Save best model
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        
        return args
    
    def train(self):
        """Execute the training process."""
        logger.info("Starting training...")
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Setup callbacks
        callbacks = []
        if self.config['early_stopping']['enabled']:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['early_stopping']['patience'],
                    early_stopping_threshold=0.0
                )
            )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=collate_fn,
            callbacks=callbacks
        )
        
        # Train!
        train_result = trainer.train()
        
        # Save final model
        trainer.save_model()
        
        # Log metrics
        metrics = train_result.metrics
        logger.info(f"Training completed. Metrics: {metrics}")
        
        return trainer, metrics
    
    def evaluate(self, trainer):
        """Evaluate the model on test set."""
        logger.info("Evaluating on test set...")
        
        test_results = trainer.evaluate(self.test_dataset)
        logger.info(f"Test results: {test_results}")
        
        return test_results
    
    def run(self):
        """Run the complete training pipeline."""
        try:
            # Setup
            self.setup_wandb()
            self.load_model_and_processor()
            self.setup_lora()
            self.prepare_datasets()
            
            # Train
            trainer, train_metrics = self.train()
            
            # Evaluate
            test_metrics = self.evaluate(trainer)
            
            # Save final artifacts
            output_dir = Path(self.training_config['output_dir'])
            
            # Save processor
            self.processor.save_pretrained(output_dir / "processor")
            
            # Save config
            with open(output_dir / "training_config.yaml", 'w') as f:
                yaml.dump(self.config, f)
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model_path': str(output_dir)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            if wandb.run is not None:
                wandb.finish()


def main():
    """Main execution function."""
    trainer = SmolVLMTrainer()
    results = trainer.run()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Model saved to: {results['model_path']}")
    print(f"\nTrain metrics: {results['train_metrics']}")
    print(f"\nTest metrics: {results['test_metrics']}")


if __name__ == "__main__":
    main()
