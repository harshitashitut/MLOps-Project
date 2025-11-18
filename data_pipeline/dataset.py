"""
Custom Dataset for SmolVLM Finetuning
Handles loading and preprocessing of labeled pitch deck slides.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoProcessor
import yaml
from loguru import logger


class PitchDeckDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        processor,
        split: str = "train",
        config_path: str = "config/data_config.yaml"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing images
            labels_file: JSON file with labels
            processor: HuggingFace processor for SmolVLM
            split: 'train', 'val', or 'test'
            config_path: Path to data configuration
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load labels
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Create dataset samples
        self.samples = self._create_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _create_samples(self) -> List[Dict]:
        """Create list of samples with images and labels."""
        samples = []
        
        for img_name, label_data in self.labels.items():
            img_path = self.data_dir / img_name
            
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            # Calculate average score
            scores = [
                label_data['clarity'],
                label_data['design'],
                label_data['data_visualization'],
                label_data['readability'],
                label_data['content_quality']
            ]
            avg_score = sum(scores) / len(scores)
            
            # Create text prompt and target
            prompt = self._create_prompt(label_data)
            target = self._create_target(label_data, avg_score)
            
            samples.append({
                'image_path': str(img_path),
                'prompt': prompt,
                'target': target,
                'labels': label_data,
                'avg_score': avg_score
            })
        
        return samples
    
    def _create_prompt(self, label_data: Dict) -> str:
        """Create instruction prompt for the model."""
        prompt = """Analyze this pitch deck slide and evaluate it on the following criteria:
1. Clarity: Is the message clear and easy to understand?
2. Design: Is the slide visually appealing and professional?
3. Data Visualization: Are charts and data presented effectively?
4. Readability: Is the text size and contrast appropriate?
5. Content Quality: Is the content relevant and valuable?

Provide a rating from 1-5 for each criterion and an overall assessment."""
        
        return prompt
    
    def _create_target(self, label_data: Dict, avg_score: float) -> str:
        """Create target response for the model."""
        # Determine quality level
        if avg_score >= 4.0:
            quality = "excellent"
        elif avg_score >= 3.0:
            quality = "good"
        elif avg_score >= 2.0:
            quality = "fair"
        else:
            quality = "poor"
        
        target = f"""Evaluation:
- Clarity: {label_data['clarity']}/5
- Design: {label_data['design']}/5
- Data Visualization: {label_data['data_visualization']}/5
- Readability: {label_data['readability']}/5
- Content Quality: {label_data['content_quality']}/5

Overall Assessment: This slide has {quality} quality with an average score of {avg_score:.1f}/5.0."""
        
        if label_data.get('notes'):
            target += f"\n\nAdditional Notes: {label_data['notes']}"
        
        return target
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Resize if needed
        max_size = self.config['preprocessing']['resize']
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
        
        # Create conversation format for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample['prompt']}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample['target']}
                ]
            }
        ]
        
        # Apply processor
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Remove batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].squeeze(0)
        
        return inputs


def create_data_splits(
    labels_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, str]:
    """
    Split labeled data into train/val/test sets.
    
    Args:
        labels_file: Path to labels JSON file
        train_ratio: Proportion for training (default 80%)
        val_ratio: Proportion for validation (default 10%)
        test_ratio: Proportion for testing (default 10%)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with paths to split label files
    """
    np.random.seed(seed)
    
    # Load all labels
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Shuffle keys
    keys = list(labels.keys())
    np.random.shuffle(keys)
    
    # Calculate split indices
    n_samples = len(keys)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Create splits
    splits = {
        'train': {k: labels[k] for k in keys[:train_end]},
        'val': {k: labels[k] for k in keys[train_end:val_end]},
        'test': {k: labels[k] for k in keys[val_end:]}
    }
    
    # Save split files
    output_dir = Path(labels_file).parent / "splits"
    output_dir.mkdir(exist_ok=True)
    
    split_files = {}
    for split_name, split_data in splits.items():
        split_file = output_dir / f"labels_{split_name}.json"
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        split_files[split_name] = str(split_file)
        logger.info(f"{split_name}: {len(split_data)} samples")
    
    return split_files


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': input_ids  # For causal LM training
    }


if __name__ == "__main__":
    # Example usage
    print("Creating data splits...")
    splits = create_data_splits(
        labels_file="../data/labeled/labels.json",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print("\nData splits created:")
    for split, path in splits.items():
        print(f"  {split}: {path}")
