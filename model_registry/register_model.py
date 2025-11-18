"""
Model Registry Module
Handles model versioning and registration to HuggingFace Hub.

Usage: python model_registry/register_model.py --model_path ./outputs --token YOUR_HF_TOKEN
"""

import sys
from pathlib import Path
import yaml
import json
from datetime import datetime
from huggingface_hub import HfApi, create_repo, upload_folder
from loguru import logger
import argparse


class ModelRegistry:
    def __init__(
        self,
        model_path: str,
        config_path: str = "config/training_config.yaml"
    ):
        """
        Initialize the model registry.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to training configuration
        """
        self.model_path = Path(model_path)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.registry_config = self.config['registry']
        self.api = HfApi()
        
        logger.info("Model Registry initialized")
    
    def generate_version(self) -> str:
        """
        Generate semantic version for the model.
        
        Returns:
            Version string (e.g., "v1.0.0")
        """
        # Check if version file exists
        version_file = Path("model_registry/version.json")
        
        if version_file.exists():
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            
            # Increment patch version
            major, minor, patch = map(int, version_data['version'].strip('v').split('.'))
            patch += 1
            version = f"v{major}.{minor}.{patch}"
        else:
            # First version
            version = "v1.0.0"
            version_data = {'history': []}
        
        # Update version file
        version_data['version'] = version
        version_data['history'].append({
            'version': version,
            'date': datetime.now().isoformat(),
            'model_path': str(self.model_path)
        })
        
        version_file.parent.mkdir(parents=True, exist_ok=True)
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        logger.info(f"Generated version: {version}")
        return version
    
    def create_model_card(self, version: str, metrics: dict) -> str:
        """
        Create a model card for HuggingFace.
        
        Args:
            version: Model version
            metrics: Training and validation metrics
            
        Returns:
            Model card content in Markdown
        """
        template = f"""---
tags:
- vision
- multimodal
- pitch-deck
- smolvlm
- lora
license: apache-2.0
---

# PitchQuest SmolVLM - {version}

This model is a finetuned version of SmolVLM-500M for analyzing pitch deck slides.

## Model Details

- **Base Model:** HuggingFaceTB/SmolVLM-500M-Instruct
- **Finetuning Method:** LoRA (Low-Rank Adaptation)
- **Version:** {version}
- **Training Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Project:** PitchQuest Multimodal

## Training Details

### Training Hyperparameters

- **Learning Rate:** {self.config['training']['learning_rate']}
- **Batch Size:** {self.config['training']['per_device_train_batch_size']}
- **Gradient Accumulation:** {self.config['training']['gradient_accumulation_steps']}
- **Epochs:** {self.config['training']['num_train_epochs']}
- **LoRA r:** {self.config['lora']['r']}
- **LoRA alpha:** {self.config['lora']['lora_alpha']}
- **Optimizer:** {self.config['training']['optim']}

### Performance Metrics

"""
        
        if metrics:
            template += "#### Validation Results\n\n"
            template += "| Metric | Value |\n"
            template += "|--------|-------|\n"
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'mae' in key.lower():
                        template += f"| {key} | {value:.4f} |\n"
                    elif 'agreement' in key.lower() or 'within' in key.lower():
                        template += f"| {key} | {value:.2%} |\n"
                    else:
                        template += f"| {key} | {value:.4f} |\n"
                else:
                    template += f"| {key} | {value} |\n"
        
        template += """

## Intended Use

This model evaluates pitch deck slides across five dimensions:

1. **Clarity:** Message clarity and comprehension
2. **Design:** Visual appeal and professionalism
3. **Data Visualization:** Effectiveness of charts and data presentation
4. **Readability:** Text legibility and contrast
5. **Content Quality:** Relevance and value of content

Each criterion is rated on a scale of 1-5.

## Usage
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image

# Load model
base_model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct"
)
model = PeftModel.from_pretrained(
    base_model, 
    "your-username/pitchquest-smolvlm"
)
processor = AutoProcessor.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct"
)

# Load and analyze slide
image = Image.open("slide.jpg")
prompt = \"\"\"Analyze this pitch deck slide and evaluate it on the following criteria:
1. Clarity: Is the message clear and easy to understand?
2. Design: Is the slide visually appealing and professional?
3. Data Visualization: Are charts and data presented effectively?
4. Readability: Is the text size and contrast appropriate?
5. Content Quality: Is the content relevant and valuable?

Provide a rating from 1-5 for each criterion.\"\"\"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    }
]

text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=text, images=[image], return_tensors="pt")

# Generate
outputs = model.generate(**inputs, max_new_tokens=256)
result = processor.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## Model Performance

- **Human Agreement:** Target >85%
- **Mean Absolute Error:** Target <0.6
- **Predictions Within 1 Point:** Target >85%

## Hardware Requirements

### Training
- **Minimum VRAM:** 16GB
- **Recommended GPU:** NVIDIA T4, A10, or better
- **Alternative:** Google Colab T4 (free tier)

### Inference
- **Minimum VRAM:** 8GB
- **Recommended VRAM:** 12GB
- **CPU Fallback:** Supported (slower)

## Limitations

- Trained on ~500-1000 pitch deck slides
- May not generalize well to non-standard slide formats
- Performance may vary with different design styles
- Best suited for business/startup pitch decks

## Training Data

The model was trained on labeled pitch deck slides from various sources:
- Y Combinator Demo Day presentations
- SlideShare pitch examples
- Sequoia Capital templates
- Manually curated examples

Data was labeled by human annotators on five quality dimensions.

## Citation
```bibtex
@misc{pitchquest-smolvlm,
  title={{PitchQuest SmolVLM: Finetuned Vision-Language Model for Pitch Deck Analysis}},
  author={{Harshita Shitut, Mohit Kakda, Muhammad Salman, Sachin Muttu Baraddi, Uttapreksha Patel}},
  year={{2024}},
  note={{MLOps Project}}
}
```

## Team

- Harshita Shitut
- Mohit Kakda
- Muhammad Salman
- Sachin Muttu Baraddi
- Uttapreksha Patel

## License

Apache 2.0

## Links

- **Project Repository:** [GitHub](https://github.com/harshitashitut/MLOps-Project)
- **Base Model:** [SmolVLM-500M](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)

---

*Generated automatically by PitchQuest Model Registry*
"""
        
        return template
    
    def register_to_huggingface(
        self,
        version: str,
        metrics: dict = None,
        token: str = None
    ) -> str:
        """
        Upload model to HuggingFace Hub.
        
        Args:
            version: Model version
            metrics: Training metrics
            token: HuggingFace API token
            
        Returns:
            Repository URL
        """
        repo_id = self.registry_config['huggingface_repo']
        
        if not repo_id or repo_id == "your-username/pitchquest-smolvlm":
            logger.error("Please set a valid HuggingFace repository ID in config/training_config.yaml")
            logger.error("   Change 'huggingface_repo' to 'your-username/pitchquest-smolvlm'")
            return None
        
        logger.info(f"Registering model to {repo_id}...")
        
        try:
            # Create repository if it doesn't exist
            try:
                create_repo(repo_id, token=token, exist_ok=True)
                logger.info(f"Repository {repo_id} ready")
            except Exception as e:
                logger.warning(f"Could not create repo: {e}")
            
            # Create model card
            model_card = self.create_model_card(version, metrics)
            readme_path = self.model_path / "README.md"
            with open(readme_path, 'w') as f:
                f.write(model_card)
            logger.info("Model card created")
            
            # Upload model
            logger.info("Uploading model files...")
            url = upload_folder(
                folder_path=str(self.model_path),
                repo_id=repo_id,
                token=token,
                commit_message=f"Upload {version}",
                revision=version
            )
            
            logger.info(f"Model registered successfully!")
            logger.info(f"URL: {url}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def register(self, validation_results: dict = None, token: str = None) -> dict:
        """
        Complete model registration pipeline.
        
        Args:
            validation_results: Results from model validation
            token: HuggingFace API token
            
        Returns:
            Registration details
        """
        # Check if auto-registration is enabled
        if not self.registry_config['auto_register']:
            logger.info(" Auto-registration is disabled")
            return {'status': 'skipped', 'reason': 'auto_register disabled'}
        
        # Check validation threshold
        if validation_results:
            human_agreement = validation_results.get('human_agreement', 0)
            threshold = self.registry_config['min_accuracy_threshold']
            
            if human_agreement < threshold:
                logger.warning(f"Model does not meet threshold ({human_agreement:.2%} < {threshold:.2%})")
                return {
                    'status': 'failed',
                    'reason': 'below_threshold',
                    'human_agreement': human_agreement,
                    'threshold': threshold
                }
        
        # Generate version
        version = self.generate_version()
        
        # Register to HuggingFace
        metrics = validation_results.get('metrics', {}) if validation_results else {}
        url = self.register_to_huggingface(version, metrics, token)
        
        result = {
            'status': 'success' if url else 'failed',
            'version': version,
            'url': url,
            'timestamp': datetime.now().isoformat()
        }
        
        return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Register model to HuggingFace Hub")
    parser.add_argument(
        '--model_path',
        type=str,
        default='./outputs',
        help='Path to the trained model'
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API token'
    )
    parser.add_argument(
        '--validation_file',
        type=str,
        default=None,
        help='Path to validation results JSON'
    )
    
    args = parser.parse_args()
    
    # Load validation results if provided
    validation_results = None
    if args.validation_file and Path(args.validation_file).exists():
        with open(args.validation_file, 'r') as f:
            validation_results = json.load(f)
    
    # Register model
    registry = ModelRegistry(model_path=args.model_path)
    results = registry.register(
        validation_results=validation_results,
        token=args.token
    )
    
    print("\n" + "="*60)
    print("MODEL REGISTRATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    print("="*60)
    
    if results['status'] == 'success':
        print(f"\nSUCCESS! Model registered as {results['version']}")
        print(f"🔗 View at: {results['url']}")
    else:
        print(f"\nRegistration {results['status']}: {results.get('reason', 'unknown')}")
    
    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'success' else 1)


if __name__ == "__main__":
    main()
