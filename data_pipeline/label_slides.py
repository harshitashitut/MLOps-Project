"""
Slide Labeling Interface
Provides a Gradio-based interface for labeling pitch deck slides.

Usage: python data_pipeline/label_slides.py
Opens at http://localhost:7860
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import yaml
import gradio as gr
from PIL import Image
from loguru import logger
import pandas as pd


class SlideLabelingInterface:
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize the labeling interface."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
      # Get paths from config
        self.raw_data_dir = Path(self.config['storage']['raw_data_dir'])
        self.labeled_data_dir = Path(self.config['storage']['labeled_data_dir'])
        self.labeled_data_dir.mkdir(parents=True, exist_ok=True)
    
        self.categories = self.config['labeling']['categories']
    
    # Load existing labels - THIS LINE MUST COME BEFORE _get_unlabeled_images()
        self.labels_file = self.labeled_data_dir / "labels.json"
        self.labels = self._load_labels()
    
        self.current_index = 0
    
    # Get list of images to label - THIS COMES AFTER labels_file is defined
        self.image_files = self._get_unlabeled_images()
    
        logger.info(f"Labeling interface initialized with {len(self.image_files)} images")

    
    def _get_unlabeled_images(self) -> List[Path]:
        """Get list of images that haven't been labeled yet."""
        # Find all images
        all_images = list(self.raw_data_dir.glob('*.jpg')) + \
                     list(self.raw_data_dir.glob('*.jpeg')) + \
                     list(self.raw_data_dir.glob('*.png'))
        
        # Filter out already labeled
        labeled_images = set()
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                existing_labels = json.load(f)
                labeled_images = set(existing_labels.keys())
        
        unlabeled = [img for img in all_images if img.name not in labeled_images]
        return sorted(unlabeled)
    
    def _load_labels(self) -> Dict:
        """Load existing labels from file."""
        if self.labels_file.exists():
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_labels(self):
        """Save labels to file."""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        logger.info(f"Labels saved: {len(self.labels)} images labeled")
    
    def get_current_image(self):
        """Get the current image to label."""
        if self.current_index >= len(self.image_files):
            return None, "All images labeled!"
        
        img_path = self.image_files[self.current_index]
        img = Image.open(img_path)
        progress = f"Image {self.current_index + 1} of {len(self.image_files)}"
        return img, progress
    
    def save_label(self, clarity, design, data_viz, readability, content_quality, notes):
        """Save the label for current image and move to next."""
        if self.current_index >= len(self.image_files):
            return None, "All images labeled!", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
        img_path = self.image_files[self.current_index]
        
        # Save label
        self.labels[img_path.name] = {
            'clarity': clarity,
            'design': design,
            'data_visualization': data_viz,
            'readability': readability,
            'content_quality': content_quality,
            'notes': notes,
            'image_path': str(img_path)
        }
        
        self._save_labels()
        
        # Move to next image
        self.current_index += 1
        
        if self.current_index >= len(self.image_files):
            return None, "All images labeled!", gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value="")
        
        # Load next image
        img, progress = self.get_current_image()
        
        # Reset sliders to middle value (3)
        return img, progress, gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value="")
    
    def skip_image(self):
        """Skip the current image without labeling."""
        self.current_index += 1
        
        if self.current_index >= len(self.image_files):
            return None, "All images processed!", gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value="")
        
        img, progress = self.get_current_image()
        return img, progress, gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value=3), gr.update(value="")
    
    def export_labels_csv(self):
        """Export labels to CSV for analysis."""
        if not self.labels:
            return "No labels to export"
        
        df = pd.DataFrame.from_dict(self.labels, orient='index')
        csv_path = self.labeled_data_dir / "labels.csv"
        df.to_csv(csv_path)
        
        return f"Exported {len(self.labels)} labels to {csv_path}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Pitch Deck Slide Labeling", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Pitch Deck Slide Labeling Interface")
            gr.Markdown("Rate each slide on a scale of 1-5 for the following criteria:")
            
            with gr.Row():
                with gr.Column(scale=2):
                    image_display = gr.Image(label="Current Slide", type="pil")
                    progress_text = gr.Textbox(label="Progress", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("### Rating Criteria")
                    
                    clarity_slider = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="Clarity (1=Confusing, 5=Very Clear)"
                    )
                    
                    design_slider = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="Design (1=Poor, 5=Excellent)"
                    )
                    
                    data_viz_slider = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="Data Visualization (1=Ineffective, 5=Highly Effective)"
                    )
                    
                    readability_slider = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="Readability (1=Hard to Read, 5=Easy to Read)"
                    )
                    
                    content_quality_slider = gr.Slider(
                        minimum=1, maximum=5, step=1, value=3,
                        label="Content Quality (1=Poor, 5=Excellent)"
                    )
                    
                    notes_input = gr.Textbox(
                        label="Additional Notes (optional)",
                        placeholder="Any observations or comments...",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button(" Submit & Next", variant="primary", size="lg")
                        skip_btn = gr.Button("Skip", size="lg")
                    
                    gr.Markdown("---")
                    export_btn = gr.Button(" Export Labels to CSV")
                    export_status = gr.Textbox(label="Export Status", interactive=False)
            
            # Load first image on interface load
            interface.load(
                fn=self.get_current_image,
                inputs=[],
                outputs=[image_display, progress_text]
            )
            
            # Submit button action
            submit_btn.click(
                fn=self.save_label,
                inputs=[
                    clarity_slider,
                    design_slider,
                    data_viz_slider,
                    readability_slider,
                    content_quality_slider,
                    notes_input
                ],
                outputs=[
                    image_display,
                    progress_text,
                    clarity_slider,
                    design_slider,
                    data_viz_slider,
                    readability_slider,
                    content_quality_slider,
                    notes_input
                ]
            )
            
            # Skip button action
            skip_btn.click(
                fn=self.skip_image,
                inputs=[],
                outputs=[
                    image_display,
                    progress_text,
                    clarity_slider,
                    design_slider,
                    data_viz_slider,
                    readability_slider,
                    content_quality_slider,
                    notes_input
                ]
            )
            
            # Export button action
            export_btn.click(
                fn=self.export_labels_csv,
                inputs=[],
                outputs=[export_status]
            )
        
        return interface


def main():
    """Main execution function."""
    labeler = SlideLabelingInterface()
    interface = labeler.create_interface()
    interface.launch(share=False, server_port=7860)


if __name__ == "__main__":
    main()
    