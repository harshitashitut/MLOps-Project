"""
Pitch Deck Scraper
Collects pitch deck slides from various sources including SlideShare and manual uploads.

Usage: python data_pipeline/scrape_pitch_decks.py
"""

import os
import time
import requests
from pathlib import Path
from typing import List, Dict
import yaml
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import hashlib
from loguru import logger
from tqdm import tqdm
import json


class PitchDeckScraper:
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize the scraper with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_dir = Path(self.config['storage']['raw_data_dir'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.sources = self.config['sources']
        self.downloaded_hashes = set()
        
        logger.info("PitchDeckScraper initialized")
    
    def setup_driver(self):
        """Setup Selenium WebDriver with Chrome."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run without opening browser
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            logger.info("Please install Chrome and chromedriver")
            logger.info("Windows: choco install chromedriver")
            logger.info("Mac: brew install chromedriver")
            logger.info("Or download from: https://chromedriver.chromium.org/")
            return None
    
    def calculate_image_hash(self, image_path: str) -> str:
        """Calculate hash of image to detect duplicates."""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def download_image(self, url: str, save_path: Path) -> bool:
        """
        Download an image from URL.
        
        Args:
            url: Image URL
            save_path: Where to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify image is valid
            img = Image.open(save_path)
            img.verify()
            
            # Check for duplicates
            img_hash = self.calculate_image_hash(save_path)
            if img_hash in self.downloaded_hashes:
                save_path.unlink()
                return False
            
            self.downloaded_hashes.add(img_hash)
            return True
            
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            if save_path.exists():
                save_path.unlink()
            return False
    
    def scrape_ycombinator(self) -> List[Dict]:
        """
        Scrape pitch decks from Y Combinator companies.
        
        Note: Y Combinator doesn't directly host pitch decks.
        This is a placeholder for manual collection or API integration.
        """
        if not self.sources['ycombinator']['enabled']:
            return []
        
        logger.info("Y Combinator scraping...")
        logger.warning("Y Combinator doesn't directly provide pitch decks")
        logger.info("Recommended: Manually collect YC pitch deck examples and place in manual_uploads/")
        
        return []
    
    def scrape_slideshare(self) -> List[Dict]:
        """
        Scrape pitch decks from SlideShare.
        
        Note: SlideShare has changed their structure. This provides a basic framework.
        You may need to adjust selectors based on current website structure.
        """
        if not self.sources['slideshare']['enabled']:
            return []
        
        logger.info("Scraping SlideShare...")
        downloaded = []
        
        driver = self.setup_driver()
        if not driver:
            logger.warning(" Selenium driver not available, skipping SlideShare scraping")
            return []
        
        try:
            for search_term in self.sources['slideshare']['search_terms']:
                logger.info(f"Searching for: {search_term}")
                
                # Construct search URL
                search_url = f"https://www.slideshare.net/search/slideshow?q={search_term.replace(' ', '+')}"
                
                try:
                    driver.get(search_url)
                    time.sleep(3)  # Wait for page load
                    
                    # Note: SlideShare structure changes frequently
                    # You may need to inspect the page and update selectors
                    logger.warning("SlideShare scraping requires manual selector updates")
                    logger.info("Please inspect SlideShare.net and update CSS selectors")
                    
                except Exception as e:
                    logger.error(f"Error accessing SlideShare: {e}")
                    continue
                
        except Exception as e:
            logger.error(f"SlideShare scraping error: {e}")
        
        finally:
            driver.quit()
        
        logger.info(f" Downloaded {len(downloaded)} slides from SlideShare")
        return downloaded
    
    def scrape_sequoia(self) -> List[Dict]:
        """
        Scrape pitch deck examples from Sequoia Capital.
        
        Sequoia has a famous pitch deck template that can be downloaded.
        """
        if not self.sources['sequoia']['enabled']:
            return []
        
        logger.info("Checking Sequoia Capital resources...")
        logger.info("Sequoia's pitch deck template: https://www.sequoiacap.com/article/writing-a-business-plan/")
        logger.warning("Manual download recommended for Sequoia templates")
        
        return []
    
    def scrape_manual_uploads(self) -> List[Dict]:
        """
        Process manually uploaded pitch decks.
        
        This is the RECOMMENDED method for data collection.
        """
        if not self.sources['manual']['enabled']:
            return []
        
        logger.info("Processing manual uploads...")
        manual_path = Path(self.sources['manual']['path'])
        
        if not manual_path.exists():
            manual_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Manual upload directory created: {manual_path}")
            logger.info("Please add pitch deck images to this folder")
            return []
        
        processed = []
        
        # Supported formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        
        for ext in image_extensions:
            for file in manual_path.glob(ext):
                try:
                    # Copy to raw data directory
                    dest = self.raw_data_dir / file.name
                    
                    if dest.exists():
                        logger.debug(f"Skipping existing file: {file.name}")
                        continue
                    
                    # Verify it's a valid image
                    img = Image.open(file)
                    img.verify()
                    
                    # Check for duplicates
                    import shutil
                    shutil.copy(file, dest)
                    
                    img_hash = self.calculate_image_hash(dest)
                    if img_hash in self.downloaded_hashes:
                        dest.unlink()
                        logger.debug(f"Duplicate detected: {file.name}")
                        continue
                    
                    self.downloaded_hashes.add(img_hash)
                    
                    processed.append({
                        'source': 'manual',
                        'path': str(dest),
                        'original_name': file.name
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to process {file.name}: {e}")
                    continue
        
        logger.info(f"Processed {len(processed)} manual uploads")
        return processed
    
    def download_sample_images(self) -> List[Dict]:
        """
        Download some sample pitch deck images from public sources.
        
        This provides a starting point if you don't have any slides yet.
        """
        logger.info("Downloading sample images...")
        
        # Public domain / CC0 business presentation images
        sample_urls = [
            # These are placeholder URLs - replace with actual public pitch deck images
            # or use manual upload instead
        ]
        
        downloaded = []
        
        for idx, url in enumerate(sample_urls):
            filename = f"sample_{idx:03d}.jpg"
            save_path = self.raw_data_dir / filename
            
            if self.download_image(url, save_path):
                downloaded.append({
                    'source': 'sample',
                    'path': str(save_path),
                    'url': url
                })
        
        if downloaded:
            logger.info(f"Downloaded {len(downloaded)} sample images")
        
        return downloaded
    
    def run(self) -> Dict:
        """Run all scrapers and return summary."""
        logger.info("="*60)
        logger.info("Starting pitch deck collection...")
        logger.info("="*60)
        
        results = {
            'ycombinator': self.scrape_ycombinator(),
            'slideshare': self.scrape_slideshare(),
            'sequoia': self.scrape_sequoia(),
            'manual': self.scrape_manual_uploads()
        }
        
        total = sum(len(v) for v in results.values())
        
        logger.info("="*60)
        logger.info(f" Collection complete! Total slides: {total}")
        logger.info("="*60)
        
        # Save metadata
        metadata_path = self.raw_data_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Main execution function."""
    scraper = PitchDeckScraper()
    results = scraper.run()
    
    print("\n" + "="*60)
    print(" COLLECTION SUMMARY")
    print("="*60)
    
    for source, items in results.items():
        print(f"  {source.capitalize():20s}: {len(items):3d} slides")
    
    total = sum(len(v) for v in results.values())
    print(f"  {'Total':20s}: {total:3d} slides")
    print("="*60)
    
    if total == 0:
        print("\n  No slides collected!")
        print("\n RECOMMENDED: Use manual upload method")
        print("   1. Find pitch deck PDFs online (Y Combinator, SlideShare, etc.)")
        print("   2. Extract slides as individual images")
        print("   3. Place them in: ../data/raw/manual_uploads/")
        print("   4. Run this script again")
    else:
        print(f"\nReady for labeling! Run: python data_pipeline/label_slides.py")


if __name__ == "__main__":
    main()