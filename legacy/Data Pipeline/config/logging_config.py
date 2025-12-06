import logging
import os
from datetime import datetime
from pathlib import Path

def get_logger(name, component='pipeline'):
    """
    Create and configure a logger
    
    Args:
        name: Logger name (usually __name__)
        component: Component name for organizing logs
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent / 'logs' / component
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    log_file = log_dir / f"{component}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class PipelineLogger:
    """Helper class for pipeline-specific logging"""
    
    def log_pipeline_start(self, logger, pipeline_name, config=None):
        """Log pipeline start"""
        logger.info(f"=" * 60)
        logger.info(f"Starting Pipeline: {pipeline_name}")
        if config:
            logger.info(f"Configuration: {config}")
        logger.info(f"=" * 60)
    
    def log_pipeline_end(self, logger, pipeline_name, status="SUCCESS", duration=None):
        """Log pipeline completion"""
        logger.info(f"=" * 60)
        logger.info(f"Pipeline {pipeline_name} completed with status: {status}")
        if duration:
            logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"=" * 60)
    
    def log_data_stats(self, logger, dataset_name, stats):
        """Log dataset statistics"""
        logger.info(f"Dataset: {dataset_name}")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")