import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import get_logger, PipelineLogger

# Create logger
logger = get_logger(__name__, component='pipeline')
pipeline_logger = PipelineLogger()

#print("\n" + "="*60)
#print(" PITCHQUEST LOGGING TEST")
#print("="*60 + "\n")

# Test basic logging
logger.info(" Logging system initialized")
logger.info("Testing INFO level")
logger.warning("  Testing WARNING level")
logger.error(" Testing ERROR level")

# Test pipeline logging
pipeline_logger.log_pipeline_start(
    logger,
    "Test Pipeline",
    config={'test_mode': True, 'version': '1.0'}
)

logger.info("Processing data...")
logger.info("Running models...")
logger.info("Generating results...")

pipeline_logger.log_pipeline_end(
    logger,
    "Test Pipeline",
    status="SUCCESS",
    duration=5.0
)

# Test stats logging
pipeline_logger.log_data_stats(
    logger,
    "Sample Dataset",
    {
        'total_records': 1000,
        'missing_values': 12,
        'columns': 25
    }
)

print(" TEST COMPLETE!")
