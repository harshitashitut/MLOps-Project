import logging
from typing import Callable, Any

class PipelineError(Exception):
    """Custom exception for pipeline failures"""
    pass

def safe_execute(func: Callable, cleanup_func: Callable = None, *args, **kwargs) -> Any:
    """
    Execute function with error handling and optional cleanup
    
    Args:
        func: Function to execute
        cleanup_func: Optional cleanup function to run on error
        
    Returns:
        Function result
        
    Raises:
        PipelineError: If execution fails
    """
    logger = logging.getLogger(func.__name__)
    
    try:
        logger.info(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Completed {func.__name__}")
        return result
        
    except Exception as e:
        logger.error(f"{func.__name__} failed: {e}", exc_info=True)
        
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
        
        raise PipelineError(f"Pipeline failed at {func.__name__}: {str(e)}")