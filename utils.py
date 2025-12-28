"""
Utility functions for the drone detection system.
"""
import logging
import os
from typing import Optional


def setup_logging(log_level: str = "INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('drone_detection.log')
        ]
    )


def ensure_directory(path: str):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def check_file_exists(file_path: str) -> bool:
    """
    Check if a file exists.
    
    Args:
        file_path: Path to file
    
    Returns:
        bool: True if file exists
    """
    return os.path.isfile(file_path)

