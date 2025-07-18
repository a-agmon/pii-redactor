#!/usr/bin/env python3
"""
Setup script for PII Redaction Model Training Pipeline

This script helps with the initial setup and installation of the PII redaction
training pipeline, including dependency installation and environment validation.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def check_gpu_availability():
    """Check if GPU is available for training"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {device_name} ({device_count} devices)")
            return True
        else:
            logger.info("GPU not available, will use CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU availability")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/synthetic",
        "models/checkpoints",
        "models/onnx",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def validate_config():
    """Validate configuration file"""
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Check required sections
        required_sections = ['model', 'training', 'dataset', 'output']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        logger.info("Configuration file validated successfully")
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def run_quick_test():
    """Run a quick test to verify setup"""
    logger.info("Running quick setup verification...")
    
    try:
        # Test imports
        import torch
        import transformers
        import datasets
        import numpy as np
        import pandas as pd
        
        logger.info("All required packages imported successfully")
        
        # Test basic functionality
        from src.dataset_creation import HebrewPIIGenerator
        
        generator = HebrewPIIGenerator()
        test_name = generator.generate_value("NAME")
        logger.info(f"Hebrew PII generator test: {test_name}")
        
        logger.info("Quick test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        return False


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("PII REDACTION MODEL - SETUP SCRIPT")
    logger.info("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Setup failed: Could not install dependencies")
        sys.exit(1)
    
    # Check GPU availability
    check_gpu_availability()
    
    # Validate configuration
    if not validate_config():
        logger.error("Setup failed: Configuration validation failed")
        sys.exit(1)
    
    # Run quick test
    if not run_quick_test():
        logger.error("Setup failed: Quick test failed")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("SETUP COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Review and customize config.yaml if needed")
    logger.info("2. Run the training pipeline: python main.py")
    logger.info("3. Test inference: python test_inference.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()