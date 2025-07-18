#!/usr/bin/env python3
"""
Main Training Pipeline for PII Redaction Model

This script orchestrates the entire training pipeline for the PII redaction model.
It handles dataset creation, model training, ONNX conversion, and inference testing.

Usage:
    python main.py [--config config.yaml] [--stage all|data|train|convert|test]

Stages:
    data: Create synthetic dataset
    train: Train the model
    convert: Convert to ONNX format
    test: Test inference
    all: Run all stages (default)
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset_creation import MultilingualPIIDataset, PIIDataProcessor
from model_training import PIIRedactionModel, train_model
from onnx_conversion import ONNXConverter
from inference import PIIRedactor, benchmark_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Validate configuration
    required_sections = ['model', 'training', 'dataset', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Create output directories
    os.makedirs(config['output']['model_path'], exist_ok=True)
    os.makedirs(config['output']['logs_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['output']['onnx_path']), exist_ok=True)
    
    return config


def create_dataset(config: Dict) -> Dict:
    """
    Create synthetic multilingual PII dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with dataset information
    """
    logger.info("=" * 50)
    logger.info("STAGE 1: DATASET CREATION")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Create dataset creator
    dataset_creator = MultilingualPIIDataset(config)
    
    # Generate dataset
    total_samples = config['dataset']['train_size'] + config['dataset']['val_size'] + config['dataset']['test_size']
    logger.info(f"Creating dataset with {total_samples} total samples")
    
    dataset_dict = dataset_creator.create_dataset(total_samples)
    
    # Save dataset information
    dataset_info = {
        'train_size': len(dataset_dict['train']),
        'validation_size': len(dataset_dict['validation']),
        'test_size': len(dataset_dict['test']),
        'languages': config['dataset']['languages'],
        'hebrew_ratio': config['dataset']['hebrew_ratio'],
        'pii_types': config['pii_types']
    }
    
    # Save dataset to disk
    dataset_path = os.path.join('data', 'processed')
    os.makedirs(dataset_path, exist_ok=True)
    
    dataset_dict.save_to_disk(dataset_path)
    
    # Save dataset info
    with open(os.path.join(dataset_path, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    end_time = time.time()
    logger.info(f"Dataset creation completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Dataset saved to: {dataset_path}")
    
    return {
        'dataset_dict': dataset_dict,
        'dataset_info': dataset_info,
        'dataset_path': dataset_path
    }


def train_pii_model(config: Dict, dataset_dict) -> Dict:
    """
    Train the PII redaction model.
    
    Args:
        config: Configuration dictionary
        dataset_dict: Dataset dictionary from create_dataset
        
    Returns:
        Training results dictionary
    """
    logger.info("=" * 50)
    logger.info("STAGE 2: MODEL TRAINING")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Initialize model
    logger.info("Initializing model...")
    model = PIIRedactionModel(config)
    
    # Process datasets
    logger.info("Processing datasets...")
    processor = PIIDataProcessor(model.tokenizer, config['model']['max_length'])
    
    # Apply tokenization and label alignment
    train_dataset = dataset_dict['train'].map(
        processor.tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset_dict['train'].column_names,
        desc="Processing training data"
    )
    
    val_dataset = dataset_dict['validation'].map(
        processor.tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset_dict['validation'].column_names,
        desc="Processing validation data"
    )
    
    test_dataset = dataset_dict['test'].map(
        processor.tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset_dict['test'].column_names,
        desc="Processing test data"
    )
    
    # Train model
    logger.info("Starting model training...")
    training_results = train_model(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    logger.info(f"Model saved to: {training_results['model_path']}")
    
    # Log final metrics
    logger.info("Final Training Metrics:")
    for metric, value in training_results['eval_metrics'].items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save training summary
    training_summary = {
        'training_time_seconds': training_time,
        'model_path': training_results['model_path'],
        'final_metrics': training_results['eval_metrics'],
        'config': config
    }
    
    summary_path = os.path.join(config['output']['logs_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    
    return training_results


def convert_to_onnx(config: Dict, model_path: str) -> str:
    """
    Convert trained model to ONNX format.
    
    Args:
        config: Configuration dictionary
        model_path: Path to trained model
        
    Returns:
        Path to ONNX model
    """
    logger.info("=" * 50)
    logger.info("STAGE 3: ONNX CONVERSION")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # Initialize converter
    converter = ONNXConverter(
        model_path=model_path,
        onnx_path=config['output']['onnx_path'],
        config=config
    )
    
    # Convert model
    optimize = config.get('onnx', {}).get('optimize', True)
    quantize = config.get('onnx', {}).get('quantize', False)
    
    logger.info(f"Converting model with optimize={optimize}, quantize={quantize}")
    
    onnx_model_path = converter.convert(optimize=optimize, quantize=quantize)
    
    # Test conversion accuracy
    test_texts = [
        "My name is John Doe and my phone is 555-1234",
        "שמי אלון ומספר תעודת הזהות שלי הוא 123456789",
        "Contact me at john.doe@example.com",
        "הכתובת שלי היא רחוב הרצל 15, תל אביב"
    ]
    
    logger.info("Testing conversion accuracy...")
    comparison_results = converter.compare_outputs(model_path, test_texts)
    
    end_time = time.time()
    
    logger.info(f"ONNX conversion completed in {end_time - start_time:.2f} seconds")
    logger.info(f"ONNX model saved to: {onnx_model_path}")
    logger.info(f"Outputs match: {comparison_results['outputs_match']}")
    
    return onnx_model_path


def test_inference(config: Dict, model_path: str, onnx_path: str) -> Dict:
    """
    Test inference with both PyTorch and ONNX models.
    
    Args:
        config: Configuration dictionary
        model_path: Path to PyTorch model
        onnx_path: Path to ONNX model
        
    Returns:
        Test results dictionary
    """
    logger.info("=" * 50)
    logger.info("STAGE 4: INFERENCE TESTING")
    logger.info("=" * 50)
    
    # Test texts in multiple languages
    test_texts = [
        # English
        "My name is John Doe and my SSN is 123-45-6789",
        "Contact me at john.doe@example.com or call (555) 123-4567",
        "I live at 123 Main Street, New York, NY",
        
        # Hebrew
        "שמי אלון כהן ומספר תעודת הזהות שלי הוא 123456789",
        "הטלפון שלי 050-1234567 והמייל alon@example.com",
        "אני גר ברחוב הרצל 15, תל אביב",
        
        # Spanish
        "Mi nombre es María García y mi teléfono es 91-234-5678",
        "Puedes contactarme en maria.garcia@correo.es",
        
        # French
        "Je m'appelle Pierre Dubois et mon numéro est 01-23-45-67-89",
        "Mon adresse email est pierre.dubois@mail.fr",
        
        # German
        "Ich heiße Hans Mueller und meine Telefonnummer ist 030-12345678",
        "Meine E-Mail-Adresse ist hans.mueller@email.de"
    ]
    
    results = {}
    
    # Test PyTorch model
    logger.info("Testing PyTorch model...")
    pytorch_redactor = PIIRedactor(model_path, use_onnx=False)
    
    pytorch_results = []
    for text in test_texts:
        redacted, entities = pytorch_redactor.redact_with_info(text)
        pytorch_results.append({
            'original': text,
            'redacted': redacted,
            'entities': len(entities)
        })
    
    # Test ONNX model
    logger.info("Testing ONNX model...")
    onnx_redactor = PIIRedactor(onnx_path, use_onnx=True)
    
    onnx_results = []
    for text in test_texts:
        redacted, entities = onnx_redactor.redact_with_info(text)
        onnx_results.append({
            'original': text,
            'redacted': redacted,
            'entities': len(entities)
        })
    
    # Benchmark performance
    logger.info("Benchmarking inference performance...")
    
    # Benchmark PyTorch
    pytorch_times = []
    for _ in range(10):
        start_time = time.time()
        for text in test_texts:
            _ = pytorch_redactor.predict(text)
        pytorch_times.append(time.time() - start_time)
    
    # Benchmark ONNX
    onnx_times = []
    for _ in range(10):
        start_time = time.time()
        for text in test_texts:
            _ = onnx_redactor.predict(text)
        onnx_times.append(time.time() - start_time)
    
    # Calculate statistics
    pytorch_avg_time = sum(pytorch_times) / len(pytorch_times)
    onnx_avg_time = sum(onnx_times) / len(onnx_times)
    speedup = pytorch_avg_time / onnx_avg_time
    
    results = {
        'pytorch_results': pytorch_results,
        'onnx_results': onnx_results,
        'performance': {
            'pytorch_avg_time': pytorch_avg_time,
            'onnx_avg_time': onnx_avg_time,
            'speedup': speedup,
            'texts_per_second_pytorch': len(test_texts) / pytorch_avg_time,
            'texts_per_second_onnx': len(test_texts) / onnx_avg_time
        }
    }
    
    # Log results
    logger.info("Inference Test Results:")
    logger.info(f"PyTorch average time: {pytorch_avg_time:.4f}s")
    logger.info(f"ONNX average time: {onnx_avg_time:.4f}s")
    logger.info(f"ONNX speedup: {speedup:.2f}x")
    logger.info(f"PyTorch throughput: {results['performance']['texts_per_second_pytorch']:.2f} texts/sec")
    logger.info(f"ONNX throughput: {results['performance']['texts_per_second_onnx']:.2f} texts/sec")
    
    # Show example redactions
    logger.info("\nExample Redactions:")
    for i in range(min(3, len(test_texts))):
        result = pytorch_results[i]
        logger.info(f"Original: {result['original']}")
        logger.info(f"Redacted: {result['redacted']}")
        logger.info(f"Entities found: {result['entities']}")
        logger.info("-" * 40)
    
    # Save test results
    test_results_path = os.path.join(config['output']['logs_dir'], 'inference_test_results.json')
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    """Main function to orchestrate the training pipeline"""
    parser = argparse.ArgumentParser(description='PII Redaction Model Training Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--stage', choices=['all', 'data', 'train', 'convert', 'test'], 
                       default='all', help='Pipeline stage to run')
    parser.add_argument('--model-path', help='Path to existing model (for convert/test stages)')
    parser.add_argument('--onnx-path', help='Path to existing ONNX model (for test stage)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("=" * 60)
    logger.info("PII REDACTION MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Base model: {config['model']['base_model']}")
    logger.info(f"Target languages: {config['dataset']['languages']}")
    logger.info(f"Hebrew ratio: {config['dataset']['hebrew_ratio']}")
    
    try:
        # Stage 1: Dataset Creation
        dataset_dict = None
        if args.stage in ['all', 'data']:
            dataset_result = create_dataset(config)
            dataset_dict = dataset_result['dataset_dict']
        elif args.stage in ['train']:
            # Only load dataset for training stage
            dataset_path = os.path.join('data', 'processed')
            if os.path.exists(dataset_path):
                from datasets import load_from_disk
                dataset_dict = load_from_disk(dataset_path)
                logger.info(f"Loaded existing dataset from {dataset_path}")
            else:
                logger.error("No existing dataset found. Please run data stage first.")
                return
        
        # Stage 2: Model Training
        if args.stage in ['all', 'train']:
            training_result = train_pii_model(config, dataset_dict)
            model_path = training_result['model_path']
        else:
            model_path = args.model_path or config['output']['model_path']
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                return
        
        # Stage 3: ONNX Conversion
        if args.stage in ['all', 'convert']:
            onnx_path = convert_to_onnx(config, model_path)
        else:
            onnx_path = args.onnx_path or config['output']['onnx_path']
            if not os.path.exists(os.path.join(onnx_path, 'model.onnx')):
                logger.error(f"ONNX model does not exist: {onnx_path}")
                return
        
        # Stage 4: Inference Testing
        if args.stage in ['all', 'test']:
            test_results = test_inference(config, model_path, onnx_path)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        if args.stage == 'all':
            logger.info("All stages completed:")
            logger.info(f"  ✓ Dataset created")
            logger.info(f"  ✓ Model trained and saved to: {model_path}")
            logger.info(f"  ✓ ONNX model converted and saved to: {onnx_path}")
            logger.info(f"  ✓ Inference tested successfully")
        
        logger.info("\nTo use the trained model:")
        logger.info("from src.inference import PIIRedactor")
        logger.info(f"redactor = PIIRedactor('{onnx_path}', use_onnx=True)")
        logger.info("redacted_text = redactor.redact('Your text here')")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
