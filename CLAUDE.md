# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Training Pipeline
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete training pipeline
python main.py

# Run specific stages
python main.py --stage data      # Generate synthetic dataset only
python main.py --stage train     # Train model only  
python main.py --stage convert   # Convert to ONNX only
python main.py --stage test      # Test inference only

# Custom configuration
python main.py --config custom_config.yaml
```

### Simple Inference (Minimal Dependencies)
```bash
# Install minimal dependencies for deployment
pip install -r requirements_minimal.txt

# Test simple inference
python simple_inference.py "Text with PII to redact"

# Use in Python code
from simple_inference import simple_pii_redact
result = simple_pii_redact("My email is john@example.com")
```

### Testing and Evaluation
```bash
# Comprehensive inference testing
python test_inference.py

# Interactive testing mode
python test_inference.py --interactive

# Specific test suites
python test_inference.py --test-suite basic
python test_inference.py --test-suite performance
```

## Architecture Overview

### Multi-Stage Pipeline
The project follows a 4-stage pipeline architecture:

1. **Dataset Creation** (`src/dataset_creation.py`): Generates synthetic multilingual PII data
2. **Model Training** (`src/model_training.py`): Fine-tunes multilingual BERT for token classification  
3. **ONNX Conversion** (`src/onnx_conversion.py`): Optimizes model for production deployment
4. **Inference** (`src/inference.py`): Provides PyTorch and ONNX inference capabilities

### Core Components

**PIIRedactionModel** (`src/model_training.py`):
- Wraps AutoModelForTokenClassification with 3 labels: O, B-PII, I-PII
- Handles device placement (CPU forced on MPS to avoid compatibility issues)
- Includes custom error analysis that temporarily moves model to CPU

**MultilingualPIIDataset** (`src/dataset_creation.py`):
- Generates synthetic data for 5 languages (en, he, es, fr, de)
- Hebrew-specific generators with RTL text handling and Israeli ID validation
- BIO tagging with subword token alignment

**ONNXConverter** (`src/onnx_conversion.py`):
- Handles LayerNormalization domain version compatibility issues
- Uses opset version 13 for better compatibility
- Validates models with graceful handling of ONNX checker warnings

### Key Design Patterns

**Device Management**: 
- MPS detection with CPU fallback to avoid "Placeholder storage not allocated" errors
- Model temporarily moved to CPU during error analysis phase

**Multilingual Support**:
- Hebrew language prioritized with dedicated generators
- Configurable language ratios via `config.yaml`
- RTL text handling and Hebrew date formats

**Stage-based Execution**:
- Each stage can run independently if outputs from previous stages exist
- Dataset loading only occurs when needed (train stage), not for convert/test stages

## Configuration

**config.yaml** controls all pipeline aspects:
- Model: base model, sequence length, tokenization settings
- Training: hyperparameters, evaluation strategy, checkpointing  
- Dataset: sizes, language ratios, PII types
- ONNX: optimization settings, opset version

**Key Settings**:
- `hebrew_ratio: 0.4` - 40% Hebrew data in training set
- `label_all_tokens: false` - Only label first subword of each token
- `opset_version: 13` - ONNX compatibility (not 14 due to LayerNormalization issues)

## Critical Implementation Details

**MPS Compatibility**: 
- Device selection logic forces CPU when MPS detected (`src/model_training.py:90-95`)
- Error analysis function moves model to CPU temporarily (`src/model_training.py:376-420`)

**ONNX Conversion Issues**:
- LayerNormalization domain version warnings are expected and handled gracefully
- JSON serialization requires numpy type conversion for comparison results

**Hebrew Language Support**:
- Custom Hebrew PII generators with realistic patterns
- Israeli ID number validation using Luhn algorithm
- Hebrew date format handling and RTL text processing

## Deployment Options

**Full Pipeline**: Requires all dependencies in `requirements.txt`
**Simple Inference**: Only needs `requirements_minimal.txt` (3 dependencies)
**Standalone**: `simple_inference.py` + ONNX model directory is fully self-contained

The ONNX model directory (`models/onnx/pii-redaction-model/`) contains all files needed for deployment and can be shipped independently.

## Model Performance

- **Languages**: English, Hebrew, Spanish, French, German
- **Accuracy**: >99% F1 score on synthetic test data  
- **Inference Speed**: ~100-300 texts/second on CPU with ONNX
- **Model Size**: ~450MB optimized ONNX model
- **PII Types**: Names, emails, phones, IDs, addresses, credit cards, dates, passports, bank accounts, license plates