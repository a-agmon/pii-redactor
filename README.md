# PII Redaction Model Training Pipeline

A comprehensive machine learning pipeline for training multilingual PII (Personally Identifiable Information) redaction models with special focus on Hebrew language support.


##  Project Structure

```
pii-redactor/
├── data/
│   ├── raw/                 # Raw data files
│   ├── processed/           # Processed datasets
│   └── synthetic/           # Synthetic training data
├── models/
│   ├── checkpoints/         # Model checkpoints
│   └── onnx/               # ONNX models
├── src/
│   ├── dataset_creation.py  # Synthetic data generation
│   ├── model_training.py    # Model training logic
│   ├── onnx_conversion.py   # ONNX conversion utilities
│   └── inference.py         # Inference and redaction
├── logs/                    # Training logs
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
├── main.py                 # Main training pipeline
└── test_inference.py       # Inference testing
```

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd pii-redactor

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.yaml` to customize training parameters:

```yaml
model:
  base_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  max_length: 128

training:
  batch_size: 16
  learning_rate: 5e-5
  num_epochs: 10

dataset:
  train_size: 10000
  val_size: 2000
  test_size: 1000
  hebrew_ratio: 0.4  # 40% Hebrew data
```

### 3. Training

Run the complete training pipeline:

```bash
# Train complete pipeline
python main.py

# Or run specific stages
python main.py --stage data     # Generate dataset only
python main.py --stage train    # Train model only
python main.py --stage convert  # Convert to ONNX only
python main.py --stage test     # Test inference only
```

### Cleanup and Fresh Start

Use the cleanup script to remove training artifacts and start fresh:

```bash
# See what would be cleaned (dry run)
python cleanup.py --dry-run

# Clean everything (models, data, logs)
python cleanup.py --all

# Clean specific components
python cleanup.py --models    # Just model checkpoints and ONNX files
python cleanup.py --data      # Just datasets
python cleanup.py --logs      # Just training logs
python cleanup.py --cache     # Hugging Face cache (use with caution)

# After cleanup, start fresh training
python main.py
```

### 4. Inference

```python
from src.inference import PIIRedactor

# Load the trained model
redactor = PIIRedactor('models/onnx/pii-redaction-model', use_onnx=True)

# Redact PII from text
text = "My name is John Doe and my phone is 555-1234"
redacted = redactor.redact(text)
print(redacted)  # "My name is [NAME_REDACTED] and my phone is [PHONE_REDACTED]"

# Hebrew example
hebrew_text = "שמי אלון ומספר תעודת הזהות שלי הוא 123456789"
redacted_hebrew = redactor.redact(hebrew_text)
print(redacted_hebrew)  # "שמי [NAME_REDACTED] ומספר תעודת הזהות שלי הוא [ID_REDACTED]"
```

##  Dataset Creation

The pipeline generates synthetic multilingual PII data:

### Hebrew PII Generator
- Generates realistic Hebrew names, addresses, and phone numbers
- Follows Israeli ID number validation (Luhn algorithm)
- Handles Hebrew date formats and RTL text
- Creates contextual sentence templates

### Multilingual Support
- Uses Faker library for non-Hebrew languages
- Maintains consistent PII patterns across languages
- Balances dataset with configurable language ratios

### BIO Tagging
- Uses Begin-Inside-Outside tagging scheme
- Handles subword tokenization alignment
- Supports complex multi-token entities

##  Model Architecture & Training

### How Token Classification Works

The model uses a fine-tuned transformer architecture for token-level PII detection:

```
Input Text → Tokenizer → Base Model (DistilBERT) → Classification Head → PII Labels
```

1. **Base Model**: Pre-trained multilingual transformer (e.g., DistilBERT)
   - Provides contextual understanding of each token
   - 768-dimensional representation per token
   - Fine-tuned during training to adapt to PII detection

2. **Classification Head**: Linear layer added on top
   - Maps 768-dim token representations to 3 classes
   - Simple linear transformation: `Linear(768, 3)`
   - Trained from scratch for PII classification

3. **Label Schema** (BIO tagging):
   - **O (0)**: Outside any PII entity
   - **B-PII (1)**: Beginning of PII entity
   - **I-PII (2)**: Inside PII entity

4. **Example Flow**:
   ```
   Input: "My name is John Doe"
   Tokens: ["My", "name", "is", "John", "Doe"]
   Predictions: [O, O, O, B-PII, I-PII]
   Output: "My name is [REDACTED]"
   ```

### Why This Architecture?

- **Contextual Understanding**: The model learns when "John" is a person's name vs. part of "John Deere"
- **Multilingual**: Base model understands 100+ languages including Hebrew
- **Efficient**: Only adds a small classification head (2,304 parameters) to the base model
- **Transfer Learning**: Leverages pre-trained language understanding

### Training Features
- **Fine-tuning**: Both base model and classification head are trained
- **Early stopping**: Prevents overfitting with patience mechanism
- **Gradient accumulation**: Enables larger effective batch sizes
- **Mixed precision**: Faster training with minimal accuracy loss
- **Checkpointing**: Saves best model based on validation F1 score
- **Error analysis**: Detailed metrics for each PII type

## ⚡ ONNX Conversion

### Optimization Features
- **Graph Optimization**: Removes unnecessary operations
- **Quantization**: Optional INT8 quantization for smaller models
- **Platform Optimization**: CPU/GPU specific optimizations
- **Validation**: Automatic output comparison between PyTorch and ONNX

### Performance Benefits
- 2-5x faster inference compared to PyTorch
- Reduced memory footprint
- Cross-platform compatibility
- Mobile deployment ready

##  Advanced Usage

### Batch Processing

```python
from src.inference import PIIRedactor, BatchProcessor

redactor = PIIRedactor('models/onnx/pii-redaction-model')
batch_processor = BatchProcessor(redactor, batch_size=32)

# Process multiple texts
texts = ["text1", "text2", "text3"]
results = batch_processor.process_batch(texts)

# Process entire files
batch_processor.process_file('input.txt', 'output_redacted.txt')
```

### Detailed Analysis

```python
# Get detailed PII analysis
analysis = redactor.analyze_text("My name is John and my email is john@example.com")

print(f"Total entities: {analysis['total_entities']}")
print(f"PII ratio: {analysis['pii_character_ratio']:.2%}")
print(f"Entity types: {analysis['entity_counts']}")
```

### Custom Redaction

```python
# Custom redaction with entity information
redacted_text, entities = redactor.redact_with_info(text)

for entity in entities:
    print(f"Found {entity.entity_type}: '{entity.text}' "
          f"(confidence: {entity.confidence:.3f})")
```

##  Performance Benchmarks

### Inference Speed
- **PyTorch**: ~50-100 texts/second
- **ONNX**: ~100-300 texts/second
- **Batch Processing**: Up to 500 texts/second

### Model Size
- **PyTorch**: ~500MB
- **ONNX**: ~130MB
- **Quantized ONNX**: ~35MB

### Accuracy Metrics
- **Overall F1**: >0.95
- **Hebrew F1**: >0.93
- **Cross-language consistency**: >0.90

## 🧪 Testing

### Comprehensive Test Suite

```bash
# Run all tests
python test_inference.py

# Run specific test suites
python test_inference.py --test-suite basic
python test_inference.py --test-suite performance
python test_inference.py --test-suite batch

# Interactive testing
python test_inference.py --interactive
```

### Test Coverage
- **Basic Redaction**: Simple PII detection and redaction
- **Multilingual**: Tests across all supported languages
- **Edge Cases**: Partial PII, mixed languages, no PII
- **Performance**: Throughput and latency benchmarks
- **Batch Processing**: Efficiency with multiple texts

##  Configuration Reference

### Model Configuration
```yaml
model:
  base_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  max_length: 128              # Maximum sequence length
  label_all_tokens: false      # Only label first subword
```

### Training Configuration
```yaml
training:
  batch_size: 16              # Training batch size
  learning_rate: 5e-5         # Learning rate
  num_epochs: 10              # Number of training epochs
  warmup_steps: 500           # Warmup steps
  weight_decay: 0.01          # Weight decay
  gradient_accumulation_steps: 2
  save_total_limit: 3         # Keep only last 3 checkpoints
  seed: 42                    # Random seed
```

### Dataset Configuration
```yaml
dataset:
  train_size: 10000           # Training samples
  val_size: 2000              # Validation samples
  test_size: 1000             # Test samples
  languages: ["en", "he", "es", "fr", "de"]
  hebrew_ratio: 0.4           # Hebrew data percentage
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config.yaml
   training:
     batch_size: 8
     gradient_accumulation_steps: 4
   ```

2. **Hebrew Text Display Issues**
   ```python
   # Ensure proper encoding
   with open('file.txt', 'r', encoding='utf-8') as f:
       text = f.read()
   ```

3. **ONNX Conversion Errors**
   ```bash
   # Install specific ONNX version
   pip install onnx==1.15.0 onnxruntime==1.16.0
   ```

### Performance Optimization

1. **For CPU Inference**
   ```python
   # Use ONNX with CPU optimization
   redactor = PIIRedactor(model_path, use_onnx=True, device='cpu')
   ```

2. **For GPU Inference**
   ```python
   # Use PyTorch with GPU
   redactor = PIIRedactor(model_path, use_onnx=False, device='cuda')
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure code follows style guidelines
5. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Hugging Face Transformers for the model architecture
- Sentence Transformers for the multilingual base model
- ONNX Runtime for optimization capabilities
- Faker library for synthetic data generation

##  Support

For questions and support:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the comprehensive test examples

---

**Note**: This model is trained on synthetic data and should be thoroughly tested before production use. Always validate performance on your specific use case and data distribution.
