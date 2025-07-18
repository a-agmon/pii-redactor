# Simple PII Redaction Deployment

This guide shows how to deploy the PII redaction model with minimal dependencies.

## Files Needed for Deployment

You only need these files from the `models/onnx/pii-redaction-model/` directory:

```
models/onnx/pii-redaction-model/
├── model_optimized.onnx          # The ONNX model (main file)
├── tokenizer.json                # Tokenizer configuration
├── tokenizer_config.json         # Tokenizer metadata
├── config.json                   # Model configuration
├── special_tokens_map.json       # Special tokens mapping
├── added_tokens.json             # Added tokens
└── unigram.json                  # Unigram tokenizer data
```

## Simple Installation

1. **Install minimal dependencies:**
   ```bash
   pip install -r requirements_minimal.txt
   ```

2. **Copy the model files to your deployment directory:**
   ```bash
   cp -r models/onnx/pii-redaction-model /path/to/your/deployment/
   cp simple_inference.py /path/to/your/deployment/
   ```

## Usage

### Command Line
```bash
python simple_inference.py "My name is John Doe and my email is john@example.com"
```

### Python Code
```python
from simple_inference import simple_pii_redact

# Simple usage
text = "My name is John Doe and my email is john@example.com"
redacted = simple_pii_redact(text)
print(redacted)  # "My name is [REDACTED] and my email is [REDACTED]"

# With custom model path
redacted = simple_pii_redact(text, model_path="path/to/your/model")

# Using the class directly
from simple_inference import SimplePIIRedactor

redactor = SimplePIIRedactor("path/to/model")
redacted = redactor.redact(text, redaction_token="[REMOVED]")
```

## Model Size and Performance

- **Model size**: ~450MB (ONNX optimized)
- **Languages**: English, Hebrew, Spanish, French, German  
- **Performance**: ~100-300 texts/second on CPU
- **Accuracy**: >99% F1 score on test data

## Dependencies

Only 3 minimal dependencies required:
- `onnxruntime>=1.16.0` - For ONNX model inference
- `transformers>=4.30.0` - For tokenizer
- `numpy>=1.24.0` - For array operations

## Deployment Options

### 1. Standalone Script
Use `simple_inference.py` directly - no project dependencies needed.

### 2. Docker Container
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements_minimal.txt .
RUN pip install -r requirements_minimal.txt

COPY simple_inference.py .
COPY models/onnx/pii-redaction-model/ models/onnx/pii-redaction-model/

CMD ["python", "simple_inference.py"]
```

### 3. API Service
```python
from flask import Flask, request, jsonify
from simple_inference import SimplePIIRedactor

app = Flask(__name__)
redactor = SimplePIIRedactor()

@app.route('/redact', methods=['POST'])
def redact_text():
    data = request.json
    text = data.get('text', '')
    redacted = redactor.redact(text)
    return jsonify({'redacted': redacted})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Security Notes

- Model processes text locally (no external API calls)
- No data is stored or logged by default
- Suitable for sensitive/private data processing
- Consider running in isolated environment for production

## Supported PII Types

The model detects and redacts:
- Personal names
- Email addresses  
- Phone numbers
- ID numbers
- Addresses
- Credit card numbers
- And other PII patterns

## Troubleshooting

1. **"Special tokens" warning**: This is normal and can be ignored
2. **ONNX provider error**: Make sure you have the CPU provider available
3. **Model not found**: Check the model path exists and contains all required files
4. **Poor performance**: The model works best on text similar to its training data