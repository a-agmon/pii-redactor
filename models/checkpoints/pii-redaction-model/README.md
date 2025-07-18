
---
language: 
- en
- he
- es
- fr
- de
tags:
- token-classification
- pii-detection
- privacy
- multilingual
widget:
- text: "My name is John Doe and my ID number is 123456789"
- text: "שמי אלון ומספר תעודת הזהות שלי הוא 123456789"
---

# PII Redaction Model

This model is fine-tuned for detecting Personally Identifiable Information (PII) in multilingual text,
with special focus on Hebrew language support.

## Model Details

- **Base Model**: distilbert-base-multilingual-cased
- **Task**: Token Classification (PII Detection)
- **Languages**: English, Hebrew, Spanish, French, German
- **Label Schema**: BIO (B-PII, I-PII, O)

## Training Data

The model was trained on synthetic multilingual data with the following PII types:
- Names
- ID Numbers
- Phone Numbers
- Email Addresses
- Physical Addresses
- Credit Card Numbers
- Dates of Birth
- Passport Numbers
- Bank Account Numbers
- License Plate Numbers

## Performance Metrics

- **Overall F1 Score**: 0.0000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **Accuracy**: 0.0000

### PII Entity Metrics
- **PII F1 Score**: 0.0000
- **PII Precision**: 0.0000
- **PII Recall**: 0.0000

## Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/checkpoints/pii-redaction-model")
model = AutoModelForTokenClassification.from_pretrained("models/checkpoints/pii-redaction-model")

# Example usage
text = "My name is John Doe and my phone is 555-1234"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    
# Convert predictions to labels
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
```

## Limitations

- The model is trained on synthetic data and may not capture all real-world PII patterns
- Performance may vary across different languages and PII types
- Hebrew text processing requires proper RTL handling

## Training Configuration

```yaml
dataset:
  hebrew_ratio: 0.4
  languages:
  - en
  - he
  - es
  - fr
  - de
  max_tokens_per_sample: 128
  test_size: 1000
  train_size: 10000
  val_size: 2000
label_map:
  B-PII: 1
  I-PII: 2
  O: 0
model:
  base_model: distilbert-base-multilingual-cased
  label_all_tokens: false
  max_length: 128
onnx:
  opset_version: 14
  optimize: true
  quantize: false
output:
  logs_dir: logs
  model_path: models/checkpoints/pii-redaction-model
  onnx_path: models/onnx/pii-redaction-model
pii_types:
- NAME
- ID_NUMBER
- PHONE
- EMAIL
- ADDRESS
- CREDIT_CARD
- DATE_OF_BIRTH
- PASSPORT
- BANK_ACCOUNT
- LICENSE_PLATE
training:
  batch_size: 16
  evaluation_strategy: epoch
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  logging_steps: 50
  num_epochs: 10
  save_strategy: epoch
  save_total_limit: 3
  seed: 42
  warmup_steps: 500
  weight_decay: 0.01

```
