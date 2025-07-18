#!/usr/bin/env python3

import os
import sys
sys.path.append('src')
from dataset_creation import PIIDataProcessor
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Create processor
processor = PIIDataProcessor(tokenizer, max_length=512)

# Test with a simple example
test_examples = {
    "tokens": [
        ["Hello", "my", "name", "is", "John", "Doe"],
        ["I", "live", "in", "New", "York"]
    ],
    "labels": [
        ["O", "O", "O", "O", "B-PII", "I-PII"],
        ["O", "O", "O", "B-PII", "I-PII"]
    ]
}

print("Input:")
print(f"Tokens: {test_examples['tokens']}")
print(f"Labels: {test_examples['labels']}")
print()

# Test tokenization
result = processor.tokenize_and_align_labels(test_examples)

print("Output:")
for key, value in result.items():
    print(f"{key}: {type(value)} - {value}")
    if isinstance(value, list) and len(value) > 0:
        print(f"  First element type: {type(value[0])}")
        if isinstance(value[0], list):
            print(f"  Nested list length: {len(value[0])}")
            print(f"  First few elements: {value[0][:10]}")