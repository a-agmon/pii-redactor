#!/usr/bin/env python3
"""
Quick Example - PII Redaction Model Usage

This script demonstrates basic usage of the PII redaction model
with simple examples in multiple languages.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from inference import PIIRedactor


def main():
    """Demonstrate basic PII redaction usage"""
    print("=" * 60)
    print("PII REDACTION - QUICK EXAMPLE")
    print("=" * 60)
    
    # Model path - adjust as needed
    model_path = "../models/onnx/pii-redaction-model"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train the model first using: python main.py")
        return
    
    # Initialize redactor
    print("Loading model...")
    redactor = PIIRedactor(model_path, use_onnx=True)
    print("Model loaded successfully!")
    
    # Example texts
    examples = [
        {
            "text": "My name is John Doe and my phone is 555-1234",
            "language": "English"
        },
        {
            "text": "שמי אלון כהן ומספר תעודת הזהות שלי הוא 123456789",
            "language": "Hebrew"
        },
        {
            "text": "Contact me at john.doe@example.com or call (555) 123-4567",
            "language": "English"
        },
        {
            "text": "אני גר ברחוב הרצל 15, תל אביב והטלפון שלי 050-1234567",
            "language": "Hebrew"
        }
    ]
    
    print("\nRedaction Examples:")
    print("-" * 60)
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i} ({example['language']}):")
        print(f"Original:  {example['text']}")
        
        # Redact PII
        redacted = redactor.redact(example['text'])
        print(f"Redacted:  {redacted}")
        
        # Show entities found
        entities = redactor.predict(example['text'])
        if entities:
            print(f"Entities:  {len(entities)} PII entities found")
            for entity in entities:
                print(f"  - {entity.entity_type}: '{entity.text}' (confidence: {entity.confidence:.3f})")
        else:
            print("Entities:  No PII detected")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()