#!/usr/bin/env python3
"""
Simple Standalone PII Redaction Inference

This script provides a minimal implementation for PII redaction using ONNX model.
Only requires: onnxruntime, transformers, numpy

Usage:
    python simple_inference.py "Your text with PII here"
    
    # Or use in code:
    from simple_inference import simple_pii_redact
    redacted = simple_pii_redact("My name is John Doe and my email is john@example.com")
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple, Optional

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install onnxruntime transformers")
    sys.exit(1)


class SimplePIIRedactor:
    """Minimal PII redactor using ONNX model."""
    
    def __init__(self, model_path: str = "models/onnx/pii-redaction-model"):
        """
        Initialize the PII redactor.
        
        Args:
            model_path: Path to directory containing ONNX model files
        """
        self.model_path = model_path
        self.max_length = 128
        
        # Load tokenizer
        print(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load ONNX model
        onnx_model_path = os.path.join(model_path, "model_optimized.onnx")
        if not os.path.exists(onnx_model_path):
            onnx_model_path = os.path.join(model_path, "model.onnx")
        
        print(f"Loading ONNX model from {onnx_model_path}")
        self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        
        # Labels
        self.id_to_label = {0: "O", 1: "B-PII", 2: "I-PII"}
        
        print("Model loaded successfully!")
    
    def redact(self, text: str, redaction_token: str = "[REDACTED]") -> str:
        """
        Redact PII from text.
        
        Args:
            text: Input text
            redaction_token: Token to replace PII with
            
        Returns:
            Text with PII redacted
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            return_token_type_ids=True
        )
        
        # Prepare inputs for ONNX
        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        # Note: DistilBERT-based models don't use token_type_ids in ONNX export
        
        # Run inference
        outputs = self.session.run(None, onnx_inputs)
        
        # Get predictions
        predictions = np.argmax(outputs[0], axis=-1)[0]
        
        # Convert tokens back to text with redaction
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Group tokens and apply redaction
        redacted_tokens = []
        in_pii = False
        
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if inputs["attention_mask"][0][i] == 0:  # Padding token
                break
                
            label = self.id_to_label[pred]
            
            # Skip special tokens
            if token in ["<s>", "</s>", "[CLS]", "[SEP]", "[PAD]", "<pad>"]:
                continue
                
            if label == "B-PII":  # Beginning of PII
                if not in_pii:
                    redacted_tokens.append(f" {redaction_token} ")
                    in_pii = True
            elif label == "I-PII":  # Inside PII
                continue  # Skip this token
            else:  # Not PII
                if in_pii:
                    in_pii = False
                redacted_tokens.append(token)
        
        # Convert back to text
        redacted_text = self.tokenizer.convert_tokens_to_string(redacted_tokens)
        
        # Clean up text formatting
        redacted_text = redacted_text.replace(" ##", "")  # Fix subword tokens
        redacted_text = redacted_text.replace("##", "")
        redacted_text = redacted_text.replace("  ", " ")  # Remove double spaces
        
        return redacted_text.strip()


def simple_pii_redact(text: str, model_path: str = "models/onnx/pii-redaction-model") -> str:
    """
    Simple function to redact PII from text.
    
    Args:
        text: Input text
        model_path: Path to ONNX model directory
        
    Returns:
        Text with PII redacted
    """
    redactor = SimplePIIRedactor(model_path)
    return redactor.redact(text)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Simple PII Redaction")
    parser.add_argument("text", help="Text to redact PII from")
    parser.add_argument("--model-path", default="models/onnx/pii-redaction-model",
                       help="Path to ONNX model directory")
    parser.add_argument("--redaction-token", default="[REDACTED]",
                       help="Token to replace PII with")
    
    args = parser.parse_args()
    
    try:
        redactor = SimplePIIRedactor(args.model_path)
        redacted = redactor.redact(args.text, args.redaction_token)
        
        print(f"Original: {args.text}")
        print(f"Redacted: {redacted}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()