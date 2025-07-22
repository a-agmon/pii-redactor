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
        # Tokenize with offset mapping to preserve character positions
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            return_token_type_ids=True,
            return_offsets_mapping=True
        )
        
        # Get offset mapping for token-to-character alignment
        offset_mapping = inputs.pop("return_offsets_mapping", None)
        if offset_mapping is None:
            offset_mapping = inputs.pop("offset_mapping", None)
        
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
        
        # Find PII entities using offset mapping
        entities = []
        current_entity_start = None
        current_entity_end = None
        
        for i, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping[0])):
            if inputs["attention_mask"][0][i] == 0:  # Padding token
                break
                
            # Skip special tokens (offset mapping is (0, 0) for special tokens)
            if start == 0 and end == 0:
                continue
                
            label = self.id_to_label[pred]
            
            if label == "B-PII":  # Beginning of PII
                # Save previous entity if exists
                if current_entity_start is not None:
                    entities.append((current_entity_start, current_entity_end))
                # Start new entity
                current_entity_start = start
                current_entity_end = end
            elif label == "I-PII":  # Inside PII
                # Extend current entity
                if current_entity_start is not None:
                    current_entity_end = end
            else:  # Not PII
                # End current entity
                if current_entity_start is not None:
                    entities.append((current_entity_start, current_entity_end))
                    current_entity_start = None
                    current_entity_end = None
        
        # Add final entity if exists
        if current_entity_start is not None:
            entities.append((current_entity_start, current_entity_end))
        
        # Adjust entity boundaries to preserve punctuation
        entities = self._adjust_entity_boundaries(text, entities)
        
        # Apply redaction by replacing character ranges (reverse order to avoid offset issues)
        redacted_text = text
        for start, end in reversed(entities):
            redacted_text = redacted_text[:start] + redaction_token + redacted_text[end:]
        
        return redacted_text
    
    def _adjust_entity_boundaries(self, text: str, entities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Adjust entity boundaries to exclude leading/trailing punctuation."""
        adjusted_entities = []
        
        for start, end in entities:
            # For phone numbers, we want to preserve the format but redact the digits
            # Check if this looks like a phone number pattern
            entity_text = text[start:end]
            
            # If it contains digits and common phone punctuation, handle specially
            if any(c.isdigit() for c in entity_text):
                # For phone numbers, only trim outer punctuation that's not part of the number
                # Trim leading non-digit, non-phone punctuation
                while start < end and text[start] in '—[]{}"\':;.,!?/\\|@#$%^&*+=~`':
                    start += 1
                
                # Trim trailing non-digit, non-phone punctuation  
                while end > start and text[end-1] in '—[]{}"\':;.,!?/\\|@#$%^&*+=~`':
                    end -= 1
            else:
                # For non-phone PII, trim all punctuation
                while start < end and text[start] in '—-()[]{}"\':;.,!?/\\|@#$%^&*+=~`':
                    start += 1
                
                while end > start and text[end-1] in '—-()[]{}"\':;.,!?/\\|@#$%^&*+=~`':
                    end -= 1
            
            # Only add the entity if there's content left
            if start < end:
                adjusted_entities.append((start, end))
        
        return adjusted_entities


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