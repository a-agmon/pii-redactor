"""
Inference Module for PII Redaction

This module provides the inference functionality for the PII redaction model.
It includes both PyTorch and ONNX inference capabilities with optimizations
for production deployment.

Classes:
    PIIRedactor: Main inference class for PII detection and redaction
    BatchProcessor: Handles batch processing for efficiency
    PIIEntity: Data class for PII entities
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
import numpy as np
from pathlib import Path

import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
from optimum.onnxruntime import ORTModelForTokenClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PIIEntity:
    """
    Data class for PII entities detected in text.
    
    Attributes:
        text: The PII text found
        label: The label (B-PII, I-PII, or O)
        start: Start position in original text
        end: End position in original text
        confidence: Confidence score (0-1)
        entity_type: Inferred type of PII (NAME, EMAIL, etc.)
    """
    text: str
    label: str
    start: int
    end: int
    confidence: float
    entity_type: Optional[str] = None


class PIIRedactor:
    """
    Main inference class for PII detection and redaction.
    
    This class provides both PyTorch and ONNX inference capabilities
    with support for Hebrew and multilingual text processing.
    """
    
    def __init__(
        self,
        model_path: str,
        use_onnx: bool = True,
        device: str = "cpu",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize PIIRedactor.
        
        Args:
            model_path: Path to the model directory
            use_onnx: Whether to use ONNX model (faster) or PyTorch model
            device: Device to run inference on ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence for PII detection
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        if use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        # Label mappings
        self.label_map = {0: "O", 1: "B-PII", 2: "I-PII"}
        self.id_to_label = self.label_map
        
        # PII type patterns for classification
        self.pii_patterns = self._initialize_pii_patterns()
        
        logger.info(f"PIIRedactor initialized with {'ONNX' if use_onnx else 'PyTorch'} model")
    
    def _load_onnx_model(self):
        """Load ONNX model for inference"""
        onnx_model_path = os.path.join(self.model_path, "model.onnx")
        
        # Check for optimized versions
        if os.path.exists(onnx_model_path.replace(".onnx", "_quantized.onnx")):
            onnx_model_path = onnx_model_path.replace(".onnx", "_quantized.onnx")
            logger.info("Using quantized ONNX model")
        elif os.path.exists(onnx_model_path.replace(".onnx", "_optimized.onnx")):
            onnx_model_path = onnx_model_path.replace(".onnx", "_optimized.onnx")
            logger.info("Using optimized ONNX model")
        
        # Set up providers
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
        
        # Create session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # Get input names
        self.input_names = [input.name for input in self.session.get_inputs()]
        logger.info(f"ONNX model loaded with inputs: {self.input_names}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model for inference"""
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.eval()
        
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        logger.info("PyTorch model loaded")
    
    def _initialize_pii_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize regex patterns for PII type classification.
        
        Returns:
            Dictionary mapping PII types to regex patterns
        """
        return {
            "EMAIL": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "PHONE": [
                r'\b\d{3}-\d{3}-\d{4}\b',  # Israeli format
                r'\b\d{3}-\d{4}-\d{3}\b',  # Alternative format
                r'\b\+\d{1,3}[-.\s]?\d{1,14}\b'  # International format
            ],
            "ID_NUMBER": [
                r'\b\d{9}\b',  # Israeli ID
                r'\b\d{3}-\d{2}-\d{4}\b'  # SSN format
            ],
            "CREDIT_CARD": [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                r'\b\*{4}[-\s]?\*{4}[-\s]?\*{4}[-\s]?\d{4}\b'
            ],
            "LICENSE_PLATE": [
                r'\b\d{2,3}-\d{3}-\d{2,3}\b',  # Israeli license plate
                r'\b[A-Z]{2}\d{3}[A-Z]{2}\b'  # European format
            ],
            "DATE_OF_BIRTH": [
                r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b',  # MM/DD/YYYY format
                r'\b\d{1,2}\s+(ב)?[א-ת]+\s+\d{4}\b'  # Hebrew date format
            ]
        }
    
    def _classify_pii_type(self, text: str) -> str:
        """
        Classify the type of PII based on text patterns.
        
        Args:
            text: The PII text to classify
            
        Returns:
            Classified PII type or "UNKNOWN"
        """
        for pii_type, patterns in self.pii_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return pii_type
        
        # Hebrew name patterns
        hebrew_name_pattern = r'^[א-ת]+(\s+[א-ת]+)*$'
        if re.match(hebrew_name_pattern, text.strip()):
            return "NAME"
        
        # English name patterns
        english_name_pattern = r'^[A-Za-z]+(\s+[A-Za-z]+)*$'
        if re.match(english_name_pattern, text.strip()) and len(text.split()) <= 3:
            return "NAME"
        
        return "UNKNOWN"
    
    def predict(self, text: str) -> List[PIIEntity]:
        """
        Predict PII entities in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PII entities
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt" if not self.use_onnx else "np",
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True
        )
        
        # Get offset mapping for token-to-character alignment
        offset_mapping = inputs.pop("offset_mapping")
        
        # Run inference
        if self.use_onnx:
            outputs = self._predict_onnx(inputs)
        else:
            outputs = self._predict_pytorch(inputs)
        
        # Process predictions
        predictions = np.argmax(outputs, axis=-1)
        probabilities = self._softmax(outputs)
        
        # Convert predictions to entities
        entities = self._predictions_to_entities(
            text, predictions[0], probabilities[0], offset_mapping[0]
        )
        
        return entities
    
    def _predict_onnx(self, inputs: Dict) -> np.ndarray:
        """Run ONNX inference"""
        # Prepare ONNX inputs
        onnx_inputs = {
            name: inputs[name] for name in self.input_names if name in inputs
        }
        
        # Run inference
        outputs = self.session.run(None, onnx_inputs)
        return outputs[0]
    
    def _predict_pytorch(self, inputs: Dict) -> np.ndarray:
        """Run PyTorch inference"""
        # Move inputs to device
        if self.device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.logits.cpu().numpy()
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to get probabilities"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _predictions_to_entities(
        self,
        text: str,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        offset_mapping: List[Tuple[int, int]]
    ) -> List[PIIEntity]:
        """
        Convert model predictions to PII entities.
        
        Args:
            text: Original text
            predictions: Model predictions (label IDs)
            probabilities: Prediction probabilities
            offset_mapping: Token offset mapping
            
        Returns:
            List of PII entities
        """
        entities = []
        current_entity = None
        
        for i, (pred_id, prob, (start, end)) in enumerate(zip(predictions, probabilities, offset_mapping)):
            # Skip special tokens
            if start == 0 and end == 0:
                continue
            
            label = self.id_to_label[pred_id]
            confidence = prob[pred_id]
            
            # Skip low confidence predictions
            if confidence < self.confidence_threshold:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            if label == "B-PII":
                # Save previous entity if exists
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                current_entity = PIIEntity(
                    text=text[start:end],
                    label=label,
                    start=start,
                    end=end,
                    confidence=confidence
                )
            
            elif label == "I-PII" and current_entity:
                # Extend current entity
                current_entity.text = text[current_entity.start:end]
                current_entity.end = end
                current_entity.confidence = min(current_entity.confidence, confidence)
            
            elif label == "O":
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        # Classify entity types
        for entity in entities:
            entity.entity_type = self._classify_pii_type(entity.text)
        
        return entities
    
    def redact(
        self,
        text: str,
        replacement: str = "[REDACTED]",
        preserve_format: bool = True
    ) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Input text
            replacement: Replacement string for PII
            preserve_format: Whether to preserve text formatting
            
        Returns:
            Text with PII redacted
        """
        entities = self.predict(text)
        
        # Sort entities by start position (descending) to avoid offset issues
        entities.sort(key=lambda x: x.start, reverse=True)
        
        redacted_text = text
        
        for entity in entities:
            # Format replacement based on entity type
            if preserve_format:
                if entity.entity_type == "NAME":
                    repl = f"[NAME_REDACTED]"
                elif entity.entity_type == "EMAIL":
                    repl = f"[EMAIL_REDACTED]"
                elif entity.entity_type == "PHONE":
                    repl = f"[PHONE_REDACTED]"
                elif entity.entity_type == "ID_NUMBER":
                    repl = f"[ID_REDACTED]"
                else:
                    repl = replacement
            else:
                repl = replacement
            
            # Replace PII with redaction
            redacted_text = (
                redacted_text[:entity.start] + 
                repl + 
                redacted_text[entity.end:]
            )
        
        return redacted_text
    
    def redact_with_info(
        self,
        text: str,
        replacement: str = "[REDACTED]"
    ) -> Tuple[str, List[PIIEntity]]:
        """
        Redact PII and return both redacted text and entity information.
        
        Args:
            text: Input text
            replacement: Replacement string for PII
            
        Returns:
            Tuple of (redacted_text, entities)
        """
        entities = self.predict(text)
        redacted_text = self.redact(text, replacement)
        
        return redacted_text, entities
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for PII and return detailed statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with analysis results
        """
        entities = self.predict(text)
        
        # Count entities by type
        entity_counts = {}
        for entity in entities:
            entity_type = entity.entity_type or "UNKNOWN"
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Calculate statistics
        total_entities = len(entities)
        total_chars = len(text)
        pii_chars = sum(len(entity.text) for entity in entities)
        pii_ratio = pii_chars / total_chars if total_chars > 0 else 0
        
        avg_confidence = np.mean([entity.confidence for entity in entities]) if entities else 0
        
        return {
            "total_entities": total_entities,
            "entity_counts": entity_counts,
            "pii_character_ratio": pii_ratio,
            "average_confidence": float(avg_confidence),
            "entities": [
                {
                    "text": entity.text,
                    "type": entity.entity_type,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence
                }
                for entity in entities
            ]
        }


class BatchProcessor:
    """
    Handles batch processing for efficient PII redaction.
    
    This class optimizes inference for multiple texts by batching
    them together for better GPU utilization.
    """
    
    def __init__(self, redactor: PIIRedactor, batch_size: int = 32):
        """
        Initialize batch processor.
        
        Args:
            redactor: PIIRedactor instance
            batch_size: Number of texts to process in each batch
        """
        self.redactor = redactor
        self.batch_size = batch_size
    
    def process_batch(self, texts: List[str]) -> List[Tuple[str, List[PIIEntity]]]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts to process
            
        Returns:
            List of (redacted_text, entities) tuples
        """
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = []
            
            for text in batch_texts:
                redacted_text, entities = self.redactor.redact_with_info(text)
                batch_results.append((redacted_text, entities))
            
            results.extend(batch_results)
        
        return results
    
    def process_file(self, input_file: str, output_file: str, encoding: str = 'utf-8'):
        """
        Process a text file and save redacted version.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            encoding: File encoding
        """
        logger.info(f"Processing file: {input_file}")
        
        with open(input_file, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        # Process in batches
        redacted_lines = []
        for i in range(0, len(lines), self.batch_size):
            batch_lines = lines[i:i + self.batch_size]
            
            for line in batch_lines:
                redacted_line = self.redactor.redact(line.strip())
                redacted_lines.append(redacted_line + '\n')
        
        # Save redacted file
        with open(output_file, 'w', encoding=encoding) as f:
            f.writelines(redacted_lines)
        
        logger.info(f"Redacted file saved to: {output_file}")


def benchmark_inference(redactor: PIIRedactor, test_texts: List[str], num_runs: int = 100):
    """
    Benchmark inference performance.
    
    Args:
        redactor: PIIRedactor instance
        test_texts: List of test texts
        num_runs: Number of benchmark runs
    """
    logger.info("Starting inference benchmark...")
    
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        for text in test_texts:
            _ = redactor.predict(text)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    logger.info(f"Benchmark results:")
    logger.info(f"Average time per batch: {avg_time:.4f}s ± {std_time:.4f}s")
    logger.info(f"Texts per second: {len(test_texts) / avg_time:.2f}")


def main():
    """Test inference functionality"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Test texts
    test_texts = [
        "My name is John Doe and my phone number is 555-1234",
        "שמי אלון ומספר תעודת הזהות שלי הוא 123456789",
        "Contact me at john.doe@example.com or call 050-1234567",
        "הכתובת שלי היא רחוב הרצל 15, תל אביב"
    ]
    
    logger.info("PII Redaction Inference Module")
    logger.info("Example usage:")
    logger.info("redactor = PIIRedactor(model_path)")
    logger.info("redacted_text = redactor.redact(text)")
    
    for i, text in enumerate(test_texts):
        logger.info(f"Test {i+1}: {text}")


if __name__ == "__main__":
    main()