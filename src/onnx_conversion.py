"""
ONNX Conversion Module for PII Redaction Model

This module handles the conversion of the trained PyTorch model to ONNX format
for efficient inference. It includes optimization techniques to reduce model size
and improve inference speed while maintaining accuracy.

Classes:
    ONNXConverter: Handles model conversion and optimization
    ONNXOptimizer: Provides advanced optimization techniques
"""

import os
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import yaml
import json
from pathlib import Path

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers import optimizer

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer
)
from optimum.onnxruntime import ORTModelForTokenClassification

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ONNXConverter:
    """
    Convert trained PyTorch model to ONNX format.
    
    This class handles the conversion process and ensures the ONNX model
    produces the same outputs as the original PyTorch model.
    """
    
    def __init__(self, model_path: str, onnx_path: str, config: Optional[Dict] = None):
        """
        Initialize ONNX converter.
        
        Args:
            model_path: Path to the trained PyTorch model
            onnx_path: Path where ONNX model will be saved
            config: Optional configuration dictionary
        """
        self.model_path = model_path
        self.onnx_path = onnx_path
        self.config = config or {}
        
        # Create output directory if it doesn't exist
        os.makedirs(onnx_path, exist_ok=True)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def convert(self, optimize: bool = True, quantize: bool = False) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            optimize: Whether to apply ONNX optimizations
            quantize: Whether to apply dynamic quantization
            
        Returns:
            Path to the converted ONNX model
        """
        logger.info("Starting ONNX conversion...")
        
        # Method 1: Using Optimum library (recommended)
        try:
            return self._convert_with_optimum(optimize, quantize)
        except Exception as e:
            logger.warning(f"Optimum conversion failed: {e}")
            logger.info("Falling back to manual conversion...")
            return self._convert_manual(optimize, quantize)
    
    def _convert_with_optimum(self, optimize: bool, quantize: bool) -> str:
        """
        Convert using Hugging Face Optimum library.
        
        Args:
            optimize: Whether to apply optimizations
            quantize: Whether to apply quantization
            
        Returns:
            Path to the ONNX model
        """
        logger.info("Converting with Optimum library...")
        
        # Load and export model
        model = ORTModelForTokenClassification.from_pretrained(
            self.model_path,
            export=True,
            provider="CPUExecutionProvider"
        )
        
        # Save model and tokenizer
        model.save_pretrained(self.onnx_path)
        self.tokenizer.save_pretrained(self.onnx_path)
        
        onnx_model_path = os.path.join(self.onnx_path, "model.onnx")
        
        # Apply optimizations if requested
        if optimize:
            onnx_model_path = self._optimize_onnx(onnx_model_path)
        
        # Apply quantization if requested
        if quantize:
            onnx_model_path = self._quantize_onnx(onnx_model_path)
        
        # Validate the converted model
        self._validate_onnx_model(onnx_model_path)
        
        logger.info(f"ONNX model saved to {onnx_model_path}")
        return onnx_model_path
    
    def _convert_manual(self, optimize: bool, quantize: bool) -> str:
        """
        Manual ONNX conversion using torch.onnx.export.
        
        Args:
            optimize: Whether to apply optimizations
            quantize: Whether to apply quantization
            
        Returns:
            Path to the ONNX model
        """
        logger.info("Performing manual ONNX conversion...")
        
        # Load PyTorch model
        model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        model.eval()
        
        # Prepare dummy input
        dummy_input = self._create_dummy_input()
        
        # Export to ONNX
        onnx_model_path = os.path.join(self.onnx_path, "model.onnx")
        
        # Define input and output names
        input_names = ["input_ids", "attention_mask"]
        output_names = ["logits"]
        
        # Dynamic axes for variable sequence length
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        }
        
        # Export model
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            onnx_model_path,
            export_params=True,
            opset_version=self.config.get('onnx', {}).get('opset_version', 13),
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.onnx_path)
        
        # Save config for inference
        config_path = os.path.join(self.onnx_path, "onnx_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "input_names": input_names,
                "output_names": output_names,
                "max_length": self.config.get('model', {}).get('max_length', 128)
            }, f)
        
        # Apply optimizations
        if optimize:
            onnx_model_path = self._optimize_onnx(onnx_model_path)
        
        # Apply quantization
        if quantize:
            onnx_model_path = self._quantize_onnx(onnx_model_path)
        
        # Validate model
        self._validate_onnx_model(onnx_model_path)
        
        logger.info(f"ONNX model saved to {onnx_model_path}")
        return onnx_model_path
    
    def _create_dummy_input(self) -> Dict[str, torch.Tensor]:
        """
        Create dummy input for ONNX export.
        
        Returns:
            Dictionary with input tensors
        """
        max_length = self.config.get('model', {}).get('max_length', 128)
        
        # Create dummy text
        dummy_text = "My name is John Doe and my ID is 123456789"
        
        # Tokenize
        inputs = self.tokenizer(
            dummy_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return inputs
    
    def _optimize_onnx(self, onnx_model_path: str) -> str:
        """
        Apply ONNX optimizations to reduce model size and improve speed.
        
        Args:
            onnx_model_path: Path to the ONNX model
            
        Returns:
            Path to the optimized model
        """
        logger.info("Optimizing ONNX model...")
        
        # Output path for optimized model
        optimized_path = onnx_model_path.replace(".onnx", "_optimized.onnx")
        
        try:
            # Use ONNX Runtime transformer optimizer
            opt_model = optimizer.optimize_model(
                onnx_model_path,
                model_type='bert',
                num_heads=12,  # for MiniLM
                hidden_size=384,  # for MiniLM
                optimization_options=None,
                opt_level=99,
                use_gpu=False,
                only_onnxruntime=True
            )
            
            # Save optimized model
            opt_model.save_model_to_file(optimized_path)
            
            # Compare model sizes
            original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)  # MB
            optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)  # MB
            reduction = ((original_size - optimized_size) / original_size) * 100
            
            logger.info(f"Model size reduced by {reduction:.1f}%")
            logger.info(f"Original: {original_size:.1f} MB, Optimized: {optimized_size:.1f} MB")
            
            return optimized_path
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            return onnx_model_path
    
    def _quantize_onnx(self, onnx_model_path: str) -> str:
        """
        Apply dynamic quantization to reduce model size.
        
        Args:
            onnx_model_path: Path to the ONNX model
            
        Returns:
            Path to the quantized model
        """
        logger.info("Applying dynamic quantization...")
        
        # Output path for quantized model
        quantized_path = onnx_model_path.replace(".onnx", "_quantized.onnx")
        
        try:
            # Apply dynamic quantization
            quantize_dynamic(
                onnx_model_path,
                quantized_path,
                weight_type=QuantType.QInt8,
                optimize_model=True,
                per_channel=True,
                reduce_range=True
            )
            
            # Compare model sizes
            original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)  # MB
            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)  # MB
            reduction = ((original_size - quantized_size) / original_size) * 100
            
            logger.info(f"Model size reduced by {reduction:.1f}% through quantization")
            logger.info(f"Original: {original_size:.1f} MB, Quantized: {quantized_size:.1f} MB")
            
            return quantized_path
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return onnx_model_path
    
    def _validate_onnx_model(self, onnx_model_path: str):
        """
        Validate the ONNX model to ensure it's correctly formatted.
        
        Args:
            onnx_model_path: Path to the ONNX model
        """
        logger.info("Validating ONNX model...")
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model validation passed!")
            
            # Test inference
            self._test_onnx_inference(onnx_model_path)
            
        except Exception as e:
            if "LayerNormalization" in str(e) and "domain_version" in str(e):
                logger.warning(f"ONNX model validation warning (LayerNormalization domain version): {e}")
                logger.info("Continuing with inference test despite LayerNormalization domain version warning...")
                # Still test inference to make sure model works
                try:
                    self._test_onnx_inference(onnx_model_path)
                    logger.info("ONNX model inference test passed despite validation warning!")
                except Exception as inference_error:
                    logger.error(f"ONNX model inference failed: {inference_error}")
                    raise
            else:
                logger.error(f"ONNX model validation failed: {e}")
                raise
    
    def _test_onnx_inference(self, onnx_model_path: str):
        """
        Test ONNX model inference to ensure it works correctly.
        
        Args:
            onnx_model_path: Path to the ONNX model
        """
        logger.info("Testing ONNX inference...")
        
        # Create inference session
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get input names
        input_names = [input.name for input in session.get_inputs()]
        
        # Create test input
        test_text = "Test: My name is John and my ID is 123456789"
        inputs = self.tokenizer(
            test_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="np"
        )
        
        # Prepare ONNX inputs
        onnx_inputs = {name: inputs[name] for name in input_names if name in inputs}
        
        # Run inference
        outputs = session.run(None, onnx_inputs)
        
        logger.info(f"ONNX inference successful! Output shape: {outputs[0].shape}")
    
    def compare_outputs(self, pytorch_model_path: str, test_texts: list) -> Dict:
        """
        Compare outputs between PyTorch and ONNX models.
        
        Args:
            pytorch_model_path: Path to PyTorch model
            test_texts: List of test texts
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing PyTorch and ONNX outputs...")
        
        # Load PyTorch model
        pytorch_model = AutoModelForTokenClassification.from_pretrained(pytorch_model_path)
        pytorch_model.eval()
        
        # Load ONNX model
        onnx_model_path = os.path.join(self.onnx_path, "model.onnx")
        if os.path.exists(onnx_model_path.replace(".onnx", "_quantized.onnx")):
            onnx_model_path = onnx_model_path.replace(".onnx", "_quantized.onnx")
        elif os.path.exists(onnx_model_path.replace(".onnx", "_optimized.onnx")):
            onnx_model_path = onnx_model_path.replace(".onnx", "_optimized.onnx")
        
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        
        input_names = [input.name for input in session.get_inputs()]
        
        # Compare outputs
        max_diff = 0.0
        avg_diff = 0.0
        num_comparisons = 0
        
        for text in test_texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_outputs = pytorch_model(**inputs).logits.numpy()
            
            # ONNX inference
            onnx_inputs = {
                name: inputs[name].numpy() for name in input_names if name in inputs
            }
            onnx_outputs = session.run(None, onnx_inputs)[0]
            
            # Compare
            diff = np.abs(pytorch_outputs - onnx_outputs)
            max_diff = max(max_diff, np.max(diff))
            avg_diff += np.mean(diff)
            num_comparisons += 1
        
        avg_diff /= num_comparisons
        
        results = {
            "max_difference": float(max_diff),
            "avg_difference": float(avg_diff),
            "num_comparisons": num_comparisons,
            "outputs_match": max_diff < 1e-4
        }
        
        logger.info(f"Comparison results: {results}")
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if hasattr(value, 'item'):  # numpy scalar
                json_results[key] = value.item()
            elif isinstance(value, bool):
                json_results[key] = bool(value)
            else:
                json_results[key] = value
        
        # Save comparison results
        results_path = os.path.join(self.onnx_path, "comparison_results.json")
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        return results


class ONNXOptimizer:
    """
    Advanced optimization techniques for ONNX models.
    
    This class provides additional optimization methods beyond standard
    ONNX optimizations.
    """
    
    @staticmethod
    def optimize_for_mobile(onnx_model_path: str) -> str:
        """
        Optimize ONNX model for mobile deployment.
        
        Args:
            onnx_model_path: Path to ONNX model
            
        Returns:
            Path to mobile-optimized model
        """
        logger.info("Optimizing for mobile deployment...")
        
        # Load model
        model = onnx.load(onnx_model_path)
        
        # Apply mobile-specific optimizations
        from onnx import optimizer as onnx_optimizer
        
        # Define optimization passes
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'eliminate_deadend',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_add_bias_into_conv',
            'fuse_transpose_into_gemm'
        ]
        
        # Apply optimizations
        optimized_model = onnx_optimizer.optimize(model, passes)
        
        # Save optimized model
        mobile_path = onnx_model_path.replace(".onnx", "_mobile.onnx")
        onnx.save(optimized_model, mobile_path)
        
        logger.info(f"Mobile-optimized model saved to {mobile_path}")
        return mobile_path
    
    @staticmethod
    def create_execution_provider_options() -> Dict:
        """
        Create optimal execution provider options for different platforms.
        
        Returns:
            Dictionary with platform-specific options
        """
        import platform
        
        system = platform.system()
        options = {}
        
        if system == "Darwin":  # macOS
            options["providers"] = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif system == "Windows":
            options["providers"] = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:  # Linux
            options["providers"] = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        
        return options


def main():
    """Test ONNX conversion"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Example usage
    logger.info("ONNX Conversion Module Loaded")
    logger.info("To convert a model, use:")
    logger.info("converter = ONNXConverter(model_path, onnx_path, config)")
    logger.info("onnx_model = converter.convert(optimize=True, quantize=False)")


if __name__ == "__main__":
    main()