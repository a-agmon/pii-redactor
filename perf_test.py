#!/usr/bin/env python3
"""
Performance Testing Script for PII Redaction Model

This script benchmarks the PII redaction model performance by running
multiple inferences and measuring various metrics including throughput,
latency, and resource usage.

Usage:
    python perf_test.py                    # Run with default settings
    python perf_test.py --iterations 1000  # Custom number of iterations
    python perf_test.py --model pytorch    # Test PyTorch model only
    python perf_test.py --batch-size 32    # Test batch processing
"""

import argparse
import time
import statistics
import json
import os
import sys
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not installed. Memory usage tracking disabled.")
    print("Install with: pip install psutil")

from src.inference import PIIRedactor


class PerformanceTester:
    """Performance testing for PII redaction models."""
    
    def __init__(self, model_path: str, use_onnx: bool = True):
        """
        Initialize performance tester.
        
        Args:
            model_path: Path to model directory
            use_onnx: Whether to use ONNX model
        """
        self.model_path = model_path
        self.use_onnx = use_onnx
        self.model_type = "ONNX" if use_onnx else "PyTorch"
        
        # Test texts of varying lengths
        self.test_texts = [
            # Short texts
            "My name is John Doe.",
            "Contact me at john@example.com",
            "My SSN is 123-45-6789",
            
            # Medium texts
            "Hello, I'm Jane Smith and you can reach me at jane.smith@company.com or call (555) 123-4567.",
            "I live at 123 Main Street, New York, NY 10001. My ID number is ABC123456.",
            "Please send the invoice to my address: 456 Oak Avenue, Boston, MA 02101.",
            
            # Long texts
            "My name is Robert Johnson, and I work at Tech Corp. You can contact me at rjohnson@techcorp.com "
            "or by phone at +1-555-987-6543. My office is located at 789 Business Park Drive, Suite 200, "
            "San Francisco, CA 94105. My employee ID is EMP-2023-1234.",
            
            "Dear Customer Service, I am writing to update my account information. My name is Maria Garcia, "
            "date of birth: 01/15/1985. My current address is 321 Elm Street, Apt 4B, Chicago, IL 60601. "
            "Please update my phone number to (312) 555-0123 and my email to mgarcia.new@email.com. "
            "My account number is ACC-789456123.",
            
            # Hebrew text
            "שמי דוד כהן ומספר הטלפון שלי הוא 050-1234567. אני גר ברחוב הרצל 15, תל אביב.",
            
            # Mixed language
            "Hello, my name is יוסי לוי and my email is yossi@example.co.il"
        ]
        
        # Initialize model
        print(f"Loading {self.model_type} model from {model_path}...")
        self.start_memory = self._get_memory_usage()
        load_start = time.time()
        self.redactor = PIIRedactor(model_path, use_onnx=use_onnx)
        self.load_time = time.time() - load_start
        self.model_memory = self._get_memory_usage() - self.start_memory
        print(f"Model loaded in {self.load_time:.2f} seconds")
        print(f"Model memory usage: {self.model_memory:.1f} MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def warmup(self, iterations: int = 10):
        """Warm up the model with a few iterations."""
        print(f"Warming up with {iterations} iterations...")
        for _ in range(iterations):
            self.redactor.redact(self.test_texts[0])
    
    def benchmark_single_inference(self, iterations: int = 100) -> Dict:
        """
        Benchmark single text inference.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking single inference ({iterations} iterations)...")
        
        latencies = []
        text_lengths = []
        
        for i in range(iterations):
            # Select text cyclically
            text = self.test_texts[i % len(self.test_texts)]
            text_lengths.append(len(text))
            
            # Measure inference time
            start_time = time.time()
            redacted, entities = self.redactor.redact_with_info(text)
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{iterations}", end='\r')
        
        print(f"  Completed: {iterations}/{iterations}")
        
        # Calculate statistics
        results = {
            "iterations": iterations,
            "total_time": sum(latencies),
            "average_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "throughput": iterations / sum(latencies),
            "avg_text_length": statistics.mean(text_lengths),
            "latencies": latencies  # Keep for detailed analysis
        }
        
        return results
    
    def benchmark_batch_inference(self, batch_size: int = 32, iterations: int = 10) -> Dict:
        """
        Benchmark batch inference.
        
        Args:
            batch_size: Size of each batch
            iterations: Number of batches to process
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking batch inference (batch_size={batch_size}, iterations={iterations})...")
        
        batch_latencies = []
        total_texts = 0
        
        for i in range(iterations):
            # Create batch
            batch = []
            for j in range(batch_size):
                text_idx = (i * batch_size + j) % len(self.test_texts)
                batch.append(self.test_texts[text_idx])
            
            # Measure batch inference time
            start_time = time.time()
            for text in batch:
                self.redactor.redact(text)
            batch_latency = time.time() - start_time
            batch_latencies.append(batch_latency)
            total_texts += batch_size
            
            # Progress indicator
            print(f"  Progress: {i + 1}/{iterations} batches", end='\r')
        
        print(f"  Completed: {iterations}/{iterations} batches")
        
        # Calculate statistics
        total_time = sum(batch_latencies)
        results = {
            "batch_size": batch_size,
            "iterations": iterations,
            "total_texts": total_texts,
            "total_time": total_time,
            "avg_batch_latency": statistics.mean(batch_latencies),
            "throughput": total_texts / total_time,
            "texts_per_second": total_texts / total_time
        }
        
        return results
    
    def benchmark_text_lengths(self) -> Dict:
        """Benchmark performance across different text lengths."""
        print("\nBenchmarking different text lengths...")
        
        # Create texts of different lengths
        length_tests = [
            ("short", "My name is John Doe."),
            ("medium", "My name is John Doe and my email is john@example.com. Call me at 555-1234."),
            ("long", " ".join(self.test_texts[6:8])),  # Combine long texts
            ("very_long", " ".join(self.test_texts[6:8]) * 2)  # Double long text
        ]
        
        results = {}
        
        for label, text in length_tests:
            latencies = []
            for _ in range(50):
                start_time = time.time()
                self.redactor.redact(text)
                latency = time.time() - start_time
                latencies.append(latency)
            
            results[label] = {
                "text_length": len(text),
                "avg_latency": statistics.mean(latencies),
                "throughput": 1 / statistics.mean(latencies)
            }
            print(f"  {label}: {len(text)} chars, {statistics.mean(latencies)*1000:.2f}ms avg")
        
        return results
    
    def run_comprehensive_benchmark(self, iterations: int = 100) -> Dict:
        """Run comprehensive benchmark suite."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE PERFORMANCE BENCHMARK - {self.model_type}")
        print(f"{'='*60}")
        
        # Warm up
        self.warmup()
        
        # Run benchmarks
        results = {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "load_time": self.load_time,
                "model_memory_mb": self.model_memory,
                "cpu_count": os.cpu_count()
            },
            "single_inference": self.benchmark_single_inference(iterations),
            "batch_inference": self.benchmark_batch_inference(),
            "text_length_analysis": self.benchmark_text_lengths()
        }
        
        # Memory usage after benchmarks
        if PSUTIL_AVAILABLE:
            final_memory = self._get_memory_usage()
            results["system_info"]["peak_memory_mb"] = final_memory
            results["system_info"]["memory_increase_mb"] = final_memory - self.start_memory
        
        return results
    
    def print_summary(self, results: Dict):
        """Print formatted summary of results."""
        print(f"\n{'='*60}")
        print(f"PERFORMANCE SUMMARY - {results['model_type']}")
        print(f"{'='*60}")
        
        single = results["single_inference"]
        print(f"\nSingle Inference Performance:")
        print(f"  Iterations: {single['iterations']}")
        print(f"  Average latency: {single['average_latency']*1000:.2f} ms")
        print(f"  Median latency: {single['median_latency']*1000:.2f} ms")
        print(f"  95th percentile: {single['p95_latency']*1000:.2f} ms")
        print(f"  99th percentile: {single['p99_latency']*1000:.2f} ms")
        print(f"  Min/Max latency: {single['min_latency']*1000:.2f} / {single['max_latency']*1000:.2f} ms")
        print(f"  Throughput: {single['throughput']:.2f} texts/second")
        
        if "batch_inference" in results:
            batch = results["batch_inference"]
            print(f"\nBatch Inference Performance:")
            print(f"  Batch size: {batch['batch_size']}")
            print(f"  Throughput: {batch['throughput']:.2f} texts/second")
            print(f"  Speedup vs single: {batch['throughput']/single['throughput']:.2f}x")
        
        print(f"\nText Length Impact:")
        for label, data in results["text_length_analysis"].items():
            print(f"  {label}: {data['text_length']} chars → {data['avg_latency']*1000:.2f} ms")
        
        print(f"\nSystem Resources:")
        print(f"  Model load time: {results['system_info']['load_time']:.2f} seconds")
        print(f"  Model memory: {results['system_info']['model_memory_mb']:.1f} MB")
        if "peak_memory_mb" in results["system_info"]:
            print(f"  Peak memory: {results['system_info']['peak_memory_mb']:.1f} MB")


def compare_models(pytorch_path: str, onnx_path: str, iterations: int = 100):
    """Compare PyTorch and ONNX model performance."""
    print("\nComparing PyTorch vs ONNX Performance...")
    
    # Test PyTorch
    pytorch_tester = PerformanceTester(pytorch_path, use_onnx=False)
    pytorch_results = pytorch_tester.run_comprehensive_benchmark(iterations)
    
    # Test ONNX
    onnx_tester = PerformanceTester(onnx_path, use_onnx=True)
    onnx_results = onnx_tester.run_comprehensive_benchmark(iterations)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    pt_single = pytorch_results["single_inference"]
    onnx_single = onnx_results["single_inference"]
    
    print(f"\nInference Speed:")
    print(f"  PyTorch: {pt_single['throughput']:.2f} texts/sec ({pt_single['average_latency']*1000:.2f} ms/text)")
    print(f"  ONNX: {onnx_single['throughput']:.2f} texts/sec ({onnx_single['average_latency']*1000:.2f} ms/text)")
    print(f"  ONNX Speedup: {onnx_single['throughput']/pt_single['throughput']:.2f}x")
    
    print(f"\nMemory Usage:")
    print(f"  PyTorch: {pytorch_results['system_info']['model_memory_mb']:.1f} MB")
    print(f"  ONNX: {onnx_results['system_info']['model_memory_mb']:.1f} MB")
    print(f"  Memory Savings: {pytorch_results['system_info']['model_memory_mb'] - onnx_results['system_info']['model_memory_mb']:.1f} MB")
    
    print(f"\nLoad Time:")
    print(f"  PyTorch: {pytorch_results['system_info']['load_time']:.2f} seconds")
    print(f"  ONNX: {onnx_results['system_info']['load_time']:.2f} seconds")
    
    # Save detailed results
    results = {
        "pytorch": pytorch_results,
        "onnx": onnx_results,
        "comparison": {
            "onnx_speedup": onnx_single['throughput'] / pt_single['throughput'],
            "memory_savings_mb": pytorch_results['system_info']['model_memory_mb'] - 
                               onnx_results['system_info']['model_memory_mb'],
            "load_time_diff": onnx_results['system_info']['load_time'] - 
                            pytorch_results['system_info']['load_time']
        }
    }
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"performance_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Performance testing for PII redaction model")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations for single inference benchmark")
    parser.add_argument("--model", choices=["pytorch", "onnx", "both"], default="onnx",
                       help="Which model to test")
    parser.add_argument("--pytorch-path", default="models/checkpoints/pii-redaction-model",
                       help="Path to PyTorch model")
    parser.add_argument("--onnx-path", default="models/onnx/pii-redaction-model",
                       help="Path to ONNX model")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for batch inference testing")
    parser.add_argument("--compare", action="store_true",
                       help="Compare PyTorch and ONNX models")
    
    args = parser.parse_args()
    
    try:
        if args.compare or args.model == "both":
            # Compare both models
            results = compare_models(args.pytorch_path, args.onnx_path, args.iterations)
        else:
            # Test single model
            if args.model == "pytorch":
                tester = PerformanceTester(args.pytorch_path, use_onnx=False)
            else:
                tester = PerformanceTester(args.onnx_path, use_onnx=True)
            
            results = tester.run_comprehensive_benchmark(args.iterations)
            tester.print_summary(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_{args.model}_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {output_file}")
    
    except Exception as e:
        print(f"\nError during performance testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()