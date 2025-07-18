#!/usr/bin/env python3
"""
Test Inference Script for PII Redaction Model

This script demonstrates how to use the trained PII redaction model
for inference with various examples in multiple languages.

Usage:
    python test_inference.py [--model-path MODEL_PATH] [--use-onnx] [--interactive]
"""

import os
import sys
import argparse
import logging
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference import PIIRedactor, BatchProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_test_examples() -> List[Dict]:
    """
    Get comprehensive test examples in multiple languages.
    
    Returns:
        List of test examples with language metadata
    """
    return [
        # English Examples
        {
            "text": "My name is John Doe and my SSN is 123-45-6789. You can reach me at john.doe@example.com or call (555) 123-4567.",
            "language": "English",
            "description": "Multiple PII types"
        },
        {
            "text": "Customer information: Sarah Johnson, DOB: 03/15/1985, Credit Card: 4532-1234-5678-9012, Address: 123 Oak Street, Springfield, IL 62701",
            "language": "English",
            "description": "Customer record format"
        },
        {
            "text": "For urgent matters, contact Dr. Michael Smith at michael.smith@hospital.org or his mobile 555-987-6543.",
            "language": "English",
            "description": "Professional contact"
        },
        
        # Hebrew Examples
        {
            "text": "שמי אלון כהן ומספר תעודת הזהות שלי הוא 123456789. הטלפון שלי 050-1234567 והמייל alon.cohen@example.co.il",
            "language": "Hebrew",
            "description": "Basic personal information"
        },
        {
            "text": "פרטי הלקוח: שרה לוי, נולדה ב-15 במרץ 1985, כתובת: רחוב הרצל 45, תל אביב. טלפון: 03-1234567",
            "language": "Hebrew",
            "description": "Customer details in Hebrew"
        },
        {
            "text": "לפרטים נוספים צרו קשר עם ד\"ר דוד מזרחי במייל david.mizrahi@clinic.co.il או בטלפון 052-9876543",
            "language": "Hebrew",
            "description": "Professional contact in Hebrew"
        },
        {
            "text": "מספר רישיון הרכב: 123-45-678, מספר דרכון: AB1234567, חשבון בנק: 12-345-678901",
            "language": "Hebrew",
            "description": "Various ID numbers"
        },
        
        # Spanish Examples
        {
            "text": "Mi nombre es María García y mi número de teléfono es 91-234-5678. Puedes contactarme en maria.garcia@correo.es",
            "language": "Spanish",
            "description": "Personal contact information"
        },
        {
            "text": "Datos del cliente: Pedro Martínez, nacido el 20/07/1990, dirección: Calle Mayor 123, Madrid, España",
            "language": "Spanish",
            "description": "Customer data"
        },
        
        # French Examples
        {
            "text": "Je m'appelle Pierre Dubois et mon numéro de téléphone est 01-23-45-67-89. Mon email est pierre.dubois@mail.fr",
            "language": "French",
            "description": "Personal information"
        },
        {
            "text": "Informations client: Marie Lefebvre, née le 12/08/1988, adresse: 15 Rue de la Paix, Paris, France",
            "language": "French",
            "description": "Client information"
        },
        
        # German Examples
        {
            "text": "Ich heiße Hans Mueller und meine Telefonnummer ist 030-12345678. Meine E-Mail ist hans.mueller@email.de",
            "language": "German",
            "description": "Personal contact"
        },
        {
            "text": "Kundendaten: Anna Schmidt, geboren am 05.04.1992, Adresse: Hauptstraße 67, Berlin, Deutschland",
            "language": "German",
            "description": "Customer data"
        },
        
        # Mixed Language Examples
        {
            "text": "Contact info: John Smith (john@example.com) and שרה כהן (sarah@example.co.il), phone: 050-1234567",
            "language": "Mixed (English-Hebrew)",
            "description": "Multilingual contact list"
        },
        
        # Edge Cases
        {
            "text": "This text contains no PII information, just regular content about technology and science.",
            "language": "English",
            "description": "No PII (negative test)"
        },
        {
            "text": "Partial info: My name is J*** D** and my phone is 555-***-****",
            "language": "English",
            "description": "Partially redacted PII"
        },
        {
            "text": "אין כאן מידע אישי, רק תוכן רגיל על טכנולוgia ומדע.",
            "language": "Hebrew",
            "description": "No PII in Hebrew"
        }
    ]


def test_basic_redaction(redactor: PIIRedactor, examples: List[Dict]):
    """
    Test basic redaction functionality.
    
    Args:
        redactor: PIIRedactor instance
        examples: List of test examples
    """
    print("\n" + "="*60)
    print("BASIC REDACTION TEST")
    print("="*60)
    
    for i, example in enumerate(examples, 1):
        print(f"\nTest {i}: {example['description']} ({example['language']})")
        print("-" * 50)
        
        original = example['text']
        redacted = redactor.redact(original)
        
        print(f"Original:  {original}")
        print(f"Redacted:  {redacted}")
        
        # Check if redaction occurred
        if original != redacted:
            print("✓ PII detected and redacted")
        else:
            print("○ No PII detected")


def test_detailed_analysis(redactor: PIIRedactor, examples: List[Dict]):
    """
    Test detailed analysis functionality.
    
    Args:
        redactor: PIIRedactor instance
        examples: List of test examples
    """
    print("\n" + "="*60)
    print("DETAILED ANALYSIS TEST")
    print("="*60)
    
    for i, example in enumerate(examples[:5], 1):  # Test first 5 examples
        print(f"\nAnalysis {i}: {example['description']} ({example['language']})")
        print("-" * 50)
        
        text = example['text']
        analysis = redactor.analyze_text(text)
        
        print(f"Text: {text}")
        print(f"Total entities found: {analysis['total_entities']}")
        print(f"PII character ratio: {analysis['pii_character_ratio']:.2%}")
        print(f"Average confidence: {analysis['average_confidence']:.3f}")
        
        if analysis['entity_counts']:
            print("Entity types found:")
            for entity_type, count in analysis['entity_counts'].items():
                print(f"  - {entity_type}: {count}")
        
        if analysis['entities']:
            print("Detailed entities:")
            for entity in analysis['entities']:
                print(f"  - '{entity['text']}' ({entity['type']}) - confidence: {entity['confidence']:.3f}")


def test_redaction_with_info(redactor: PIIRedactor, examples: List[Dict]):
    """
    Test redaction with entity information.
    
    Args:
        redactor: PIIRedactor instance
        examples: List of test examples
    """
    print("\n" + "="*60)
    print("REDACTION WITH ENTITY INFO TEST")
    print("="*60)
    
    for i, example in enumerate(examples[:3], 1):  # Test first 3 examples
        print(f"\nTest {i}: {example['description']} ({example['language']})")
        print("-" * 50)
        
        text = example['text']
        redacted_text, entities = redactor.redact_with_info(text)
        
        print(f"Original: {text}")
        print(f"Redacted: {redacted_text}")
        print(f"Entities found: {len(entities)}")
        
        for j, entity in enumerate(entities, 1):
            print(f"  Entity {j}: '{entity.text}' ({entity.entity_type}) "
                  f"at position {entity.start}-{entity.end} "
                  f"(confidence: {entity.confidence:.3f})")


def test_batch_processing(redactor: PIIRedactor, examples: List[Dict]):
    """
    Test batch processing functionality.
    
    Args:
        redactor: PIIRedactor instance
        examples: List of test examples
    """
    print("\n" + "="*60)
    print("BATCH PROCESSING TEST")
    print("="*60)
    
    # Create batch processor
    batch_processor = BatchProcessor(redactor, batch_size=4)
    
    # Extract texts
    texts = [example['text'] for example in examples]
    
    print(f"Processing {len(texts)} texts in batches...")
    
    # Process batch
    import time
    start_time = time.time()
    results = batch_processor.process_batch(texts)
    end_time = time.time()
    
    print(f"Batch processing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per text: {(end_time - start_time) / len(texts):.4f} seconds")
    
    # Show some results
    print("\nSample results:")
    for i, (original, (redacted, entities)) in enumerate(zip(texts[:3], results[:3])):
        print(f"\nText {i+1}:")
        print(f"  Original: {original}")
        print(f"  Redacted: {redacted}")
        print(f"  Entities: {len(entities)}")


def test_performance_benchmark(redactor: PIIRedactor, examples: List[Dict]):
    """
    Test performance benchmarking.
    
    Args:
        redactor: PIIRedactor instance
        examples: List of test examples
    """
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    texts = [example['text'] for example in examples]
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        for text in texts[:3]:
            _ = redactor.predict(text)
    
    # Benchmark
    print("Running benchmark...")
    import time
    
    num_runs = 10
    times = []
    
    for run in range(num_runs):
        start_time = time.time()
        for text in texts:
            _ = redactor.predict(text)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Benchmark results ({num_runs} runs):")
    print(f"  Average time: {avg_time:.4f} seconds")
    print(f"  Min time: {min_time:.4f} seconds")
    print(f"  Max time: {max_time:.4f} seconds")
    print(f"  Throughput: {len(texts) / avg_time:.2f} texts/second")
    print(f"  Average per text: {avg_time / len(texts):.4f} seconds")


def interactive_mode(redactor: PIIRedactor):
    """
    Interactive mode for testing custom inputs.
    
    Args:
        redactor: PIIRedactor instance
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter text to test PII redaction (type 'quit' to exit)")
    print("Commands:")
    print("  'quit' or 'exit' - Exit interactive mode")
    print("  'analyze' - Show detailed analysis")
    print("  'info' - Show redaction with entity info")
    print("-" * 60)
    
    while True:
        try:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit']:
                break
            
            if text.lower() == 'analyze':
                text = input("Enter text to analyze: ").strip()
                analysis = redactor.analyze_text(text)
                print(f"\nAnalysis results:")
                print(f"  Total entities: {analysis['total_entities']}")
                print(f"  PII ratio: {analysis['pii_character_ratio']:.2%}")
                print(f"  Average confidence: {analysis['average_confidence']:.3f}")
                if analysis['entities']:
                    print("  Entities found:")
                    for entity in analysis['entities']:
                        print(f"    - '{entity['text']}' ({entity['type']}) - {entity['confidence']:.3f}")
                continue
            
            if text.lower() == 'info':
                text = input("Enter text to redact: ").strip()
                redacted, entities = redactor.redact_with_info(text)
                print(f"\nOriginal: {text}")
                print(f"Redacted: {redacted}")
                print(f"Entities found: {len(entities)}")
                for entity in entities:
                    print(f"  - '{entity.text}' ({entity.entity_type}) at {entity.start}-{entity.end}")
                continue
            
            if not text:
                continue
            
            # Regular redaction
            redacted = redactor.redact(text)
            print(f"Original: {text}")
            print(f"Redacted: {redacted}")
            
            if text != redacted:
                print("✓ PII detected and redacted")
            else:
                print("○ No PII detected")
                
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run inference tests"""
    parser = argparse.ArgumentParser(description='Test PII Redaction Model Inference')
    parser.add_argument('--model-path', 
                       default='models/onnx/pii-redaction-model', 
                       help='Path to model directory')
    parser.add_argument('--use-onnx', action='store_true', default=True,
                       help='Use ONNX model (default: True)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--test-suite', 
                       choices=['basic', 'detailed', 'batch', 'performance', 'all'],
                       default='all',
                       help='Test suite to run')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        print("Please train the model first using: python main.py")
        return
    
    print("="*60)
    print("PII REDACTION MODEL - INFERENCE TEST")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Using ONNX: {args.use_onnx}")
    print(f"Test suite: {args.test_suite}")
    
    try:
        # Initialize redactor
        print("\nInitializing redactor...")
        redactor = PIIRedactor(
            model_path=args.model_path,
            use_onnx=args.use_onnx,
            confidence_threshold=0.5
        )
        
        # Get test examples
        examples = get_test_examples()
        
        # Run tests based on selection
        if args.test_suite in ['basic', 'all']:
            test_basic_redaction(redactor, examples)
        
        if args.test_suite in ['detailed', 'all']:
            test_detailed_analysis(redactor, examples)
        
        if args.test_suite in ['batch', 'all']:
            test_batch_processing(redactor, examples)
        
        if args.test_suite in ['performance', 'all']:
            test_performance_benchmark(redactor, examples)
        
        # Interactive mode
        if args.interactive:
            interactive_mode(redactor)
        
        print("\n" + "="*60)
        print("INFERENCE TESTING COMPLETED")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()