#!/usr/bin/env python3
"""
Cleanup Script for PII Redactor
Removes all training artifacts to start fresh
"""

import os
import shutil
import argparse
from pathlib import Path


def cleanup_directory(path: str, description: str):
    """Remove all contents from a directory."""
    if os.path.exists(path):
        try:
            if os.path.isdir(path):
                # Remove all contents but keep the directory
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                print(f"✓ Cleaned {description}: {path}")
            else:
                os.remove(path)
                print(f"✓ Removed {description}: {path}")
        except Exception as e:
            print(f"✗ Error cleaning {description}: {e}")
    else:
        print(f"- {description} not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Clean up PII Redactor training artifacts")
    parser.add_argument("--all", action="store_true", help="Clean all artifacts (default)")
    parser.add_argument("--models", action="store_true", help="Clean only model checkpoints")
    parser.add_argument("--data", action="store_true", help="Clean only dataset")
    parser.add_argument("--logs", action="store_true", help="Clean only logs")
    parser.add_argument("--cache", action="store_true", help="Clean Hugging Face cache")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    
    args = parser.parse_args()
    
    # If no specific option, clean all
    if not any([args.models, args.data, args.logs, args.cache]):
        args.all = True
    
    print("PII Redactor Cleanup")
    print("=" * 50)
    
    if args.dry_run:
        print("DRY RUN MODE - Nothing will be deleted")
        print("=" * 50)
    
    # Define cleanup targets
    cleanup_targets = []
    
    if args.all or args.models:
        cleanup_targets.extend([
            ("models/checkpoints/pii-redaction-model", "PyTorch checkpoints"),
            ("models/onnx/pii-redaction-model", "ONNX models"),
        ])
    
    if args.all or args.data:
        cleanup_targets.extend([
            ("data/processed", "Processed datasets"),
            ("data/raw", "Raw datasets"),
        ])
    
    if args.all or args.logs:
        cleanup_targets.extend([
            ("logs", "Training logs"),
            ("wandb", "Weights & Biases logs"),
            ("runs", "TensorBoard runs"),
        ])
    
    if args.cache:
        # Get HuggingFace cache directory
        hf_cache = Path.home() / ".cache" / "huggingface"
        if hf_cache.exists():
            response = input(f"\nWarning: This will clear ALL Hugging Face cache at {hf_cache}\nContinue? (y/N): ")
            if response.lower() == 'y':
                cleanup_targets.append((str(hf_cache / "hub"), "Hugging Face model cache"))
    
    # Perform cleanup
    if not args.dry_run:
        for path, description in cleanup_targets:
            cleanup_directory(path, description)
    else:
        print("\nWould clean the following:")
        for path, description in cleanup_targets:
            if os.path.exists(path):
                if os.path.isdir(path):
                    size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(path)
                             for filename in filenames)
                    size_mb = size / 1024 / 1024
                    print(f"  - {description}: {path} ({size_mb:.1f} MB)")
                else:
                    size_mb = os.path.getsize(path) / 1024 / 1024
                    print(f"  - {description}: {path} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 50)
    
    if not args.dry_run:
        print("Cleanup complete! You can now run 'python main.py' to start fresh training.")
    else:
        print("Dry run complete. Run without --dry-run to actually clean.")
    
    # Show next steps
    print("\nNext steps:")
    print("1. Ensure config.yaml has the desired model (e.g., distilbert-base-multilingual-cased)")
    print("2. Run: python main.py")


if __name__ == "__main__":
    main()