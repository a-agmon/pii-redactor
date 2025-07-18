"""
Model Training Module for PII Redaction

This module contains the implementation of the PII redaction model training pipeline.
It uses a multilingual BERT-like model (MiniLM) as the base and fine-tunes it for
token classification to identify PII entities.

Classes:
    PIIRedactionModel: Main model class for training and evaluation
    PIITrainer: Custom trainer with additional logging and callbacks
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from transformers.trainer_callback import TrainerCallback

from sklearn.metrics import classification_report, confusion_matrix
import seqeval.metrics
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomLoggingCallback(TrainerCallback):
    """
    Custom callback for enhanced logging during training.
    
    This callback provides detailed logging of training progress,
    including per-epoch metrics and Hebrew-specific performance tracking.
    """
    
    def __init__(self, log_dir: str):
        """Initialize callback with log directory"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.training_history = []
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Log metrics at the end of each epoch"""
        if state.log_history:
            latest_metrics = state.log_history[-1]
            self.training_history.append(latest_metrics)
            
            # Save training history
            history_path = os.path.join(self.log_dir, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Epoch {state.epoch}: {latest_metrics}")


class PIIRedactionModel:
    """
    Fine-tune multilingual model for PII detection.
    
    This class encapsulates the model architecture, training configuration,
    and evaluation metrics for the PII redaction task.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the PII redaction model.
        
        Args:
            config: Configuration dictionary containing model and training parameters
        """
        self.config = config
        self.num_labels = 3  # O, B-PII, I-PII
        # Use CPU for MPS to avoid "Placeholder storage has not been allocated" error
        if torch.backends.mps.is_available():
            self.device = torch.device("cpu")
            logger.info("MPS available but using CPU to avoid MPS-specific errors")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Load base model
        model_name = config['model']['base_model']
        logger.info(f"Loading base model: {model_name}")
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        special_tokens = ["[PII]", "[/PII]"]
        special_tokens_dict = {"additional_special_tokens": special_tokens}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        
        if num_added_toks > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added_toks} special tokens")
        
        # Label mappings
        self.label_to_id = config['label_map']
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # Initialize metrics storage
        self.best_metrics = {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'epoch': 0
        }
    
    def setup_training_args(self) -> TrainingArguments:
        """
        Configure training arguments.
        
        Returns:
            TrainingArguments object with all training hyperparameters
        """
        training_config = self.config['training']
        output_dir = self.config['output']['model_path']
        
        return TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=float(training_config['learning_rate']),
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            num_train_epochs=training_config['num_epochs'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            logging_dir=self.config['output']['logs_dir'],
            logging_steps=training_config['logging_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            fp16=torch.cuda.is_available(),  # Enable mixed precision training if GPU available
            save_total_limit=training_config['save_total_limit'],
            seed=training_config['seed'],
            run_name="pii-redaction-training",
            report_to=["tensorboard"],  # Enable tensorboard logging
            gradient_checkpointing=True,  # Save memory
            optim="adamw_torch",  # Use torch implementation of AdamW
            remove_unused_columns=False,
            label_names=["labels"]
        )
    
    def compute_metrics(self, eval_pred) -> Dict:
        """
        Compute evaluation metrics for token classification.
        
        This function calculates precision, recall, and F1 scores for the
        PII detection task, with special attention to entity-level metrics.
        
        Args:
            eval_pred: EvalPrediction object with predictions and labels
            
        Returns:
            Dictionary containing computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove padding and special tokens
        true_labels = []
        true_predictions = []
        
        for prediction, label in zip(predictions, labels):
            true_label = []
            true_pred = []
            
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:  # -100 is padding/special token
                    true_label.append(self.id_to_label[label_id])
                    true_pred.append(self.id_to_label[pred_id])
            
            true_labels.append(true_label)
            true_predictions.append(true_pred)
        
        # Calculate metrics using seqeval
        results = seqeval.metrics.classification_report(
            true_labels, 
            true_predictions, 
            output_dict=True,
            zero_division=0
        )
        
        # Extract relevant metrics
        metrics = {
            "precision": results.get("weighted avg", {}).get("precision", 0.0),
            "recall": results.get("weighted avg", {}).get("recall", 0.0),
            "f1": results.get("weighted avg", {}).get("f1-score", 0.0),
        }
        
        # Add entity-level metrics if available
        if "PII" in results:
            metrics.update({
                "pii_precision": results["PII"]["precision"],
                "pii_recall": results["PII"]["recall"],
                "pii_f1": results["PII"]["f1-score"],
                "pii_support": results["PII"]["support"]
            })
        
        # Calculate accuracy
        flat_true = [label for seq in true_labels for label in seq]
        flat_pred = [label for seq in true_predictions for label in seq]
        accuracy = sum(1 for t, p in zip(flat_true, flat_pred) if t == p) / len(flat_true)
        metrics["accuracy"] = accuracy
        
        # Log detailed results
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Save detailed classification report
        report_path = os.path.join(self.config['output']['logs_dir'], "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(seqeval.metrics.classification_report(true_labels, true_predictions))
        
        return metrics
    
    def create_data_collator(self) -> DataCollatorForTokenClassification:
        """
        Create data collator for token classification.
        
        Returns:
            DataCollatorForTokenClassification instance
        """
        return DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.device.type == "cuda" else None,
            return_tensors="pt"
        )
    
    def save_model_card(self, output_dir: str, metrics: Dict):
        """
        Save model card with training information and metrics.
        
        Args:
            output_dir: Directory to save the model card
            metrics: Dictionary of evaluation metrics
        """
        model_card = f"""
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

- **Base Model**: {self.config['model']['base_model']}
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

- **Overall F1 Score**: {metrics.get('f1', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **Accuracy**: {metrics.get('accuracy', 0):.4f}

### PII Entity Metrics
- **PII F1 Score**: {metrics.get('pii_f1', 0):.4f}
- **PII Precision**: {metrics.get('pii_precision', 0):.4f}
- **PII Recall**: {metrics.get('pii_recall', 0):.4f}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
model = AutoModelForTokenClassification.from_pretrained("{output_dir}")

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
{yaml.dump(self.config, allow_unicode=True)}
```
"""
        
        # Save model card
        model_card_path = os.path.join(output_dir, "README.md")
        with open(model_card_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        logger.info(f"Model card saved to {model_card_path}")
    
    def analyze_errors(self, dataset, output_dir: str):
        """
        Perform error analysis on the validation dataset.
        
        Args:
            dataset: Validation dataset
            output_dir: Directory to save error analysis results
        """
        logger.info("Performing error analysis...")
        
        # Force model to CPU for error analysis to avoid MPS issues
        original_device = next(self.model.parameters()).device
        self.model.to('cpu')
        self.model.eval()
        error_examples = []
        
        # Create dataloader
        data_collator = self.create_data_collator()
        dataloader = DataLoader(
            dataset, 
            batch_size=16, 
            collate_fn=data_collator
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Error analysis"):
                # Move batch to CPU for error analysis
                batch = {k: v.to('cpu') for k, v in batch.items()}
                
                # Get predictions
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Analyze errors
                for i in range(len(batch["input_ids"])):
                    input_ids = batch["input_ids"][i]
                    labels = batch["labels"][i]
                    preds = predictions[i]
                    
                    # Find mismatches
                    for j, (label_id, pred_id) in enumerate(zip(labels, preds)):
                        if label_id != -100 and label_id != pred_id:
                            token = self.tokenizer.decode([input_ids[j]])
                            true_label = self.id_to_label[label_id.item()]
                            pred_label = self.id_to_label[pred_id.item()]
                            
                            error_examples.append({
                                "token": token,
                                "true_label": true_label,
                                "predicted_label": pred_label,
                                "context": self.tokenizer.decode(input_ids[max(0, j-5):j+6])
                            })
        
        # Restore model to original device
        self.model.to(original_device)
        
        # Save error analysis
        error_analysis_path = os.path.join(output_dir, "error_analysis.json")
        with open(error_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(error_examples[:100], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error analysis saved to {error_analysis_path}")
        logger.info(f"Found {len(error_examples)} errors in validation set")


def create_trainer(
    model: PIIRedactionModel,
    train_dataset,
    eval_dataset,
    output_dir: str
) -> Trainer:
    """
    Create and configure the Trainer instance.
    
    Args:
        model: PIIRedactionModel instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Output directory for checkpoints
        
    Returns:
        Configured Trainer instance
    """
    # Setup training arguments
    training_args = model.setup_training_args()
    
    # Create data collator
    data_collator = model.create_data_collator()
    
    # Setup callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),
        CustomLoggingCallback(log_dir=model.config['output']['logs_dir'])
    ]
    
    # Create trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
        compute_metrics=model.compute_metrics,
        callbacks=callbacks
    )
    
    return trainer


def train_model(
    model: PIIRedactionModel,
    train_dataset,
    eval_dataset,
    test_dataset=None
) -> Dict:
    """
    Train the PII redaction model.
    
    Args:
        model: PIIRedactionModel instance
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        test_dataset: Optional test dataset
        
    Returns:
        Dictionary containing training results and metrics
    """
    output_dir = model.config['output']['model_path']
    
    # Create trainer
    trainer = create_trainer(model, train_dataset, eval_dataset, output_dir)
    
    # Train model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    model.tokenizer.save_pretrained(output_dir)
    
    # Save training results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Evaluate on test set if provided
    if test_dataset:
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
    else:
        test_metrics = {}
    
    # Save model configuration
    model.model.config.id2label = model.id_to_label
    model.model.config.label2id = model.label_to_id
    model.model.config.save_pretrained(output_dir)
    
    # Save model card
    all_metrics = {**eval_metrics, **test_metrics}
    model.save_model_card(output_dir, all_metrics)
    
    # Perform error analysis
    model.analyze_errors(eval_dataset, output_dir)
    
    # Return results
    results = {
        "train_metrics": metrics,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
        "model_path": output_dir
    }
    
    return results


def main():
    """Test model training with dummy data"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create model
    model = PIIRedactionModel(config)
    logger.info(f"Model initialized: {model.model}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.model.parameters()):,}")


if __name__ == "__main__":
    main()