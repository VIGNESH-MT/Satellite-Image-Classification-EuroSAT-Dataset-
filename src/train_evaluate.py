"""
Training and evaluation script for EuroSAT satellite image classification.
Handles model training, evaluation, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

from config import *
from models import create_model, fine_tune_model
from data_preprocessing import EuroSATDataProcessor

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training and evaluation of satellite image classification models.
    """
    
    def __init__(self, data_processor: EuroSATDataProcessor):
        """
        Initialize the trainer.
        
        Args:
            data_processor: Data processor instance
        """
        self.data_processor = data_processor
        self.models = {}
        self.results = {}
        
    def train_model(self, model_name: str, train_data, validation_data, 
                   epochs: int = EPOCHS, fine_tune: bool = True) -> Dict[str, Any]:
        """
        Train a model with optional fine-tuning.
        
        Args:
            model_name: Name of the model ('vgg16' or 'resnet50')
            train_data: Training data
            validation_data: Validation data
            epochs: Number of training epochs
            fine_tune: Whether to perform fine-tuning
            
        Returns:
            Training results
        """
        logger.info(f"Starting training for {model_name}")
        
        # Create and build model
        model = create_model(model_name)
        model.build_model()
        model.compile_model()
        
        # Initial training with frozen base
        logger.info("Phase 1: Training with frozen base layers")
        history1 = model.train(train_data, validation_data, epochs=epochs//2)
        
        results = {
            'model_name': model_name,
            'phase1_history': history1,
            'phase1_epochs': epochs//2
        }
        
        # Fine-tuning phase
        if fine_tune:
            logger.info("Phase 2: Fine-tuning with unfrozen top layers")
            history2 = fine_tune_model(
                model, train_data, validation_data,
                trainable_layers=10, epochs=epochs//2, learning_rate=1e-5
            )
            results['phase2_history'] = history2
            results['phase2_epochs'] = epochs//2
        
        # Store model and results
        self.models[model_name] = model
        self.results[model_name] = results
        
        logger.info(f"Training completed for {model_name}")
        return results
        
    def evaluate_model(self, model_name: str, test_data) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model_name: Name of the model
            test_data: Test data (X_test, y_test) or generator
            
        Returns:
            Evaluation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train it first.")
            
        model = self.models[model_name]
        
        # Basic evaluation
        eval_results = model.evaluate(test_data)
        
        # Detailed predictions for additional metrics
        if isinstance(test_data, tuple):
            X_test, y_test = test_data
            predictions = model.predict(X_test)
        else:
            # For generators, we need to predict on all batches
            predictions = model.model.predict(test_data)
            y_test = []
            test_data.reset()
            for i in range(len(test_data)):
                _, batch_labels = test_data[i]
                y_test.extend(batch_labels)
            y_test = np.array(y_test)
        
        # Convert predictions to class indices
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
        
        # Calculate additional metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            y_true, y_pred, 
            target_names=EUROSAT_CLASSES,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        detailed_results = {
            **eval_results,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': predictions.tolist(),
            'true_labels': y_true.tolist()
        }
        
        # Update stored results
        if model_name in self.results:
            self.results[model_name]['evaluation'] = detailed_results
        else:
            self.results[model_name] = {'evaluation': detailed_results}
            
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test F1-Score: {f1:.4f}")
        
        return detailed_results
        
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'evaluation' in results:
                eval_data = results['evaluation']
                comparison_data.append({
                    'Model': model_name,
                    'Test Accuracy': eval_data['accuracy'],
                    'Test Loss': eval_data['test_loss'],
                    'Precision': eval_data['precision'],
                    'Recall': eval_data['recall'],
                    'F1-Score': eval_data['f1_score']
                })
        
        df = pd.DataFrame(comparison_data)
        logger.info("Model comparison:")
        logger.info(f"\n{df.to_string(index=False)}")
        
        return df
        
    def plot_training_history(self, model_name: str, save_path: str = None):
        """
        Plot training history for a model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model {model_name}")
            
        results = self.results[model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name.upper()}', fontsize=16)
        
        # Combine histories if fine-tuning was performed
        if 'phase2_history' in results:
            history1 = results['phase1_history']
            history2 = results['phase2_history']
            
            # Combine metrics
            combined_history = {}
            for key in history1.keys():
                combined_history[key] = history1[key] + history2[key]
                
            epochs1 = results['phase1_epochs']
            total_epochs = epochs1 + results['phase2_epochs']
            
            # Add vertical line to show fine-tuning start
            for ax in axes.flat:
                ax.axvline(x=epochs1, color='red', linestyle='--', alpha=0.7, 
                          label='Fine-tuning starts')
        else:
            combined_history = results['phase1_history']
            total_epochs = results['phase1_epochs']
        
        epochs = range(1, total_epochs + 1)
        
        # Plot accuracy
        axes[0, 0].plot(epochs, combined_history['accuracy'], 'b-', label='Training')
        axes[0, 0].plot(epochs, combined_history['val_accuracy'], 'r-', label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(epochs, combined_history['loss'], 'b-', label='Training')
        axes[0, 1].plot(epochs, combined_history['val_loss'], 'r-', label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot top-k accuracy if available
        if 'top_k_categorical_accuracy' in combined_history:
            axes[1, 0].plot(epochs, combined_history['top_k_categorical_accuracy'], 
                           'b-', label='Training Top-K')
            axes[1, 0].plot(epochs, combined_history['val_top_k_categorical_accuracy'], 
                           'r-', label='Validation Top-K')
            axes[1, 0].set_title('Top-K Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-K Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot learning rate if available
        if 'lr' in combined_history:
            axes[1, 1].plot(epochs, combined_history['lr'], 'g-')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
        
    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if model_name not in self.results or 'evaluation' not in self.results[model_name]:
            raise ValueError(f"No evaluation results found for model {model_name}")
            
        conf_matrix = np.array(self.results[model_name]['evaluation']['confusion_matrix'])
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=EUROSAT_CLASSES, yticklabels=EUROSAT_CLASSES)
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
            
        plt.show()
        
    def save_results(self, filepath: str):
        """
        Save training and evaluation results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in self.results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[model_name][key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_results[model_name][key][k] = v.tolist()
                        else:
                            serializable_results[model_name][key][k] = v
                else:
                    serializable_results[model_name][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")


def main():
    """
    Main training and evaluation pipeline.
    """
    logger.info("Starting EuroSAT classification training pipeline")
    
    # Initialize data processor
    data_processor = EuroSATDataProcessor()
    
    # Check if dataset exists
    if not data_processor.data_path.exists():
        logger.warning("EuroSAT dataset not found. Creating sample structure...")
        data_processor.download_eurosat_sample()
        logger.info("Please download the actual EuroSAT dataset and place it in the data directory")
        return
    
    # Load and split dataset
    logger.info("Loading dataset...")
    images, labels, paths = data_processor.load_dataset(subset_size=500)  # Use subset for demo
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_dataset(images, labels)
    
    # Initialize trainer
    trainer = ModelTrainer(data_processor)
    
    # Train models
    models_to_train = ['vgg16', 'resnet50']
    
    for model_name in models_to_train:
        logger.info(f"Training {model_name}...")
        trainer.train_model(
            model_name, 
            (X_train, y_train), 
            (X_val, y_val),
            epochs=20,  # Reduced for demo
            fine_tune=True
        )
        
        # Evaluate model
        logger.info(f"Evaluating {model_name}...")
        trainer.evaluate_model(model_name, (X_test, y_test))
        
        # Plot results
        trainer.plot_training_history(model_name, 
                                    save_path=LOGS_DIR / f"{model_name}_training_history.png")
        trainer.plot_confusion_matrix(model_name,
                                    save_path=LOGS_DIR / f"{model_name}_confusion_matrix.png")
    
    # Compare models
    comparison_df = trainer.compare_models()
    comparison_df.to_csv(LOGS_DIR / "model_comparison.csv", index=False)
    
    # Save results
    trainer.save_results(LOGS_DIR / "training_results.json")
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
