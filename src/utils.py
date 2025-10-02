"""
Utility functions for the EuroSAT satellite image classification project.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import requests
from PIL import Image
import cv2

from config import *

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def setup_directories():
    """
    Create necessary directories if they don't exist.
    """
    directories = [DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")


def download_sample_images(num_samples: int = 5) -> List[str]:
    """
    Download sample satellite images for testing.
    
    Args:
        num_samples: Number of sample images to download
        
    Returns:
        List of downloaded image paths
    """
    sample_urls = [
        "https://via.placeholder.com/224x224/228B22/FFFFFF?text=Forest",
        "https://via.placeholder.com/224x224/8FBC8F/FFFFFF?text=Crop",
        "https://via.placeholder.com/224x224/4682B4/FFFFFF?text=Water",
        "https://via.placeholder.com/224x224/696969/FFFFFF?text=Urban",
        "https://via.placeholder.com/224x224/DEB887/FFFFFF?text=Desert"
    ]
    
    downloaded_paths = []
    sample_dir = DATA_DIR / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    for i, url in enumerate(sample_urls[:num_samples]):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                filename = f"sample_{i+1}.png"
                filepath = sample_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded_paths.append(str(filepath))
                logger.info(f"Downloaded sample image: {filename}")
                
        except Exception as e:
            logger.error(f"Error downloading sample image {i+1}: {str(e)}")
    
    return downloaded_paths


def validate_image(image_path: str) -> Tuple[bool, str]:
    """
    Validate if an image file is suitable for classification.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not Path(image_path).exists():
            return False, "File does not exist"
        
        # Check file size
        file_size = Path(image_path).stat().st_size
        if file_size > MAX_CONTENT_LENGTH:
            return False, f"File too large. Maximum size: {MAX_CONTENT_LENGTH / (1024*1024):.1f}MB"
        
        # Check if it's a valid image
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            return False, "Invalid image file"
        
        # Check image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 32 or height < 32:
                return False, "Image too small. Minimum size: 32x32 pixels"
            
            if width > 4096 or height > 4096:
                return False, "Image too large. Maximum size: 4096x4096 pixels"
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"


def create_model_summary(model_results: Dict[str, Any]) -> str:
    """
    Create a formatted summary of model results.
    
    Args:
        model_results: Dictionary containing model evaluation results
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=" * 60)
    summary.append("MODEL EVALUATION SUMMARY")
    summary.append("=" * 60)
    
    for model_name, results in model_results.items():
        summary.append(f"\n{model_name.upper()} MODEL:")
        summary.append("-" * 30)
        
        if 'evaluation' in results:
            eval_data = results['evaluation']
            summary.append(f"Test Accuracy: {eval_data['accuracy']:.4f}")
            summary.append(f"Test Loss: {eval_data['test_loss']:.4f}")
            summary.append(f"Precision: {eval_data['precision']:.4f}")
            summary.append(f"Recall: {eval_data['recall']:.4f}")
            summary.append(f"F1-Score: {eval_data['f1_score']:.4f}")
        
        if 'phase1_history' in results:
            phase1 = results['phase1_history']
            final_acc = phase1['val_accuracy'][-1]
            summary.append(f"Final Validation Accuracy (Phase 1): {final_acc:.4f}")
        
        if 'phase2_history' in results:
            phase2 = results['phase2_history']
            final_acc = phase2['val_accuracy'][-1]
            summary.append(f"Final Validation Accuracy (Phase 2): {final_acc:.4f}")
    
    summary.append("\n" + "=" * 60)
    return "\n".join(summary)


def plot_class_distribution(labels: np.ndarray, save_path: str = None):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        labels: Array of class labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Count occurrences of each class
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [EUROSAT_CLASSES[i] for i in unique]
    
    # Create bar plot
    bars = plt.bar(class_names, counts, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', va='bottom')
    
    plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Land Use Classes', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def create_prediction_report(predictions: np.ndarray, true_labels: np.ndarray, 
                           class_names: List[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive prediction report.
    
    Args:
        predictions: Model predictions (probabilities)
        true_labels: True class labels
        class_names: List of class names
        
    Returns:
        Dictionary containing prediction analysis
    """
    if class_names is None:
        class_names = EUROSAT_CLASSES
    
    # Convert predictions to class indices
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1) if true_labels.ndim > 1 else true_labels
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import classification_report, confusion_matrix
    
    accuracy = accuracy_score(true_classes, pred_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_classes, pred_classes, average=None
    )
    
    # Per-class analysis
    class_analysis = {}
    for i, class_name in enumerate(class_names):
        class_mask = true_classes == i
        if np.sum(class_mask) > 0:
            class_predictions = predictions[class_mask]
            class_confidence = np.mean(np.max(class_predictions, axis=1))
            
            class_analysis[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i],
                'avg_confidence': class_confidence
            }
    
    # Overall statistics
    report = {
        'overall_accuracy': accuracy,
        'macro_precision': np.mean(precision),
        'macro_recall': np.mean(recall),
        'macro_f1': np.mean(f1),
        'class_analysis': class_analysis,
        'confusion_matrix': confusion_matrix(true_classes, pred_classes).tolist(),
        'classification_report': classification_report(
            true_classes, pred_classes, target_names=class_names, output_dict=True
        )
    }
    
    return report


def save_experiment_config(config: Dict[str, Any], filepath: str):
    """
    Save experiment configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Experiment configuration saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")


def load_experiment_config(filepath: str) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        logger.info(f"Experiment configuration loaded from {filepath}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


def calculate_model_complexity(model) -> Dict[str, Any]:
    """
    Calculate model complexity metrics.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with complexity metrics
    """
    try:
        total_params = model.count_params()
        trainable_params = sum([np.prod(var.shape) for var in model.trainable_variables])
        non_trainable_params = total_params - trainable_params
        
        # Estimate model size in MB (rough approximation)
        model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        
        complexity = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'estimated_size_mb': model_size_mb,
            'num_layers': len(model.layers)
        }
        
        return complexity
        
    except Exception as e:
        logger.error(f"Error calculating model complexity: {str(e)}")
        return {}


def create_data_augmentation_preview(image_path: str, save_path: str = None):
    """
    Create a preview of data augmentation effects.
    
    Args:
        image_path: Path to the input image
        save_path: Path to save the preview
    """
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image.astype(np.float32) / 255.0
        
        # Create data generator
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2]
        )
        
        # Generate augmented images
        image_batch = np.expand_dims(image, axis=0)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Augmented images
        i = 1
        for batch in datagen.flow(image_batch, batch_size=1):
            if i >= 8:
                break
            axes[i].imshow(batch[0])
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
            i += 1
        
        plt.suptitle('Data Augmentation Preview', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Augmentation preview saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating augmentation preview: {str(e)}")


if __name__ == "__main__":
    # Setup directories
    setup_directories()
    
    # Download sample images
    sample_paths = download_sample_images()
    print(f"Downloaded {len(sample_paths)} sample images")
    
    # Test image validation
    for path in sample_paths:
        is_valid, message = validate_image(path)
        print(f"{path}: {message}")
    
    print("Utility functions test completed!")
