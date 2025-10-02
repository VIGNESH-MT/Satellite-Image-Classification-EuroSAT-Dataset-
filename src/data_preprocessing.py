"""
Data preprocessing module for EuroSAT satellite image classification.
Handles image loading, preprocessing, and augmentation using OpenCV and TensorFlow.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
from typing import Tuple, List, Optional
from tqdm import tqdm

from config import *

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class EuroSATDataProcessor:
    """
    Data processor for EuroSAT satellite images.
    Handles loading, preprocessing, and augmentation of satellite images.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the EuroSAT dataset directory
        """
        self.data_path = Path(data_path) if data_path else DATA_DIR / "eurosat"
        self.img_height = IMG_HEIGHT
        self.img_width = IMG_WIDTH
        self.classes = EUROSAT_CLASSES
        self.num_classes = NUM_CLASSES
        
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocess a single image using OpenCV.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing (height, width)
            
        Returns:
            Preprocessed image as numpy array
        """
        if target_size is None:
            target_size = (self.img_height, self.img_width)
            
        try:
            # Read image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, (target_size[1], target_size[0]), 
                             interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Apply histogram equalization for better contrast
            image_yuv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def load_dataset(self, subset_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load the EuroSAT dataset from directory structure.
        
        Args:
            subset_size: If specified, load only this many images per class
            
        Returns:
            Tuple of (images, labels, image_paths)
        """
        images = []
        labels = []
        image_paths = []
        
        logger.info(f"Loading dataset from {self.data_path}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.data_path / class_name
            
            if not class_path.exists():
                logger.warning(f"Class directory not found: {class_path}")
                continue
                
            # Get all image files in the class directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif']:
                image_files.extend(class_path.glob(ext))
                
            if subset_size:
                image_files = image_files[:subset_size]
                
            logger.info(f"Loading {len(image_files)} images for class {class_name}")
            
            for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
                try:
                    image = self.preprocess_image(img_path)
                    images.append(image)
                    labels.append(class_idx)
                    image_paths.append(str(img_path))
                except Exception as e:
                    logger.warning(f"Skipping image {img_path}: {str(e)}")
                    continue
        
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} images with shape {images.shape}")
        logger.info(f"Class distribution: {np.bincount(labels)}")
        
        return images, labels, image_paths
    
    def create_data_generators(self, validation_split: float = 0.2) -> Tuple:
        """
        Create data generators for training and validation with augmentation.
        
        Args:
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            validation_split=validation_split,
            rescale=1./255
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(
            validation_split=validation_split,
            rescale=1./255
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            classes=self.classes
        )
        
        validation_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            classes=self.classes
        )
        
        return train_generator, validation_generator
    
    def split_dataset(self, images: np.ndarray, labels: np.ndarray, 
                     test_size: float = 0.2, val_size: float = 0.2) -> Tuple:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            images: Array of preprocessed images
            labels: Array of labels
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining data)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Convert labels to categorical
        labels_categorical = to_categorical(labels, num_classes=self.num_classes)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels_categorical, test_size=test_size, 
            stratify=labels, random_state=42
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            stratify=np.argmax(y_temp, axis=1), random_state=42
        )
        
        logger.info(f"Dataset split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def download_eurosat_sample(self):
        """
        Download a sample of EuroSAT dataset for demonstration.
        Note: This is a placeholder. In practice, you would download from the official source.
        """
        logger.info("For the full EuroSAT dataset, please download from:")
        logger.info("https://github.com/phelber/EuroSAT")
        logger.info("Or use TensorFlow Datasets: tfds.load('eurosat/rgb')")
        
        # Create sample directory structure for demonstration
        for class_name in self.classes:
            class_dir = self.data_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Created sample directory structure at {self.data_path}")


def preprocess_single_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image ready for model prediction
    """
    processor = EuroSATDataProcessor()
    image = processor.preprocess_image(image_path, target_size)
    return np.expand_dims(image, axis=0)  # Add batch dimension


if __name__ == "__main__":
    # Example usage
    processor = EuroSATDataProcessor()
    processor.download_eurosat_sample()
    
    # If you have the dataset, uncomment the following:
    # images, labels, paths = processor.load_dataset(subset_size=100)  # Load 100 images per class
    # X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(images, labels)
    # print(f"Training set shape: {X_train.shape}")
    # print(f"Validation set shape: {X_val.shape}")
    # print(f"Test set shape: {X_test.shape}")
