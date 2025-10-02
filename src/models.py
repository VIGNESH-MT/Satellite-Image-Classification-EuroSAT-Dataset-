"""
CNN models for EuroSAT satellite image classification.
Implements VGG16 and ResNet50 with transfer learning.
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

from config import *

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class SatelliteImageClassifier:
    """
    Base class for satellite image classification models.
    """
    
    def __init__(self, model_name: str, num_classes: int = NUM_CLASSES):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the model ('vgg16' or 'resnet50')
            num_classes: Number of output classes
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the model architecture.
        To be implemented by subclasses.
        """
        raise NotImplementedError
        
    def compile_model(self, learning_rate: float = LEARNING_RATE):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate: Learning rate for the optimizer
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
            
        optimizer = Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info(f"Model {self.model_name} compiled successfully")
        
    def get_callbacks(self, model_path: str) -> list:
        """
        Get training callbacks.
        
        Args:
            model_path: Path to save the best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks
        
    def train(self, train_data, validation_data, epochs: int = EPOCHS) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data (generator or tuple)
            validation_data: Validation data (generator or tuple)
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
            
        model_path = MODELS_DIR / f"{self.model_name}_best.h5"
        callbacks = self.get_callbacks(str(model_path))
        
        logger.info(f"Starting training for {self.model_name}")
        
        self.history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed for {self.model_name}")
        return self.history.history
        
    def evaluate(self, test_data) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data (generator or tuple)
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
            
        results = self.model.evaluate(test_data, verbose=1)
        
        metrics = {
            'test_loss': results[0],
            'test_accuracy': results[1],
            'test_top_k_accuracy': results[2]
        }
        
        logger.info(f"Evaluation results for {self.model_name}: {metrics}")
        return metrics
        
    def predict(self, images):
        """
        Make predictions on images.
        
        Args:
            images: Input images
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be loaded before prediction")
            
        return self.model.predict(images)
        
    def save_model(self, filepath: str):
        """
        Save the model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """
        Load a model from file.
        
        Args:
            filepath: Path to the model file
        """
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class VGG16Classifier(SatelliteImageClassifier):
    """
    VGG16-based classifier with transfer learning.
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__('vgg16', num_classes)
        
    def build_model(self, trainable_layers: int = 0) -> Model:
        """
        Build VGG16 model with transfer learning.
        
        Args:
            trainable_layers: Number of top layers to make trainable (0 = freeze all)
            
        Returns:
            Compiled Keras model
        """
        config = MODELS_CONFIG['vgg16']
        
        # Load pre-trained VGG16
        base_model = VGG16(
            weights=config['weights'],
            include_top=config['include_top'],
            input_shape=config['input_shape']
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Make top layers trainable if specified
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
                
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        logger.info(f"VGG16 model built with {len(self.model.layers)} layers")
        logger.info(f"Trainable parameters: {self.model.count_params()}")
        
        return self.model


class ResNet50Classifier(SatelliteImageClassifier):
    """
    ResNet50-based classifier with transfer learning.
    """
    
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__('resnet50', num_classes)
        
    def build_model(self, trainable_layers: int = 0) -> Model:
        """
        Build ResNet50 model with transfer learning.
        
        Args:
            trainable_layers: Number of top layers to make trainable (0 = freeze all)
            
        Returns:
            Compiled Keras model
        """
        config = MODELS_CONFIG['resnet50']
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights=config['weights'],
            include_top=config['include_top'],
            input_shape=config['input_shape']
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Make top layers trainable if specified
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True
                
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        logger.info(f"ResNet50 model built with {len(self.model.layers)} layers")
        logger.info(f"Trainable parameters: {self.model.count_params()}")
        
        return self.model


def create_model(model_name: str, num_classes: int = NUM_CLASSES) -> SatelliteImageClassifier:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model ('vgg16' or 'resnet50')
        num_classes: Number of output classes
        
    Returns:
        Model instance
    """
    if model_name.lower() == 'vgg16':
        return VGG16Classifier(num_classes)
    elif model_name.lower() == 'resnet50':
        return ResNet50Classifier(num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def fine_tune_model(model: SatelliteImageClassifier, train_data, validation_data, 
                   trainable_layers: int = 10, epochs: int = 20, 
                   learning_rate: float = 1e-5) -> Dict[str, Any]:
    """
    Fine-tune a pre-trained model by unfreezing top layers.
    
    Args:
        model: Pre-trained model instance
        train_data: Training data
        validation_data: Validation data
        trainable_layers: Number of top layers to unfreeze
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        
    Returns:
        Fine-tuning history
    """
    logger.info(f"Starting fine-tuning with {trainable_layers} trainable layers")
    
    # Unfreeze top layers
    if model.model_name == 'vgg16':
        base_model = model.model.layers[0]
    else:  # resnet50
        base_model = model.model.layers[0]
        
    base_model.trainable = True
    
    # Freeze all layers except the top ones
    for layer in base_model.layers[:-trainable_layers]:
        layer.trainable = False
        
    # Recompile with lower learning rate
    model.compile_model(learning_rate=learning_rate)
    
    # Fine-tune
    model_path = MODELS_DIR / f"{model.model_name}_finetuned.h5"
    callbacks = model.get_callbacks(str(model_path))
    
    history = model.model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Fine-tuning completed")
    return history.history


if __name__ == "__main__":
    # Example usage
    print("Creating VGG16 model...")
    vgg_model = create_model('vgg16')
    vgg_model.build_model()
    vgg_model.compile_model()
    
    print("Creating ResNet50 model...")
    resnet_model = create_model('resnet50')
    resnet_model.build_model()
    resnet_model.compile_model()
    
    print("Models created successfully!")
    print(f"VGG16 parameters: {vgg_model.model.count_params()}")
    print(f"ResNet50 parameters: {resnet_model.model.count_params()}")
