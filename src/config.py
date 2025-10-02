"""
Configuration file for EuroSAT satellite image classification project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
STATIC_DIR = PROJECT_ROOT / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, UPLOADS_DIR]:
    dir_path.mkdir(exist_ok=True)

# EuroSAT dataset configuration
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

NUM_CLASSES = len(EUROSAT_CLASSES)

# Image preprocessing parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
BATCH_SIZE = 32

# Training parameters
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model parameters
MODELS_CONFIG = {
    'vgg16': {
        'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        'weights': 'imagenet',
        'include_top': False,
        'pooling': 'avg'
    },
    'resnet50': {
        'input_shape': (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        'weights': 'imagenet',
        'include_top': False,
        'pooling': 'avg'
    }
}

# Google Earth Engine configuration
GEE_SERVICE_ACCOUNT_KEY = os.getenv('GEE_SERVICE_ACCOUNT_KEY')
GEE_PROJECT_ID = os.getenv('GEE_PROJECT_ID', 'your-project-id')

# Flask configuration
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = True
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
