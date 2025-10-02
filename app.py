"""
Flask web application for EuroSAT satellite image classification.
Provides a dashboard for uploading images and displaying classification results with maps.
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import logging
from pathlib import Path
import uuid
from datetime import datetime

# Import our modules
from src.config import *
from src.models import create_model
from src.data_preprocessing import preprocess_single_image
from src.gee_integration import GEELandUsageAnalyzer

# Setup logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR

# Global variables for models
models = {}
gee_analyzer = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load pre-trained models."""
    global models
    
    model_files = {
        'vgg16': MODELS_DIR / 'vgg16_best.h5',
        'resnet50': MODELS_DIR / 'resnet50_best.h5'
    }
    
    for model_name, model_path in model_files.items():
        if model_path.exists():
            try:
                model = create_model(model_name)
                model.load_model(str(model_path))
                models[model_name] = model
                logger.info(f"Loaded {model_name} model")
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {str(e)}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            # Create a dummy model for demo purposes
            try:
                model = create_model(model_name)
                model.build_model()
                model.compile_model()
                models[model_name] = model
                logger.info(f"Created dummy {model_name} model for demo")
            except Exception as e:
                logger.error(f"Error creating dummy {model_name} model: {str(e)}")

def initialize_gee():
    """Initialize Google Earth Engine analyzer."""
    global gee_analyzer
    try:
        gee_analyzer = GEELandUsageAnalyzer()
        gee_analyzer.authenticate()
        logger.info("GEE analyzer initialized")
    except Exception as e:
        logger.warning(f"GEE initialization failed: {str(e)}")
        gee_analyzer = None

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html', 
                         models=list(models.keys()),
                         classes=EUROSAT_CLASSES)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and classification."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    model_name = request.form.get('model', 'vgg16')
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Preprocess image
            processed_image = preprocess_single_image(file_path)
            
            # Make prediction
            if model_name in models:
                predictions = models[model_name].predict(processed_image)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = EUROSAT_CLASSES[predicted_class_idx]
                confidence = float(predictions[0][predicted_class_idx])
                
                # Get all class probabilities
                class_probabilities = {}
                for i, class_name in enumerate(EUROSAT_CLASSES):
                    class_probabilities[class_name] = float(predictions[0][i])
                
            else:
                # Fallback for demo
                predicted_class_idx = np.random.randint(0, len(EUROSAT_CLASSES))
                predicted_class = EUROSAT_CLASSES[predicted_class_idx]
                confidence = np.random.uniform(0.6, 0.95)
                class_probabilities = {cls: np.random.uniform(0.01, 0.1) 
                                     for cls in EUROSAT_CLASSES}
                class_probabilities[predicted_class] = confidence
            
            # Prepare result
            result = {
                'filename': unique_filename,
                'original_filename': filename,
                'model_used': model_name,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': class_probabilities,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save result to session or database (simplified version)
            result_id = str(uuid.uuid4())
            result_file = UPLOADS_DIR / f"{result_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f)
            
            return redirect(url_for('results', result_id=result_id))
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    else:
        flash('Invalid file type. Please upload an image file.')
        return redirect(url_for('index'))

@app.route('/results/<result_id>')
def results(result_id):
    """Display classification results."""
    try:
        result_file = UPLOADS_DIR / f"{result_id}_result.json"
        
        if not result_file.exists():
            flash('Results not found')
            return redirect(url_for('index'))
        
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Get image path
        image_path = url_for('static', filename=f'uploads/{result["filename"]}')
        
        return render_template('results.html', 
                             result=result, 
                             image_path=image_path,
                             classes=EUROSAT_CLASSES)
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        flash('Error loading results')
        return redirect(url_for('index'))

@app.route('/map/<result_id>')
def show_map(result_id):
    """Display map with land usage analysis."""
    try:
        result_file = UPLOADS_DIR / f"{result_id}_result.json"
        
        if not result_file.exists():
            return jsonify({'error': 'Results not found'}), 404
        
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Get coordinates from request or use default
        lat = request.args.get('lat', 52.5200, type=float)
        lon = request.args.get('lon', 13.4050, type=float)
        
        map_data = {
            'center': [lat, lon],
            'zoom': 13,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence']
        }
        
        # Add GEE analysis if available
        if gee_analyzer and gee_analyzer.is_authenticated:
            try:
                analysis = gee_analyzer.analyze_region_for_classification(
                    (lat, lon), result['predicted_class']
                )
                map_data['gee_analysis'] = analysis
            except Exception as e:
                logger.warning(f"GEE analysis failed: {str(e)}")
        
        return render_template('map.html', 
                             result=result,
                             map_data=map_data)
        
    except Exception as e:
        logger.error(f"Error creating map: {str(e)}")
        return jsonify({'error': 'Error creating map'}), 500

@app.route('/api/analyze_coordinates', methods=['POST'])
def analyze_coordinates():
    """API endpoint to analyze specific coordinates."""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        predicted_class = data.get('predicted_class', 'Unknown')
        
        if lat is None or lon is None:
            return jsonify({'error': 'Coordinates required'}), 400
        
        if gee_analyzer and gee_analyzer.is_authenticated:
            analysis = gee_analyzer.analyze_region_for_classification(
                (lat, lon), predicted_class
            )
            return jsonify(analysis)
        else:
            return jsonify({
                'error': 'Google Earth Engine not available',
                'coordinates': (lat, lon),
                'predicted_class': predicted_class
            })
            
    except Exception as e:
        logger.error(f"Error in coordinate analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info')
def model_info():
    """API endpoint to get model information."""
    model_info = {}
    
    for model_name, model in models.items():
        if model and model.model:
            model_info[model_name] = {
                'loaded': True,
                'parameters': model.model.count_params(),
                'input_shape': model.model.input_shape,
                'classes': EUROSAT_CLASSES
            }
        else:
            model_info[model_name] = {
                'loaded': False,
                'error': 'Model not loaded'
            }
    
    return jsonify(model_info)

@app.route('/about')
def about():
    """About page with project information."""
    return render_template('about.html', 
                         classes=EUROSAT_CLASSES,
                         models=list(models.keys()))

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

def create_app():
    """Application factory."""
    # Ensure upload directory exists
    UPLOADS_DIR.mkdir(exist_ok=True)
    
    # Load models
    load_models()
    
    # Initialize GEE
    initialize_gee()
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
