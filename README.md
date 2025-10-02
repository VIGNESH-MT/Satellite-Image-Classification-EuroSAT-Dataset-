<<<<<<< HEAD (GitHub remote content)
>>>>>>> origin/main
# 🛰️ EuroSAT Satellite Image Classification

A comprehensive machine learning project for satellite image classification using the EuroSAT dataset. This project combines state-of-the-art deep learning models (VGG16 and ResNet50) with Google Earth Engine integration to provide accurate land use classification and interactive geospatial analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **🧠 Deep Learning Models**: VGG16 and ResNet50 with transfer learning
- **🖼️ Image Preprocessing**: Advanced preprocessing pipeline using OpenCV
- **📊 Comprehensive Evaluation**: Detailed accuracy metrics and visualizations
- **🌍 Google Earth Engine Integration**: Real-time satellite data analysis
- **🗺️ Interactive Dashboard**: Flask web application with responsive design
- **📱 Mobile Friendly**: Works seamlessly on all devices
- **🔄 Real-time Analysis**: Live land cover validation and mapping

## 🎯 Classification Categories

The model can classify satellite images into 10 different land use categories:

| Category | Description |
|----------|-------------|
| 🌾 AnnualCrop | Annual cropland areas |
| 🌲 Forest | Forest and wooded areas |
| 🌿 HerbaceousVegetation | Herbaceous vegetation |
| 🛣️ Highway | Highway and road infrastructure |
| 🏭 Industrial | Industrial areas and facilities |
| 🐄 Pasture | Pasture and grazing land |
| 🌳 PermanentCrop | Permanent crop areas |
| 🏘️ Residential | Residential and urban areas |
| 🏞️ River | Rivers and waterways |
| 🏊 SeaLake | Seas, lakes, and water bodies |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Google Earth Engine account for full functionality

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/eurosat-classifier.git
   cd eurosat-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp env_example.txt .env
   # Edit .env with your configuration
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:5000`

## 📁 Project Structure

```
eurosat-classifier/
├── src/                          # Source code
│   ├── config.py                 # Configuration settings
│   ├── data_preprocessing.py     # Image preprocessing pipeline
│   ├── models.py                 # CNN model implementations
│   ├── train_evaluate.py         # Training and evaluation scripts
│   ├── gee_integration.py        # Google Earth Engine integration
│   └── utils.py                  # Utility functions
├── templates/                    # HTML templates
│   ├── base.html                 # Base template
│   ├── index.html                # Home page
│   ├── results.html              # Results page
│   ├── about.html                # About page
│   ├── 404.html                  # Error pages
│   └── 500.html
├── static/                       # Static files
│   ├── css/                      # Custom stylesheets
│   ├── js/                       # JavaScript files
│   └── uploads/                  # Uploaded images
├── data/                         # Dataset directory
├── models/                       # Trained models
├── logs/                         # Training logs and plots
├── app.py                        # Flask application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env_example.txt`:

```env
# Google Earth Engine Configuration
GEE_SERVICE_ACCOUNT_KEY=path/to/your/service-account-key.json
GEE_PROJECT_ID=your-gee-project-id

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True

# Model Configuration
MODEL_PATH=models/
DATA_PATH=data/eurosat/

# Logging
LOG_LEVEL=INFO
```

### Google Earth Engine Setup

1. **Create a Google Earth Engine account**
   - Visit [Google Earth Engine](https://earthengine.google.com/)
   - Sign up for access

2. **Authenticate Earth Engine**
   ```bash
   earthengine authenticate
   ```

3. **For service account authentication** (optional)
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Set the `GEE_SERVICE_ACCOUNT_KEY` environment variable

## 📊 Dataset

### EuroSAT Dataset

- **Total Images**: 27,000 satellite images
- **Image Size**: 64x64 pixels (upscaled to 224x224 for training)
- **Spectral Bands**: RGB
- **Source**: Sentinel-2 satellite imagery
- **Coverage**: 34 European countries
- **Classes**: 10 land use/land cover classes

### Download Dataset

1. **Official Source**
   ```bash
   # Download from the official repository
   wget https://madm.dfki.de/files/sentinel/EuroSAT.zip
   unzip EuroSAT.zip -d data/
   ```

2. **Using TensorFlow Datasets**
   ```python
   import tensorflow_datasets as tfds
   dataset = tfds.load('eurosat/rgb', split='train', as_supervised=True)
   ```

## 🏋️ Training Models

### Basic Training

```bash
# Train both VGG16 and ResNet50 models
python src/train_evaluate.py
```

### Custom Training

```python
from src.models import create_model
from src.data_preprocessing import EuroSATDataProcessor

# Initialize data processor
processor = EuroSATDataProcessor('data/eurosat')

# Load and preprocess data
images, labels, paths = processor.load_dataset()
X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(images, labels)

# Create and train model
model = create_model('vgg16')
model.build_model()
model.compile_model()
history = model.train((X_train, y_train), (X_val, y_val), epochs=50)

# Evaluate model
results = model.evaluate((X_test, y_test))
```

## 🌐 Web Application

### Features

- **📤 File Upload**: Drag-and-drop interface for image upload
- **🤖 Model Selection**: Choose between VGG16 and ResNet50
- **📈 Results Visualization**: Interactive charts and confidence scores
- **🗺️ Interactive Maps**: Google Earth Engine integration
- **📱 Responsive Design**: Mobile-friendly interface

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/upload` | POST | Upload and classify image |
| `/results/<id>` | GET | View classification results |
| `/api/model_info` | GET | Get model information |
| `/api/analyze_coordinates` | POST | Analyze coordinates with GEE |

### API Usage Example

```bash
# Analyze coordinates
curl -X POST http://localhost:5000/api/analyze_coordinates \
  -H "Content-Type: application/json" \
  -d '{"lat": 52.5200, "lon": 13.4050, "predicted_class": "Residential"}'
```

## 📈 Model Performance

### VGG16 Model
- **Parameters**: ~15M (trainable: ~2M)
- **Training Time**: ~30 minutes (GPU)
- **Test Accuracy**: ~92%
- **Best for**: General use, faster inference

### ResNet50 Model
- **Parameters**: ~25M (trainable: ~4M)
- **Training Time**: ~45 minutes (GPU)
- **Test Accuracy**: ~94%
- **Best for**: Higher accuracy, complex patterns

## 🛠️ Development

### Running Tests

```bash
# Test data preprocessing
python src/data_preprocessing.py

# Test model creation
python src/models.py

# Test utilities
python src/utils.py
```

### Adding New Models

1. Create a new model class in `src/models.py`
2. Inherit from `SatelliteImageClassifier`
3. Implement the `build_model` method
4. Update the `create_model` factory function

```python
class CustomClassifier(SatelliteImageClassifier):
    def build_model(self):
        # Implement your model architecture
        pass
```

### Custom Preprocessing

```python
from src.data_preprocessing import EuroSATDataProcessor

class CustomProcessor(EuroSATDataProcessor):
    def preprocess_image(self, image_path):
        # Implement custom preprocessing
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Google Earth Engine Authentication**
   ```bash
   earthengine authenticate
   ```

2. **Memory Issues During Training**
   - Reduce batch size in `config.py`
   - Use data generators instead of loading all data

3. **Model Loading Errors**
   - Ensure model files exist in the `models/` directory
   - Check TensorFlow version compatibility

4. **Upload Issues**
   - Check file size limits (16MB max)
   - Verify supported file formats (JPG, PNG, TIFF)

### Debug Mode

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python app.py
```

## 📚 Documentation

### Code Documentation

All modules are thoroughly documented with docstrings. Generate HTML documentation:

```bash
pip install pdoc3
pdoc --html --output-dir docs src/
```

### Jupyter Notebooks

Example notebooks are available in the `notebooks/` directory:
- `01_data_exploration.ipynb`: Dataset exploration
- `02_model_training.ipynb`: Model training examples
- `03_gee_integration.ipynb`: Google Earth Engine examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **EuroSAT Dataset**: [Helber et al., 2019](https://github.com/phelber/EuroSAT)
- **Google Earth Engine**: For satellite data access
- **TensorFlow Team**: For the deep learning framework
- **Flask Community**: For the web framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/eurosat-classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/eurosat-classifier/discussions)
- **Email**: your.email@example.com

## 🔮 Future Enhancements

- [ ] Multi-temporal analysis
- [ ] Additional satellite data sources
- [ ] Real-time monitoring dashboard
- [ ] Mobile application
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Advanced data augmentation
- [ ] Model ensemble methods

---

**Made with ❤️ for the Earth observation community**
=======
# Satellite-Image-Classification-EuroSAT-Dataset-
>>>>>>> d460abd59d5d2d3eadf195e8dcea2af38e7e6f75
