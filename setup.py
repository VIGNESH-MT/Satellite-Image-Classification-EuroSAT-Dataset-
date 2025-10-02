"""
Setup script for EuroSAT Satellite Image Classification project.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def setup_project():
    """Set up the EuroSAT classification project."""
    print("🛰️ EuroSAT Satellite Image Classification Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create directories
    print("\n📁 Creating project directories...")
    directories = ['data', 'models', 'logs', 'static/uploads', 'static/css', 'static/js']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠️  Some packages may have failed to install. Please check manually.")
    
    # Create environment file
    env_file = Path(".env")
    if not env_file.exists():
        print("\n⚙️ Creating environment configuration...")
        env_content = """# EuroSAT Configuration
# Copy from env_example.txt and customize as needed

# Google Earth Engine (optional)
# GEE_SERVICE_ACCOUNT_KEY=path/to/your/service-account-key.json
# GEE_PROJECT_ID=your-gee-project-id

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-change-this
FLASK_DEBUG=True

# Logging
LOG_LEVEL=INFO
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("   Created: .env file")
    
    # Download sample data
    print("\n🖼️ Setting up sample data...")
    try:
        from src.utils import setup_directories, download_sample_images
        setup_directories()
        sample_paths = download_sample_images()
        print(f"   Downloaded {len(sample_paths)} sample images")
    except Exception as e:
        print(f"   ⚠️ Could not download sample images: {e}")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        "tensorflow",
        "flask",
        "opencv-python",
        "numpy",
        "matplotlib",
        "sklearn"
    ]
    
    failed_imports = []
    for package in test_imports:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            failed_imports.append(package)
    
    # Final status
    print("\n" + "=" * 50)
    if failed_imports:
        print("⚠️  Setup completed with warnings")
        print("   Failed imports:", ", ".join(failed_imports))
        print("   Please install missing packages manually")
    else:
        print("🎉 Setup completed successfully!")
    
    print("\n🚀 Next steps:")
    print("   1. (Optional) Set up Google Earth Engine authentication:")
    print("      earthengine authenticate")
    print("   2. Download the EuroSAT dataset to data/eurosat/")
    print("   3. Run the application:")
    print("      python app.py")
    print("   4. Open http://localhost:5000 in your browser")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    setup_project()
