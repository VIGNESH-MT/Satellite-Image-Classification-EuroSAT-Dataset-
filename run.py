#!/usr/bin/env python3
"""
Simple run script for the EuroSAT Satellite Image Classification web application.
"""

import os
import sys
import argparse
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met."""
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Check if src directory exists
    if not Path("src").exists():
        print("❌ src directory not found")
        return False
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found")
        return False
    
    return True

def setup_environment():
    """Set up the environment."""
    # Create necessary directories
    directories = ['data', 'models', 'logs', 'static/uploads']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Add src to Python path
    src_path = str(Path("src").absolute())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Run EuroSAT Classifier Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--setup", action="store_true", help="Run setup first")
    
    args = parser.parse_args()
    
    print("🛰️ EuroSAT Satellite Image Classification")
    print("=" * 45)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed")
        print("Please run: python setup.py")
        sys.exit(1)
    
    # Run setup if requested
    if args.setup:
        print("🔄 Running setup...")
        try:
            import setup
            setup.setup_project()
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            sys.exit(1)
    
    # Set up environment
    setup_environment()
    
    # Import and run the Flask app
    try:
        print("🔄 Starting Flask application...")
        
        # Set environment variables
        os.environ['FLASK_HOST'] = args.host
        os.environ['FLASK_PORT'] = str(args.port)
        if args.debug:
            os.environ['FLASK_DEBUG'] = 'True'
        
        # Import and run the app
        from app import create_app
        app = create_app()
        
        print(f"🚀 Starting server on http://{args.host}:{args.port}")
        print("📱 The application is mobile-friendly!")
        print("🗺️ Google Earth Engine integration available (if configured)")
        print("\n💡 Tips:")
        print("   - Upload satellite images in JPG, PNG, or TIFF format")
        print("   - Try both VGG16 and ResNet50 models")
        print("   - Use the interactive map for location analysis")
        print("\n🛑 Press Ctrl+C to stop the server")
        print("-" * 45)
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
