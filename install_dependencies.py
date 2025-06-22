#!/usr/bin/env python3
"""
Installation script for Automated Metadata Generation System
Handles common dependency issues and provides fallback installation methods
"""

import subprocess
import sys
import os
import importlib.util

def run_command(command, description=""):
    """Run a command with error handling"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    if version.major == 3 and version.minor >= 12:
        print("⚠️  Python 3.12+ detected - some packages may need special handling")
    
    return True

def upgrade_pip_setuptools():
    """Upgrade pip, setuptools, and wheel to latest versions"""
    commands = [
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        ("python -m pip install --upgrade setuptools>=65.0.0", "Upgrading setuptools"),
        ("python -m pip install --upgrade wheel", "Upgrading wheel"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    return True

def install_minimal_requirements():
    """Install minimal requirements first"""
    print("\n📦 Installing minimal requirements...")
    return run_command("pip install -r requirements-minimal.txt", "Installing minimal packages")

def install_advanced_packages():
    """Install advanced ML/NLP packages with error handling"""
    print("\n🧠 Installing advanced AI/ML packages...")
    
    # Try to install packages one by one to identify problematic ones
    advanced_packages = [
        ("transformers>=4.30.0", "Transformers library"),
        ("torch>=2.0.0", "PyTorch"),
        ("sentence-transformers>=2.2.0", "Sentence Transformers"),
        ("scikit-learn>=1.3.0", "Scikit-learn"),
        ("seaborn>=0.12.0", "Seaborn"),
        ("wordcloud>=1.9.0", "WordCloud"),
        ("opencv-python>=4.8.0", "OpenCV"),
    ]
    
    failed_packages = []
    
    for package, description in advanced_packages:
        if not run_command(f"pip install '{package}'", f"Installing {description}"):
            failed_packages.append((package, description))
    
    if failed_packages:
        print("\n⚠️  Some advanced packages failed to install:")
        for package, description in failed_packages:
            print(f"   - {description}: {package}")
        print("\nThe system will work with basic functionality. You can try installing these manually later.")
    
    return len(failed_packages) == 0

def install_optional_packages():
    """Install optional packages"""
    print("\n🔧 Installing optional packages...")
    
    optional_packages = [
        ("jupyter>=1.0.0", "Jupyter Notebook"),
        ("ipywidgets>=8.0.0", "Jupyter Widgets"),
        ("flask>=2.3.0", "Flask"),
        ("fastapi>=0.104.0", "FastAPI"),
        ("uvicorn>=0.24.0", "Uvicorn"),
    ]
    
    for package, description in optional_packages:
        run_command(f"pip install '{package}'", f"Installing {description}")

def test_imports():
    """Test if key packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        ("streamlit", "Streamlit"),
        ("PyPDF2", "PyPDF2"),
        ("docx", "Python-docx"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("nltk", "NLTK"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
    ]
    
    failed_imports = []
    
    for package, name in test_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"✅ {name} - OK")
            else:
                print(f"❌ {name} - Not found")
                failed_imports.append(name)
        except ImportError:
            print(f"❌ {name} - Import error")
            failed_imports.append(name)
    
    return len(failed_imports) == 0

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def create_fallback_module():
    """Create a fallback module for missing advanced features"""
    fallback_code = '''"""
Fallback module for missing advanced ML/NLP features
"""

class FallbackClassifier:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, text, labels):
        return {
            'labels': ['unknown'],
            'scores': [1.0]
        }

class FallbackSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    
    def encode(self, texts):
        import numpy as np
        return np.random.rand(len(texts) if isinstance(texts, list) else 1, 384)

# Fallback imports
try:
    from transformers import pipeline
except ImportError:
    def pipeline(*args, **kwargs):
        return FallbackClassifier()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = FallbackSentenceTransformer
'''
    
    with open('fallback_imports.py', 'w') as f:
        f.write(fallback_code)
    
    print("📝 Created fallback module for missing features")

def main():
    """Main installation process"""
    print("🚀 Starting Automated Metadata Generation System Installation\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade core tools
    if not upgrade_pip_setuptools():
        print("⚠️  Failed to upgrade pip/setuptools, continuing anyway...")
    
    # Install minimal requirements
    if not install_minimal_requirements():
        print("❌ Failed to install minimal requirements")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Try creating a new virtual environment:")
        print("   python -m venv metadata_env")
        print("   metadata_env\\Scripts\\activate  # Windows")
        print("   source metadata_env/bin/activate  # Linux/Mac")
        print("2. Update your Python installation")
        print("3. Try installing packages individually")
        sys.exit(1)
    
    # Install advanced packages (optional)
    install_advanced_packages()
    
    # Install optional packages
    install_optional_packages()
    
    # Test imports
    if test_imports():
        print("\n✅ All core packages installed successfully!")
    else:
        print("\n⚠️  Some packages failed to install, creating fallback module...")
        create_fallback_module()
    
    # Download NLTK data
    download_nltk_data()
    
    print("\n🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Run: streamlit run streamlit_app.py")
    print("2. Or open: automated_metadata_generation.ipynb in Jupyter")
    print("\nIf you encounter issues, check the README.md for troubleshooting tips.")

if __name__ == "__main__":
    main() 