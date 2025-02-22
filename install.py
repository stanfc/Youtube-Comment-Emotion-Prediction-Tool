import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
packages = [
    "google-api-python-client",
    "jieba",
    "torch",
    "transformers",
    "datasets",
    "matplotlib",
    "wordcloud",
    "pillow",
    "numpy",
    "tqdm"
]

# Install each package
for package in packages:
    try:
        install(package)
        print(f"Successfully installed {package}")
    except Exception as e:
        print(f"Failed to install {package}: {e}")