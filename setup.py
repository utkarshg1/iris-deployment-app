from setuptools import setup
import os
import subprocess

# Define dependencies
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "streamlit"
]

def create_project_structure():
    """Create necessary files and directories for the ML/Streamlit project."""
    os.makedirs("notebook", exist_ok=True)

    # Create a blank Jupyter notebook
    model_notebook = os.path.join("notebook", "model.ipynb")
    if not os.path.exists(model_notebook):
        with open(model_notebook, "w") as f:
            f.write("")

    # Create utility and app files
    for filename in ["app.py", "utils.py"]:
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write("# Write your code here\n")

    print("✅ Project structure created successfully!")

def upgrade_pip():
    """Upgrade pip to the latest version."""
    print("⬆️ Upgrading pip...")
    subprocess.run(["python", "-m", "pip", "install", "--upgrade", "pip"], check=True)

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    subprocess.run(["pip", "install"] + dependencies, check=True)

def create_requirements_file():
    """Create a requirements.txt file with dependencies."""
    with open("requirements.txt", "w") as f:
        for dep in dependencies:
            f.write(dep + "\n")
    print("✅ requirements.txt created successfully!")

# Run setup tasks
create_project_structure()
upgrade_pip()
install_dependencies()
create_requirements_file()

# Define setup
setup(
    name="MLStreamlitProject",
    version="0.1",
    description="A setup script for initializing an ML project with Streamlit and Jupyter",
    install_requires=dependencies,
)
