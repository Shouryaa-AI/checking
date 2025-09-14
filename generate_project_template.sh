#!/bin/bash

# Script to generate a comprehensive ML project structure
echo "Creating ML project template structure..."

# Create app directory and files
mkdir -p app/pages
touch app/backend.py
echo "# pylint: disable=invalid-name" > app/Home.py

# Create data directories
mkdir -p data/raw
mkdir -p data/processed

# Create notebooks directory
mkdir -p notebooks

# Create src module and submodules
mkdir -p src/data
mkdir -p src/models
mkdir -p src/visualization

# Create Python package markers
touch src/__init__.py
touch src/main.py
touch src/data/__init__.py
touch src/data/processor.py
touch src/data/eda_util.py
touch src/models/__init__.py
touch src/models/trainer.py
touch src/models/inference.py
touch src/visualization/__init__.py

# Create models directory
mkdir -p models

# Create test directory
mkdir -p test

# Create additional files
touch create_environment.sh
touch Dockerfile
touch setup.py
touch README.md

# Create .gitignore
cat > .gitignore << 'EOT'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/*
!models/.gitkeep

# VS Code
.vscode/

# Mac OS
.DS_Store
EOT

# Create dev_requirements.txt with required libraries
cat > dev_requirements.txt << 'EOT'
black
jupyter-black
pylint
scikit-learn
pandas
numpy
matplotlib
seaborn
EOT

# Add .gitkeep files to empty directories that might be ignored
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep

echo "ML project template created successfully."
