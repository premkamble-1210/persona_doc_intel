# Run Without Docker - Local Setup Instructions

## Prerequisites
- Python 3.9+ installed
- pip package manager

## Setup Steps

### 1. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Or using conda
conda create -n persona-doc-intel python=3.9
conda activate persona-doc-intel
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python run_pipeline.py
```

### 4. Check Results
Results will be in: `data/output/persona_document_intelligence_results.json`

## Troubleshooting

### If you get import errors:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### If you get CUDA/GPU errors:
The system is designed to run on CPU only, so this shouldn't happen.

### If you get memory errors:
Reduce the number of documents or chunk size in config/settings.py

## Configuration
Edit `config/settings.py` to customize:
- CHUNK_SIZE = 1000  # Reduce if memory issues
- TOP_K_CHUNKS = 20  # Number of top results
- MODEL_NAME = "all-MiniLM-L6-v2"  # Embedding model
