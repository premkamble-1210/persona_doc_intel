# Challenge Submission Instructions

## System Requirements
- Python 3.9+
- 4GB RAM minimum
- 2GB free disk space

## Quick Start (Without Docker)

### 1. Setup Environment
```bash
# Windows PowerShell
cd persona_doc_intel
python -m pip install -r requirements.txt
```

### 2. Add Test Documents
Place 3-10 PDF documents in: `data/input_documents/`

### 3. Run Pipeline
```bash
python run_pipeline.py
```

### 4. Get Results
Output will be in: `data/output/persona_document_intelligence_results.json`

## Expected Output Format
The system generates output in the required challenge format:
- ✅ Metadata with persona, job-to-be-done, processing timestamp
- ✅ Extracted sections with importance rankings
- ✅ Sub-sections with refined text
- ✅ Processing under 60 seconds for 3-5 documents

## Performance
- Processes 6 documents in ~10 seconds
- CPU-only operation (no GPU required)
- Caches embeddings for faster subsequent runs

## Docker Alternative (If Available)
If Docker is installed:
```bash
docker-compose up --build
```

## Architecture
- Generic document processing pipeline
- Persona-driven content ranking
- Multi-factor relevance scoring
- Adaptive text chunking
- Quality-based filtering
