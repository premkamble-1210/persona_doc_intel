# Persona Document Intelligence Pipeline

A sophisticated document intelligence system that extracts and ranks relevant content from PDF documents based on user personas and job-to-be-done requirements. The system uses advanced NLP techniques including transformer-based embeddings and semantic similarity to deliver highly relevant document insights.

## 🎯 Key Features

- **Persona-Based Analysis**: Customizable user personas with expertise areas and priorities
- **Job-to-Be-Done Alignment**: Ranks content based on specific objectives and success criteria
- **Multi-Modal PDF Processing**: Robust PDF text extraction with multiple fallback methods
- **Intelligent Text Chunking**: Section-aware chunking that preserves document structure
- **Advanced Ranking**: Multi-factor scoring including semantic similarity, content quality, and context
- **Scalable Architecture**: Docker support and configurable processing pipeline
- **Comprehensive Output**: Structured JSON results with detailed scoring metadata

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Files     │───▶│  Text Extraction │───▶│  Section & Chunk│
│                 │    │  (PyMuPDF,       │    │  Generation     │
│                 │    │   pdfplumber)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Persona Config  │───▶│  Embedding       │◀───│                 │
│ (JSON/YAML)     │    │  Generation      │    │                 │
└─────────────────┘    │ (Transformers)   │    │                 │
                       └──────────────────┘    │                 │
                                 │             │                 │
┌─────────────────┐    ┌──────────────────┐    │                 │
│ Ranked Results  │◀───│  Relevance       │◀───┘                 │
│ (JSON Output)   │    │  Scoring &       │                      │
│                 │    │  Ranking         │                      │
└─────────────────┘    └──────────────────┘                      │
```

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Build the container:**
```bash
docker build -t persona_doc_intel .
```

2. **Prepare your data:**
```bash
mkdir -p data/input_documents data/output
# Copy your PDF files to data/input_documents/
```

3. **Run the pipeline:**
```bash
docker run -v $(pwd)/data:/app/data persona_doc_intel \
  python run_pipeline.py --input-dir data/input_documents --output-dir data/output
```

### Local Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

2. **Run the pipeline:**
```bash
python run_pipeline.py --input-dir data/input_documents --output-dir data/output
```

## 📋 Configuration

### Persona Configuration

Create a persona configuration file (JSON or YAML) to define your user persona and job requirements:

```json
{
  "persona": {
    "role": "Data Scientist",
    "description": "An experienced data scientist focused on machine learning and analytics",
    "expertise_areas": [
      "Machine learning",
      "Statistical analysis", 
      "Data visualization",
      "Python programming"
    ],
    "priorities": [
      "Technical accuracy",
      "Practical implementation",
      "Performance optimization"
    ],
    "experience_level": "experienced",
    "industry": "technology"
  },
  "job_to_be_done": {
    "title": "Extract ML Model Implementation Details",
    "description": "Find specific information about machine learning model architectures, training procedures, and performance metrics",
    "key_objectives": [
      "Identify model architectures used",
      "Extract training methodologies", 
      "Find performance benchmarks",
      "Locate implementation details"
    ],
    "success_criteria": [
      "Technical depth and accuracy",
      "Actionable implementation guidance",
      "Quantitative results and metrics"
    ],
    "urgency": "high"
  }
}
```

### Environment Configuration

Set environment variables or create a `.env` file:

```bash
# Model configuration
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
MAX_MODEL_SIZE=384
USE_GPU=false

# Processing configuration  
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_CHUNKS=20

# Ranking weights
PERSONA_WEIGHT=0.6
JOB_WEIGHT=0.4

# Output configuration
INCLUDE_EMBEDDINGS=false
INCLUDE_METADATA=true
```

## 🔧 Advanced Usage

### Custom Persona Creation

```python
from utils.persona_parser import PersonaParser

parser = PersonaParser()

# Create a template
parser.create_persona_template("my_persona.json", format_type="json")

# Validate existing persona
validation_result = parser.validate_file("my_persona.json")
print(validation_result)
```

### Programmatic Usage

```python
from run_pipeline import DocumentIntelligencePipeline
from config.settings import Settings

# Initialize with custom settings
config = Settings()
config.CHUNK_SIZE = 1024
config.TOP_K_CHUNKS = 30

# Run pipeline
pipeline = DocumentIntelligencePipeline(config)
results = pipeline.run(
    input_dir="data/input_documents",
    output_dir="data/output", 
    persona_file="my_persona.json"
)

print(f"Processed {results['processing_summary']['total_documents']} documents")
```

### Batch Processing

```python
from utils.pdf_loader import PDFBatchProcessor

processor = PDFBatchProcessor()

# Process all PDFs in directory
results = processor.process_directory(
    "data/input_documents",
    output_callback=lambda name, i, total, success: print(f"Processed {name} ({i}/{total})")
)

# Get processing statistics
stats = processor.get_processing_stats(results)
print(f"Total characters processed: {stats['total_chars']}")
```

## 📊 Output Format

The pipeline generates a comprehensive JSON output:

```json
{
  "persona": { ... },
  "job_to_be_done": { ... },
  "processing_summary": {
    "total_documents": 5,
    "total_sections": 47,
    "total_chunks": 156,
    "final_top_chunks": 20
  },
  "ranked_content": [
    {
      "content": "Machine learning model architectures...",
      "section_title": "Model Architecture Design",
      "relevance_score": 0.892,
      "persona_score": 0.856,
      "job_score": 0.923,
      "content_quality_score": 0.734,
      "rank_position": 1,
      "source_document": "ml_research_paper.pdf",
      "ranking_metadata": {
        "semantic_score": 0.887,
        "context_bonus": 0.2,
        "has_keywords": true
      }
    }
  ],
  "document_details": [ ... ]
}
```

## 🎛️ Configuration Profiles

Use different configuration profiles for various scenarios:

```python
from config.settings import get_settings

# Development profile - fast processing
config = get_settings('development')

# Production profile - high quality
config = get_settings('production') 

# Lightweight profile - resource constrained
config = get_settings('lightweight')
```

## 🧪 Testing and Evaluation

### Model Benchmarking

```python
from utils.embedding_model import EmbeddingBenchmark

benchmark = EmbeddingBenchmark()

# Test different models
test_texts = ["Sample text for testing...", ...]
models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2'
]

results = benchmark.compare_models(models, test_texts)
print(f"Best model: {results['best_model']}")
```

### Ranking Evaluation

```python
from utils.ranker import RankingEvaluator

evaluator = RankingEvaluator()
evaluation = evaluator.evaluate_ranking_quality(ranked_chunks)

print(f"High quality chunks: {evaluation['score_distribution']['high_scores']}")
print(f"Content coverage: {evaluation['content_coverage']['unique_sections']} sections")
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues with Large Documents**
   -