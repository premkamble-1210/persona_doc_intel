# Docker Execution Instructions

## Prerequisites
- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)

## Quick Start

### Method 1: Using Docker Compose (Recommended)

1. **Build and run the container:**
```bash
docker-compose up --build
```

2. **Run in detached mode:**
```bash
docker-compose up -d --build
```

3. **View logs:**
```bash
docker-compose logs -f
```

4. **Stop the container:**
```bash
docker-compose down
```

### Method 2: Using Docker directly

1. **Build the image:**
```bash
docker build -t persona-doc-intel .
```

2. **Run the container:**
```bash
docker run -v $(pwd)/data/input_documents:/app/data/input_documents:ro \
           -v $(pwd)/data/output:/app/data/output \
           -v $(pwd)/cache:/app/cache \
           persona-doc-intel
```

### Method 3: Interactive mode for development

1. **Run with bash shell:**
```bash
docker run -it -v $(pwd):/app persona-doc-intel bash
```

2. **Then inside the container:**
```bash
python run_pipeline.py
```

## Input/Output

### Input Documents
Place your PDF documents in the `data/input_documents/` directory before running.

### Output
Results will be saved in `data/output/persona_document_intelligence_results.json`

### Caching
Embeddings are cached in `cache/embeddings/` for faster subsequent runs.

## Configuration

### Custom Configuration
To use custom settings:
```bash
docker run -v $(pwd)/config/custom_settings.py:/app/config/settings.py \
           -v $(pwd)/data:/app/data \
           persona-doc-intel
```

### Environment Variables
You can override default settings using environment variables:
```bash
docker run -e CHUNK_SIZE=1500 \
           -e TOP_K_CHUNKS=25 \
           -v $(pwd)/data:/app/data \
           persona-doc-intel
```

## Development Mode

For development with live code changes:
```bash
docker-compose --profile dev up --build
```

## Troubleshooting

### Common Issues

1. **Permission errors on Linux/Mac:**
```bash
sudo chown -R $USER:$USER data/ cache/
```

2. **Out of memory errors:**
```bash
docker run --memory=4g -v $(pwd)/data:/app/data persona-doc-intel
```

3. **View container logs:**
```bash
docker logs persona-document-intelligence
```

### Resource Requirements
- **Memory:** Minimum 2GB RAM, recommended 4GB
- **Storage:** ~1GB for image + space for documents and cache
- **CPU:** Any modern CPU (optimized for CPU-only inference)

## Challenge Submission

For Round 1B submission, ensure:
1. Input PDFs are in `data/input_documents/`
2. Run: `docker-compose up --build`
3. Output will be in `data/output/persona_document_intelligence_results.json`
4. Processing time should be under 60 seconds for 3-5 documents
