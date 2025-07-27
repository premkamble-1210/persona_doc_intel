#!/bin/bash

# Persona Document Intelligence - Docker Runner Script

set -e

echo "==================================================="
echo "Persona Document Intelligence - Docker Runner"
echo "==================================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker and try again."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to build and run
run_pipeline() {
    echo "🔨 Building Docker image..."
    docker-compose build
    
    echo "📂 Checking input documents..."
    if [ ! -d "data/input_documents" ] || [ -z "$(ls -A data/input_documents 2>/dev/null)" ]; then
        echo "⚠️  Warning: No documents found in data/input_documents/"
        echo "   Please add PDF documents to data/input_documents/ directory"
        exit 1
    fi
    
    doc_count=$(ls -1 data/input_documents/*.pdf 2>/dev/null | wc -l)
    echo "📄 Found $doc_count PDF documents"
    
    echo "🚀 Running persona document intelligence pipeline..."
    docker-compose up
    
    echo "✅ Pipeline completed!"
    echo "📊 Results saved to: data/output/persona_document_intelligence_results.json"
}

# Function to run in development mode
run_dev() {
    echo "🔨 Building Docker image for development..."
    docker-compose --profile dev build
    
    echo "🚀 Running in development mode..."
    docker-compose --profile dev up
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up Docker containers and images..."
    docker-compose down --remove-orphans
    docker image prune -f
    echo "✅ Cleanup completed"
}

# Function to show logs
show_logs() {
    echo "📋 Showing container logs..."
    docker-compose logs -f
}

# Main script logic
case "${1:-run}" in
    "run")
        check_docker
        run_pipeline
        ;;
    "dev")
        check_docker
        run_dev
        ;;
    "logs")
        show_logs
        ;;
    "clean")
        cleanup
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  run     Run the persona document intelligence pipeline (default)"
        echo "  dev     Run in development mode with live code updates"
        echo "  logs    Show container logs"
        echo "  clean   Clean up Docker containers and images"
        echo "  help    Show this help message"
        echo ""
        echo "Example:"
        echo "  $0 run     # Run the pipeline"
        echo "  $0 dev     # Development mode"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
