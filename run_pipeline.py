#!/usr/bin/env python3
"""
Main pipeline entry point for persona-based document intelligence.
This script orchestrates the entire process from PDF loading to ranked output.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse

from utils.pdf_loader import PDFLoader
from utils.persona_parser import PersonaParser
from utils.text_chunker import TextChunker
from utils.embedding_model import EmbeddingModel
from utils.ranker import DocumentRanker
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentIntelligencePipeline:
    """Main pipeline class that orchestrates the document processing workflow."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.pdf_loader = PDFLoader()
        self.persona_parser = PersonaParser()
        self.text_chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        self.embedding_model = EmbeddingModel(
            model_name=config.MODEL_NAME,
            max_model_size=config.MAX_MODEL_SIZE
        )
        self.ranker = DocumentRanker()
        
    def run(self, input_dir: str, output_dir: str, persona_file: str = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output files
            persona_file: Path to persona configuration file
            
        Returns:
            Dictionary containing processing results
        """
        try:
            logger.info("Starting document intelligence pipeline...")
            
            # Step 1: Load and parse persona configuration
            persona_config = self._load_persona_config(persona_file)
            logger.info(f"Loaded persona: {persona_config['persona']['role']}")
            
            # Step 2: Load PDF documents
            documents = self._load_documents(input_dir)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Step 3: Process each document
            processed_docs = []
            for doc_path, doc_content in documents.items():
                logger.info(f"Processing document: {doc_path}")
                processed_doc = self._process_document(doc_path, doc_content, persona_config)
                processed_docs.append(processed_doc)
            
            # Step 4: Generate final output
            final_output = self._generate_final_output(processed_docs, persona_config)
            
            # Step 5: Save results
            output_path = self._save_results(final_output, output_dir)
            
            logger.info(f"Pipeline completed successfully. Output saved to: {output_path}")
            return final_output
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _load_persona_config(self, persona_file: str = None) -> Dict[str, Any]:
        """Load persona and job-to-be-done configuration."""
        if persona_file and os.path.exists(persona_file):
            return self.persona_parser.load_from_file(persona_file)
        else:
            # Use default persona if no file provided
            return self.persona_parser.get_default_persona()
    
    def _load_documents(self, input_dir: str) -> Dict[str, str]:
        """Load all PDF documents from input directory."""
        documents = {}
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {input_dir}")
        
        for pdf_file in pdf_files:
            try:
                content = self.pdf_loader.extract_text(str(pdf_file))
                documents[str(pdf_file)] = content
                logger.info(f"Successfully loaded: {pdf_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {pdf_file.name}: {str(e)}")
        
        return documents
    
    def _process_document(self, doc_path: str, content: str, persona_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single document through the pipeline."""
        
        # Extract sections and subsections
        sections = self.text_chunker.extract_sections(content)
        logger.info(f"Extracted {len(sections)} sections from {Path(doc_path).name}")
        
        # Generate chunks for each section
        all_chunks = []
        for section in sections:
            chunks = self.text_chunker.chunk_text(section['content'])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'section_title': section['title'],
                    'section_level': section['level'],
                    'chunk_id': f"{section['title']}_chunk_{i}",
                    'content': chunk.text,
                    'start_char': chunk.start_pos,
                    'end_char': chunk.end_pos
                })
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        chunk_embeddings = []
        for chunk in all_chunks:
            embedding = self.embedding_model.encode(chunk['content'])
            chunk_embeddings.append({
                **chunk,
                'embedding': embedding.tolist()  # Convert numpy array to list for JSON serialization
            })
        
        # Generate persona and job embeddings
        persona_embedding = self.embedding_model.encode(persona_config['persona']['description'])
        job_embedding = self.embedding_model.encode(persona_config['job_to_be_done']['description'])
        
        # Rank chunks by relevance
        ranked_chunks = self.ranker.rank_chunks(
            chunk_embeddings,
            persona_embedding,
            job_embedding,
            persona_config,
            top_k=self.config.TOP_K_CHUNKS
        )
        
        return {
            'document_path': doc_path,
            'document_name': Path(doc_path).name,
            'total_sections': len(sections),
            'total_chunks': len(all_chunks),
            'ranked_chunks': ranked_chunks,
            'processing_metadata': {
                'chunk_size': self.config.CHUNK_SIZE,
                'chunk_overlap': self.config.CHUNK_OVERLAP,
                'model_name': self.config.MODEL_NAME,
                'top_k': self.config.TOP_K_CHUNKS
            }
        }
    
    def _generate_final_output(self, processed_docs: List[Dict[str, Any]], persona_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final structured output in the required format."""
        
        # Combine all ranked chunks from all documents
        all_ranked_chunks = []
        for doc in processed_docs:
            for chunk in doc['ranked_chunks']:
                chunk['source_document'] = doc['document_name']
                all_ranked_chunks.append(chunk)
        
        # Re-rank globally across all documents
        global_top_chunks = sorted(
            all_ranked_chunks, 
            key=lambda x: x['relevance_score'], 
            reverse=True
        )[:self.config.TOP_K_CHUNKS]
        
        # Format output according to challenge requirements
        extracted_sections = []
        sub_sections = []
        
        for i, chunk in enumerate(global_top_chunks):
            # Extract section information
            extracted_sections.append({
                "document": chunk['source_document'],
                "page_number": "N/A",  # PDF page extraction would need additional logic
                "section_title": chunk['section_title'],
                "importance_rank": i + 1
            })
            
            # Extract sub-section information
            sub_sections.append({
                "document": chunk['source_document'],
                "refined_text": chunk['content'],
                "page_number": "N/A"  # PDF page extraction would need additional logic
            })
        
        return {
            "metadata": {
                "input_documents": [doc['document_name'] for doc in processed_docs],
                "persona": {
                    "role": persona_config['persona'].get('role', 'Unknown'),
                    "description": persona_config['persona'].get('description', ''),
                    "expertise_areas": persona_config['persona'].get('expertise_areas', []),
                    "priorities": persona_config['persona'].get('priorities', [])
                },
                "job_to_be_done": {
                    "title": persona_config['job_to_be_done'].get('title', ''),
                    "description": persona_config['job_to_be_done'].get('description', ''),
                    "key_objectives": persona_config['job_to_be_done'].get('key_objectives', [])
                },
                "processing_timestamp": self._get_timestamp(),
                "processing_summary": {
                    "total_documents": len(processed_docs),
                    "total_sections": sum(doc['total_sections'] for doc in processed_docs),
                    "total_chunks": sum(doc['total_chunks'] for doc in processed_docs),
                    "final_top_chunks": len(global_top_chunks)
                }
            },
            "extracted_sections": extracted_sections,
            "sub_sections": sub_sections,
            # Keep detailed analysis for internal use
            "detailed_analysis": {
                "persona": persona_config['persona'],
                "job_to_be_done": persona_config['job_to_be_done'],
                "ranked_content": global_top_chunks,
                "document_details": [
                    {
                        "name": doc['document_name'],
                        "sections": doc['total_sections'],
                        "chunks": doc['total_chunks']
                    } for doc in processed_docs
                ]
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _generate_final_output_old(self, processed_docs: List[Dict[str, Any]], persona_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final structured output (original format)."""
        
        # Combine all ranked chunks from all documents
        all_ranked_chunks = []
        for doc in processed_docs:
            for chunk in doc['ranked_chunks']:
                chunk['source_document'] = doc['document_name']
                all_ranked_chunks.append(chunk)
        
        # Re-rank globally across all documents
        global_top_chunks = sorted(
            all_ranked_chunks, 
            key=lambda x: x['relevance_score'], 
            reverse=True
        )[:self.config.TOP_K_CHUNKS]
        
        return {
            'persona': persona_config['persona'],
            'job_to_be_done': persona_config['job_to_be_done'],
            'processing_summary': {
                'total_documents': len(processed_docs),
                'total_sections': sum(doc['total_sections'] for doc in processed_docs),
                'total_chunks': sum(doc['total_chunks'] for doc in processed_docs),
                'final_top_chunks': len(global_top_chunks)
            },
            'ranked_content': global_top_chunks,
            'document_details': [
                {
                    'name': doc['document_name'],
                    'sections': doc['total_sections'],
                    'chunks': doc['total_chunks']
                }
                for doc in processed_docs
            ]
        }
    
    def _save_results(self, output: Dict[str, Any], output_dir: str) -> str:
        """Save the final results to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / "persona_document_intelligence_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(output_file)


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description='Run persona-based document intelligence pipeline')
    parser.add_argument('--input-dir', default='data/input_documents', 
                       help='Directory containing PDF files')
    parser.add_argument('--output-dir', default='data/output',
                       help='Directory for output files')
    parser.add_argument('--persona-file', default=None,
                       help='Path to persona configuration file')
    parser.add_argument('--config', default=None,
                       help='Path to custom configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Settings()
    if args.config and os.path.exists(args.config):
        config.load_from_file(args.config)
    
    # Initialize and run pipeline
    pipeline = DocumentIntelligencePipeline(config)
    
    try:
        results = pipeline.run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            persona_file=args.persona_file
        )
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        
        # Handle both old and new output formats
        if 'metadata' in results and 'processing_summary' in results['metadata']:
            # New format
            summary = results['metadata']['processing_summary']
        elif 'processing_summary' in results:
            # Old format fallback
            summary = results['processing_summary']
        else:
            # Fallback if neither exists
            summary = {
                'total_documents': len(results.get('document_details', [])),
                'total_sections': 0,
                'total_chunks': 0,
                'final_top_chunks': len(results.get('ranked_content', []))
            }
        
        print(f"Processed {summary['total_documents']} documents")
        print(f"Found {summary['total_sections']} sections")
        print(f"Generated {summary['total_chunks']} chunks")
        print(f"Top {summary['final_top_chunks']} most relevant chunks identified")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())