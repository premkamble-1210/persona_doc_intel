"""
PDF text extraction utilities.
Supports multiple extraction methods for robust PDF processing.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
from io import BytesIO

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass


class PDFLoader:
    """
    PDF text extraction with multiple fallback methods.
    Handles various PDF formats and extraction challenges.
    """
    
    def __init__(self, extraction_method: str = 'auto'):
        """
        Initialize PDF loader.
        
        Args:
            extraction_method: 'pymupdf', 'pdfplumber', or 'auto' for automatic selection
        """
        self.extraction_method = extraction_method
        self.supported_methods = ['pymupdf', 'pdfplumber']
        
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            PDFExtractionError: If extraction fails with all methods
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Extracting text from: {pdf_path.name}")
        
        if self.extraction_method == 'auto':
            return self._extract_with_fallback(pdf_path)
        elif self.extraction_method in self.supported_methods:
            return self._extract_with_method(pdf_path, self.extraction_method)
        else:
            raise ValueError(f"Unsupported extraction method: {self.extraction_method}")
    
    def _extract_with_fallback(self, pdf_path: Path) -> str:
        """Extract text with automatic method fallback."""
        errors = []
        
        # Try each method in order of preference
        for method in ['pdfplumber', 'pymupdf']:
            try:
                text = self._extract_with_method(pdf_path, method)
                if text and len(text.strip()) > 0:
                    logger.info(f"Successfully extracted text using {method}")
                    return text
                else:
                    logger.warning(f"No text extracted using {method}")
            except Exception as e:
                error_msg = f"{method} failed: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        # If all methods fail
        raise PDFExtractionError(f"All extraction methods failed: {'; '.join(errors)}")
    
    def _extract_with_method(self, pdf_path: Path, method: str) -> str:
        """Extract text using specific method."""
        if method == 'pdfplumber':
            return self._extract_with_pdfplumber(pdf_path)
        elif method == 'pymupdf':
            return self._extract_with_pymupdf(pdf_path)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber (good for tables and complex layouts)."""
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            logger.debug(f"PDF has {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Clean and format text
                        cleaned_text = self._clean_text(page_text)
                        text_content.append(f"\n--- Page {page_num} ---\n{cleaned_text}")
                    
                    # Try to extract tables if text extraction yields little content
                    if not page_text or len(page_text.strip()) < 50:
                        tables = page.extract_tables()
                        if tables:
                            table_text = self._format_tables(tables)
                            text_content.append(f"\n--- Page {page_num} (Tables) ---\n{table_text}")
                
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {str(e)}")
                    continue
        
        return '\n'.join(text_content)
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (fast and reliable for most PDFs)."""
        text_content = []
        
        doc = fitz.open(pdf_path)
        
        try:
            logger.debug(f"PDF has {doc.page_count} pages")
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    
                    # Extract text with formatting preserved
                    page_text = page.get_text()
                    
                    if page_text:
                        cleaned_text = self._clean_text(page_text)
                        text_content.append(f"\n--- Page {page_num + 1} ---\n{cleaned_text}")
                    else:
                        # Try OCR-like extraction for scanned pages
                        blocks = page.get_text("dict")
                        block_text = self._extract_from_blocks(blocks)
                        if block_text:
                            text_content.append(f"\n--- Page {page_num + 1} (Blocks) ---\n{block_text}")
                
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num + 1}: {str(e)}")
                    continue
        
        finally:
            doc.close()
        
        return '\n'.join(text_content)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        
        # Remove page numbers (simple patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone numbers on lines
        
        # Remove headers/footers (lines with very few words)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Keep lines with substantial content
                words = line.split()
                if len(words) > 2 or any(len(word) > 3 for word in words):
                    cleaned_lines.append(line)
                elif len(words) <= 2 and not any(char.isdigit() for char in line):
                    # Keep short lines that might be headings (no digits)
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _format_tables(self, tables: List[List[List[str]]]) -> str:
        """Format extracted tables as text."""
        formatted_tables = []
        
        for i, table in enumerate(tables):
            if not table:
                continue
            
            table_text = [f"Table {i + 1}:"]
            
            for row in table:
                if row:  # Skip empty rows
                    # Clean and join cells
                    cells = [str(cell).strip() if cell else "" for cell in row]
                    if any(cells):  # Only add rows with content
                        table_text.append(" | ".join(cells))
            
            if len(table_text) > 1:  # Only add if has content beyond header
                formatted_tables.append("\n".join(table_text))
        
        return "\n\n".join(formatted_tables)
    
    def _extract_from_blocks(self, blocks_dict: Dict) -> str:
        """Extract text from PyMuPDF block structure."""
        text_parts = []
        
        if 'blocks' not in blocks_dict:
            return ""
        
        for block in blocks_dict['blocks']:
            if 'lines' in block:
                for line in block['lines']:
                    if 'spans' in line:
                        line_text = []
                        for span in line['spans']:
                            if 'text' in span:
                                line_text.append(span['text'])
                        if line_text:
                            text_parts.append(''.join(line_text))
        
        return '\n'.join(text_parts)
    
    def get_document_info(self, pdf_path: str) -> Dict[str, any]:
        """
        Get document metadata and basic information.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing document information
        """
        pdf_path = Path(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            
            info = {
                'filename': pdf_path.name,
                'file_size': pdf_path.stat().st_size,
                'page_count': doc.page_count,
                'metadata': doc.metadata,
                'is_encrypted': doc.is_encrypted,
                'is_pdf': doc.is_pdf
            }
            
            # Calculate approximate character count
            sample_pages = min(3, doc.page_count)
            total_chars = 0
            
            for i in range(sample_pages):
                page_text = doc[i].get_text()
                total_chars += len(page_text)
            
            # Estimate total characters
            if sample_pages > 0:
                avg_chars_per_page = total_chars / sample_pages
                info['estimated_total_chars'] = int(avg_chars_per_page * doc.page_count)
            else:
                info['estimated_total_chars'] = 0
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get document info: {str(e)}")
            return {
                'filename': pdf_path.name,
                'file_size': pdf_path.stat().st_size if pdf_path.exists() else 0,
                'error': str(e)
            }
    
    def extract_with_structure(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text with structural information (headings, paragraphs, etc.).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing structured text data
        """
        pdf_path = Path(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            structured_content = {
                'pages': [],
                'toc': doc.get_toc(),  # Table of contents if available
                'metadata': doc.metadata
            }
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text blocks with formatting
                blocks = page.get_text("dict")
                page_structure = self._analyze_page_structure(blocks)
                
                structured_content['pages'].append({
                    'page_number': page_num + 1,
                    'structure': page_structure
                })
            
            doc.close()
            return structured_content
            
        except Exception as e:
            logger.error(f"Failed to extract structured content: {str(e)}")
            raise PDFExtractionError(f"Structure extraction failed: {str(e)}")
    
    def _analyze_page_structure(self, blocks_dict: Dict) -> List[Dict]:
        """Analyze page structure to identify headings, paragraphs, etc."""
        elements = []
        
        if 'blocks' not in blocks_dict:
            return elements
        
        for block in blocks_dict['blocks']:
            if 'lines' not in block:
                continue
            
            block_text = []
            font_sizes = []
            
            for line in block['lines']:
                if 'spans' not in line:
                    continue
                
                line_text = []
                for span in line['spans']:
                    if 'text' in span and span['text'].strip():
                        line_text.append(span['text'])
                        if 'size' in span:
                            font_sizes.append(span['size'])
                
                if line_text:
                    block_text.append(''.join(line_text))
            
            if block_text:
                # Determine element type based on formatting
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                text_content = '\n'.join(block_text).strip()
                
                element_type = 'paragraph'
                if avg_font_size > 14:  # Likely heading
                    element_type = 'heading'
                elif len(text_content) < 100 and '\n' not in text_content:
                    element_type = 'title'
                
                elements.append({
                    'type': element_type,
                    'content': text_content,
                    'font_size': avg_font_size,
                    'bbox': block.get('bbox', [0, 0, 0, 0])
                })
        
        return elements


class PDFBatchProcessor:
    """Process multiple PDF files in batch."""
    
    def __init__(self, loader: PDFLoader = None):
        self.loader = loader or PDFLoader()
    
    def process_directory(self, directory_path: str, output_callback=None) -> Dict[str, str]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            output_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary mapping file paths to extracted text
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in: {directory_path}")
            return {}
        
        results = {}
        errors = []
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                logger.info(f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})")
                
                text = self.loader.extract_text(str(pdf_file))
                results[str(pdf_file)] = text
                
                if output_callback:
                    output_callback(pdf_file.name, i+1, len(pdf_files), success=True)
                    
            except Exception as e:
                error_msg = f"Failed to process {pdf_file.name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                if output_callback:
                    output_callback(pdf_file.name, i+1, len(pdf_files), success=False, error=str(e))
        
        if errors:
            logger.warning(f"Processing completed with {len(errors)} errors")
        
        return results
    
    def get_processing_stats(self, results: Dict[str, str]) -> Dict[str, any]:
        """Get statistics about processing results."""
        if not results:
            return {'total_files': 0, 'total_chars': 0, 'avg_chars_per_file': 0}
        
        total_chars = sum(len(text) for text in results.values())
        
        return {
            'total_files': len(results),
            'total_chars': total_chars,
            'avg_chars_per_file': total_chars // len(results),
            'file_stats': [
                {
                    'filename': Path(filepath).name,
                    'char_count': len(text),
                    'word_count': len(text.split()) if text else 0
                }
                for filepath, text in results.items()
            ]
        }


# Utility functions
def validate_pdf(pdf_path: str) -> bool:
    """Check if file is a valid PDF."""
    try:
        with fitz.open(pdf_path) as doc:
            return doc.is_pdf and doc.page_count > 0
    except:
        return False


def estimate_processing_time(pdf_path: str) -> float:
    """Estimate processing time in seconds based on file size."""
    try:
        file_size = Path(pdf_path).stat().st_size
        # Rough estimate: 1MB per 10 seconds
        return (file_size / (1024 * 1024)) * 10
    except:
        return 60.0  # Default estimate