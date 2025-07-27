"""
Text chunking utilities for document processing.
Handles splitting large documents into smaller, manageable chunks.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """
    Text chunking utility that splits documents into smaller chunks
    while preserving context and maintaining semantic boundaries.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap: int = 200,
                 preserve_sentences: bool = True,
                 preserve_paragraphs: bool = True,
                 min_chunk_size: int = 100):
        """
        Initialize TextChunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of characters to overlap between chunks
            preserve_sentences: Whether to avoid breaking sentences
            preserve_paragraphs: Whether to prefer paragraph boundaries
            min_chunk_size: Minimum chunk size to avoid very small chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_sentences = preserve_sentences
        self.preserve_paragraphs = preserve_paragraphs
        self.min_chunk_size = min_chunk_size
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        
        logger.info(f"TextChunker initialized: chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean and normalize text
        text = self._normalize_text(text)
        
        if len(text) <= self.chunk_size:
            # Text is small enough to be a single chunk
            return [TextChunk(
                text=text,
                start_pos=0,
                end_pos=len(text),
                chunk_id=0,
                metadata=metadata or {}
            )]
        
        # Choose chunking strategy based on preferences
        if self.preserve_paragraphs:
            chunks = self._chunk_by_paragraphs(text, metadata)
        elif self.preserve_sentences:
            chunks = self._chunk_by_sentences(text, metadata)
        else:
            chunks = self._chunk_by_characters(text, metadata)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)     # Normalize spaces
        text = text.strip()
        return text
    
    def _chunk_by_paragraphs(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunk text by paragraph boundaries."""
        chunks = []
        paragraphs = self.paragraph_breaks.split(text)
        
        current_chunk = ""
        current_pos = 0
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                # Add paragraph to current chunk
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_end = current_pos + len(current_chunk)
                    chunks.append(TextChunk(
                        text=current_chunk,
                        start_pos=current_pos,
                        end_pos=chunk_end,
                        chunk_id=chunk_id,
                        metadata=metadata or {}
                    ))
                    
                    # Calculate overlap position
                    current_pos = max(0, chunk_end - self.overlap)
                    chunk_id += 1
                
                current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk,
                start_pos=current_pos,
                end_pos=current_pos + len(current_chunk),
                chunk_id=chunk_id,
                metadata=metadata or {}
            ))
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunk text by sentence boundaries."""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_pos = 0
        chunk_id = 0
        
        for sentence in sentences:
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunk_end = current_pos + len(current_chunk)
                    chunks.append(TextChunk(
                        text=current_chunk,
                        start_pos=current_pos,
                        end_pos=chunk_end,
                        chunk_id=chunk_id,
                        metadata=metadata or {}
                    ))
                    
                    current_pos = max(0, chunk_end - self.overlap)
                    chunk_id += 1
                
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk,
                start_pos=current_pos,
                end_pos=current_pos + len(current_chunk),
                chunk_id=chunk_id,
                metadata=metadata or {}
            ))
        
        return chunks
    
    def _chunk_by_characters(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Chunk text by character count."""
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_start = i
            chunk_end = min(i + self.chunk_size, len(text))
            chunk_text = text[chunk_start:chunk_end]
            
            if chunk_text.strip():
                chunks.append(TextChunk(
                    text=chunk_text,
                    start_pos=chunk_start,
                    end_pos=chunk_end,
                    chunk_id=chunk_id,
                    metadata=metadata or {}
                ))
                chunk_id += 1
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with more sophisticated NLP
        sentences = []
        current_pos = 0
        
        for match in self.sentence_endings.finditer(text):
            sentence = text[current_pos:match.end()].strip()
            if sentence:
                sentences.append(sentence)
            current_pos = match.end()
        
        # Add remaining text as final sentence
        if current_pos < len(text):
            final_sentence = text[current_pos:].strip()
            if final_sentence:
                sentences.append(final_sentence)
        
        return sentences
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-process chunks to ensure quality."""
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Skip chunks that are too small (except the last one)
            if len(chunk.text.strip()) < self.min_chunk_size and i < len(chunks) - 1:
                # Merge with next chunk if possible
                if i + 1 < len(chunks):
                    next_chunk = chunks[i + 1]
                    merged_text = chunk.text + " " + next_chunk.text
                    
                    # Update next chunk with merged content
                    chunks[i + 1] = TextChunk(
                        text=merged_text,
                        start_pos=chunk.start_pos,
                        end_pos=next_chunk.end_pos,
                        chunk_id=chunk.chunk_id,
                        metadata=chunk.metadata
                    )
                    continue
            
            processed_chunks.append(chunk)
        
        # Renumber chunk IDs
        for i, chunk in enumerate(processed_chunks):
            chunk.chunk_id = i
        
        return processed_chunks
    
    def extract_sections(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Extract document sections based on headings and structure.
        
        Args:
            text: Input text to extract sections from
            metadata: Optional metadata to attach to sections
            
        Returns:
            List of section dictionaries with 'title', 'content', 'level', and 'metadata'
        """
        sections = []
        
        # Common heading patterns
        heading_patterns = [
            (r'^#{1,6}\s+(.+)$', 'markdown'),  # Markdown headings
            (r'^(.+)\n[=-]{3,}$', 'underline'),  # Underlined headings
            (r'^\d+\.\s+(.+)$', 'numbered'),  # 1. Numbered headings
            (r'^[A-Z][A-Z\s]{2,}$', 'allcaps'),  # ALL CAPS headings
            (r'^(.+):$', 'colon'),  # Colon-terminated headings
        ]
        
        lines = text.split('\n')
        current_section = {
            'title': 'Introduction',
            'content': '',
            'level': 0,
            'start_line': 0,
            'metadata': metadata or {}
        }
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            if not line_stripped:
                current_section['content'] += '\n'
                continue
            
            # Check if line matches any heading pattern
            heading_found = False
            for pattern, heading_type in heading_patterns:
                match = re.match(pattern, line_stripped, re.MULTILINE)
                if match:
                    # Save current section if it has content
                    if current_section['content'].strip():
                        current_section['end_line'] = i - 1
                        sections.append(current_section.copy())
                    
                    # Start new section
                    title = match.group(1) if heading_type != 'allcaps' else line_stripped
                    level = self._determine_heading_level(line_stripped, heading_type)
                    
                    current_section = {
                        'title': title.strip(),
                        'content': '',
                        'level': level,
                        'start_line': i,
                        'heading_type': heading_type,
                        'metadata': metadata or {}
                    }
                    heading_found = True
                    break
            
            if not heading_found:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            current_section['end_line'] = len(lines) - 1
            sections.append(current_section)
        
        # Clean up sections
        for section in sections:
            section['content'] = section['content'].strip()
            section['word_count'] = len(section['content'].split())
            section['char_count'] = len(section['content'])
        
        return sections
    
    def _determine_heading_level(self, line: str, heading_type: str) -> int:
        """Determine the hierarchical level of a heading."""
        if heading_type == 'markdown':
            return len(re.match(r'^#+', line.strip()).group())
        elif heading_type == 'numbered':
            # Count dots to determine nesting level
            match = re.match(r'^(\d+\.)+', line.strip())
            if match:
                return match.group().count('.') 
            return 1
        elif heading_type in ['underline', 'allcaps', 'colon']:
            return 1  # Default level for these types
        return 1
    
    def get_chunk_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {
                'total_chunks': 0,
                'total_length': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0
            }
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'total_length': sum(chunk_sizes),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes)
        }


class AdaptiveTextChunker(TextChunker):
    """
    Advanced text chunker that adapts chunk size based on content type.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.content_patterns = {
            'code': re.compile(r'```|class\s+\w+|def\s+\w+|function\s+\w+'),
            'list': re.compile(r'^\s*[-*•]\s+|^\s*\d+\.\s+', re.MULTILINE),
            'table': re.compile(r'\|.*\||\t.*\t'),
            'headers': re.compile(r'^#{1,6}\s+', re.MULTILINE)
        }
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Adaptive chunking based on content type."""
        content_type = self._detect_content_type(text)
        
        # Adjust parameters based on content type
        original_chunk_size = self.chunk_size
        original_preserve_paragraphs = self.preserve_paragraphs
        
        if content_type == 'code':
            self.chunk_size = int(self.chunk_size * 1.5)  # Larger chunks for code
            self.preserve_paragraphs = False
        elif content_type == 'list':
            self.preserve_paragraphs = True
        elif content_type == 'table':
            self.chunk_size = int(self.chunk_size * 2)  # Much larger for tables
        
        # Add content type to metadata
        if metadata is None:
            metadata = {}
        metadata['content_type'] = content_type
        
        # Perform chunking
        chunks = super().chunk_text(text, metadata)
        
        # Restore original settings
        self.chunk_size = original_chunk_size
        self.preserve_paragraphs = original_preserve_paragraphs
        
        return chunks
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the primary content type of the text."""
        scores = {}
        
        for content_type, pattern in self.content_patterns.items():
            matches = len(pattern.findall(text))
            scores[content_type] = matches / len(text) * 1000  # Normalize by length
        
        # Return the content type with highest score, or 'text' as default
        if scores:
            max_type = max(scores, key=scores.get)
            if scores[max_type] > 0.1:  # Threshold for detection
                return max_type
        
        return 'text'
