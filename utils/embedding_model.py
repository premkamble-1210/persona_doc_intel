"""
Text embedding model utilities.
Handles loading and using transformer models for text embeddings.
"""

import os
import logging
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class EmbeddingModelError(Exception):
    """Custom exception for embedding model errors."""
    pass


class EmbeddingModel:
    """
    Text embedding model with caching and optimization.
    Supports various transformer models for generating text embeddings.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 max_model_size: int = 384, cache_dir: str = None, use_gpu: bool = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the transformer model to use
            max_model_size: Maximum embedding dimension
            cache_dir: Directory for caching embeddings
            use_gpu: Whether to use GPU if available
        """
        self.model_name = model_name
        self.max_model_size = max_model_size
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if use_gpu is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        self._load_model()
        
        # Cache for embeddings
        self.embedding_cache = {}
        self._load_cache()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            if self.model_name.startswith('sentence-transformers/'):
                # Use SentenceTransformers library
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                
                # Check if model size exceeds limit
                if self.embedding_dim > self.max_model_size:
                    logger.warning(f"Model dimension ({self.embedding_dim}) exceeds max size ({self.max_model_size})")
                    # Could implement dimensionality reduction here if needed
                
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
                
            else:
                # Use transformers library directly
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                
                # Get embedding dimension from model config
                self.embedding_dim = self.model.config.hidden_size
                
                if self.embedding_dim > self.max_model_size:
                    logger.warning(f"Model dimension ({self.embedding_dim}) exceeds max size ({self.max_model_size})")
                
                logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
                
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary model fails."""
        fallback_models = [
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2'
        ]
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                self.model = SentenceTransformer(fallback_model, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_name = fallback_model
                logger.info(f"Fallback model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Fallback model {fallback_model} failed: {str(e)}")
                continue
        
        raise EmbeddingModelError("Failed to load any embedding model")
    
    def encode(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode text into embeddings.
        
        Args:
            text: Text string or list of text strings
            batch_size: Batch size for processing multiple texts
            
        Returns:
            NumPy array of embeddings
        """
        if isinstance(text, str):
            return self._encode_single(text)
        elif isinstance(text, list):
            return self._encode_batch(text, batch_size)
        else:
            raise ValueError("Text must be string or list of strings")
    
    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        if not text or not text.strip():
            return np.zeros(self.embedding_dim)
        
        # Check cache first
        text_hash = self._get_text_hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Generate embedding
            if hasattr(self.model, 'encode'):
                # SentenceTransformers model
                embedding = self.model.encode(processed_text, convert_to_numpy=True)
            else:
                # HuggingFace transformers model
                embedding = self._encode_with_transformers(processed_text)
            
            # Ensure embedding is the right shape
            if embedding.ndim == 2:
                embedding = embedding[0]  # Take first (and only) embedding
            
            # Cache the result
            self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim)
    
    def _encode_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Encode a batch of texts."""
        if not texts:
            return np.zeros((0, self.embedding_dim))
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Check cache for each text
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                text_hash = self._get_text_hash(text)
                if text_hash in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[text_hash])
                else:
                    batch_embeddings.append(None)
                    uncached_texts.append(self._preprocess_text(text))
                    uncached_indices.append(j)
            
            # Encode uncached texts
            if uncached_texts:
                try:
                    if hasattr(self.model, 'encode'):
                        # SentenceTransformers model
                        uncached_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
                    else:
                        # HuggingFace transformers model
                        uncached_embeddings = np.array([
                            self._encode_with_transformers(text) for text in uncached_texts
                        ])
                    
                    # Insert uncached embeddings and cache them
                    for idx, embedding in zip(uncached_indices, uncached_embeddings):
                        if embedding.ndim == 2:
                            embedding = embedding[0]
                        batch_embeddings[idx] = embedding
                        text_hash = self._get_text_hash(batch[idx])
                        self.embedding_cache[text_hash] = embedding
                        
                except Exception as e:
                    logger.error(f"Failed to encode batch: {str(e)}")
                    # Fill with zero vectors
                    for idx in uncached_indices:
                        batch_embeddings[idx] = np.zeros(self.embedding_dim)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _encode_with_transformers(self, text: str) -> np.ndarray:
        """Encode text using HuggingFace transformers."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()[0]
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before encoding."""
        if not text:
            return ""
        
        # Clean text
        text = text.strip()
        
        # Truncate if too long (models have token limits)
        max_chars = 2000  # Approximate limit
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        return text
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        text_content = f"{self.model_name}:{text}"
        return hashlib.md5(text_content.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
                self.embedding_cache = {}
        else:
            self.embedding_cache = {}
    
    def _save_cache(self):
        """Save cached embeddings to disk."""
        if not self.embedding_cache:
            return
            
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache = {}
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_embeddings': len(self.embedding_cache),
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'cache_dir': str(self.cache_dir)
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self._save_cache()
        except:
            pass  # Ignore errors during cleanup