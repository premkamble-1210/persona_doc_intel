"""
Configuration settings for the persona document intelligence pipeline.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class Settings:
    """Configuration class for pipeline settings."""
    
    def __init__(self):
        # Model Configuration
        self.MODEL_NAME = os.getenv('MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
        self.MAX_MODEL_SIZE = int(os.getenv('MAX_MODEL_SIZE', 384))  # Max embedding dimension
        self.USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
        
        # Text Processing Configuration
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 512))  # Characters per chunk
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))  # Overlap between chunks
        self.MIN_CHUNK_SIZE = int(os.getenv('MIN_CHUNK_SIZE', 100))  # Minimum chunk size
        
        # Ranking Configuration
        self.TOP_K_CHUNKS = int(os.getenv('TOP_K_CHUNKS', 20))  # Top chunks to return
        self.SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))  # Min similarity score
        
        # Persona Weight Configuration
        self.PERSONA_WEIGHT = float(os.getenv('PERSONA_WEIGHT', 0.6))  # Weight for persona similarity (0-1)
        self.JOB_WEIGHT = float(os.getenv('JOB_WEIGHT', 0.4))  # Weight for job similarity (0-1)
        
        # Document Processing Configuration
        self.MAX_DOCUMENT_SIZE = int(os.getenv('MAX_DOCUMENT_SIZE', 10_000_000))  # Max chars per document
        self.SUPPORTED_FORMATS = ['pdf']
        
        # Output Configuration
        self.OUTPUT_FORMAT = os.getenv('OUTPUT_FORMAT', 'json')
        self.INCLUDE_EMBEDDINGS = os.getenv('INCLUDE_EMBEDDINGS', 'false').lower() == 'true'
        self.INCLUDE_METADATA = os.getenv('INCLUDE_METADATA', 'true').lower() == 'true'
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.getenv('LOG_FILE', None)
        
        # Cache Configuration
        self.ENABLE_CACHE = os.getenv('ENABLE_CACHE', 'true').lower() == 'true'
        self.CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate weights sum to 1.0
        total_weight = self.PERSONA_WEIGHT + self.JOB_WEIGHT
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"PERSONA_WEIGHT + JOB_WEIGHT must equal 1.0, got {total_weight}")
        
        # Validate positive values
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.TOP_K_CHUNKS <= 0:
            raise ValueError("TOP_K_CHUNKS must be positive")
        if not 0 <= self.SIMILARITY_THRESHOLD <= 1:
            raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
    
    def load_from_file(self, config_file: str):
        """Load configuration from a JSON file."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update settings from file
        for key, value in config_data.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)
        
        # Re-validate after loading
        self._validate_config()
    
    def save_to_file(self, config_file: str):
        """Save current configuration to a JSON file."""
        config_data = {
            'model_name': self.MODEL_NAME,
            'max_model_size': self.MAX_MODEL_SIZE,
            'use_gpu': self.USE_GPU,
            'chunk_size': self.CHUNK_SIZE,
            'chunk_overlap': self.CHUNK_OVERLAP,
            'min_chunk_size': self.MIN_CHUNK_SIZE,
            'top_k_chunks': self.TOP_K_CHUNKS,
            'similarity_threshold': self.SIMILARITY_THRESHOLD,
            'persona_weight': self.PERSONA_WEIGHT,
            'job_weight': self.JOB_WEIGHT,
            'max_document_size': self.MAX_DOCUMENT_SIZE,
            'output_format': self.OUTPUT_FORMAT,
            'include_embeddings': self.INCLUDE_EMBEDDINGS,
            'include_metadata': self.INCLUDE_METADATA,
            'log_level': self.LOG_LEVEL,
            'enable_cache': self.ENABLE_CACHE,
            'cache_dir': self.CACHE_DIR
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            'model_name': self.MODEL_NAME,
            'max_model_size': self.MAX_MODEL_SIZE,
            'use_gpu': self.USE_GPU,
            'cache_dir': self.CACHE_DIR,
            'enable_cache': self.ENABLE_CACHE
        }
    
    def get_chunking_config(self) -> Dict[str, Any]:
        """Get text chunking configuration."""
        return {
            'chunk_size': self.CHUNK_SIZE,
            'chunk_overlap': self.CHUNK_OVERLAP,
            'min_chunk_size': self.MIN_CHUNK_SIZE,
            'max_document_size': self.MAX_DOCUMENT_SIZE
        }
    
    def get_ranking_config(self) -> Dict[str, Any]:
        """Get ranking configuration."""
        return {
            'top_k_chunks': self.TOP_K_CHUNKS,
            'similarity_threshold': self.SIMILARITY_THRESHOLD,
            'persona_weight': self.PERSONA_WEIGHT,
            'job_weight': self.JOB_WEIGHT
        }
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return {
            'output_format': self.OUTPUT_FORMAT,
            'include_embeddings': self.INCLUDE_EMBEDDINGS,
            'include_metadata': self.INCLUDE_METADATA
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""Settings Configuration:
Model: {self.MODEL_NAME} (max_size: {self.MAX_MODEL_SIZE})
Chunking: size={self.CHUNK_SIZE}, overlap={self.CHUNK_OVERLAP}
Ranking: top_k={self.TOP_K_CHUNKS}, threshold={self.SIMILARITY_THRESHOLD}
Weights: persona={self.PERSONA_WEIGHT}, job={self.JOB_WEIGHT}
Output: format={self.OUTPUT_FORMAT}, embeddings={self.INCLUDE_EMBEDDINGS}
"""


# Default configuration for different use cases
class DevelopmentSettings(Settings):
    """Settings optimized for development and testing."""
    
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 256
        self.TOP_K_CHUNKS = 10
        self.LOG_LEVEL = 'DEBUG'
        self.ENABLE_CACHE = False


class ProductionSettings(Settings):
    """Settings optimized for production deployment."""
    
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 1024
        self.TOP_K_CHUNKS = 50
        self.LOG_LEVEL = 'INFO'
        self.ENABLE_CACHE = True


class LightweightSettings(Settings):
    """Settings for resource-constrained environments."""
    
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2'
        self.MAX_MODEL_SIZE = 384
        self.CHUNK_SIZE = 256
        self.TOP_K_CHUNKS = 15
        self.USE_GPU = False


def get_settings(profile: str = 'default') -> Settings:
    """Factory function to get settings based on profile."""
    profiles = {
        'default': Settings,
        'development': DevelopmentSettings,
        'production': ProductionSettings,
        'lightweight': LightweightSettings
    }
    
    if profile not in profiles:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(profiles.keys())}")
    
    return profiles[profile]()