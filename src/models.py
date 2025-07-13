from typing import List, Optional, Union
import os
import time
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from model2vec import StaticModel
import logging

from .config import (
    MODEL_NAME,
    USE_MODEL2VEC,
    MODEL_CACHE_PATH,
    EMBEDDING_DIMENSIONS,
    TRUST_REMOTE_CODE,
    ENABLE_QUANTIZATION,
    TORCH_COMPILE,
    get_cache_path,
    MODELS_DIR,
    ENABLE_HARDWARE_OPTIMIZATION,
    AUTO_DEVICE_SELECTION,
    ENABLE_TORCH_OPTIMIZATION
)

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str = None, use_model2vec: bool = None, device: str = None):
        self.model_name = model_name or MODEL_NAME
        self.use_model2vec = use_model2vec if use_model2vec is not None else USE_MODEL2VEC
        self.model = None
        self.device = device
        self.cache_path = get_cache_path(self.model_name)
        self.expected_dimensions = EMBEDDING_DIMENSIONS
        self.hardware_optimizer = None
        
        # Initialize hardware optimizer if enabled
        if ENABLE_HARDWARE_OPTIMIZATION:
            self._init_hardware_optimizer()
        
        self.load_model()
        self._optimize_model()

    def _is_model_cached(self) -> bool:
        """Check if model is already cached locally"""
        if not self.cache_path.exists():
            return False
        
        # Check for essential model files
        if self.use_model2vec:
            # Model2Vec typically has .safetensors and config.json
            required_files = ["config.json"]
            optional_files = ["model.safetensors", "pytorch_model.bin"]
        else:
            # SentenceTransformer models have these files
            required_files = ["config.json"]
            optional_files = ["pytorch_model.bin", "model.safetensors", "config_sentence_transformers.json"]
        
        # Check required files
        for file in required_files:
            if not (self.cache_path / file).exists():
                return False
        
        # Check at least one model file exists
        has_model_file = any((self.cache_path / file).exists() for file in optional_files)
        
        return has_model_file
    
    def _init_hardware_optimizer(self):
        """Initialize hardware optimizer for optimal performance"""
        try:
            from .hardware_optimizer import get_hardware_optimizer, OptimizationLevel
            from .config import OPTIMIZATION_LEVEL
            
            level_map = {
                "conservative": OptimizationLevel.CONSERVATIVE,
                "balanced": OptimizationLevel.BALANCED,
                "aggressive": OptimizationLevel.AGGRESSIVE
            }
            
            optimization_level = level_map.get(OPTIMIZATION_LEVEL, OptimizationLevel.BALANCED)
            self.hardware_optimizer = get_hardware_optimizer(optimization_level)
            
            # Select optimal device if not specified
            if AUTO_DEVICE_SELECTION and not self.device:
                self.device = self.hardware_optimizer.get_optimal_device()
                logger.info(f"Auto-selected device: {self.device}")
            
        except Exception as e:
            logger.warning(f"Hardware optimizer initialization failed: {e}")

    def load_model(self) -> None:
        try:
            # Ensure cache directory exists
            self.cache_path.mkdir(parents=True, exist_ok=True)
            
            if self._is_model_cached():
                logger.info(f"Loading cached model from: {self.cache_path}")
                model_path = str(self.cache_path)
            else:
                logger.info(f"Model not found in cache. Downloading: {self.model_name}")
                model_path = self.model_name
            
            if self.use_model2vec:
                logger.info(f"Loading Model2Vec model: {self.model_name}")
                if self._is_model_cached():
                    self.model = StaticModel.from_pretrained(model_path)
                else:
                    # Download and cache - Model2Vec uses HF cache by default
                    self.model = StaticModel.from_pretrained(self.model_name)
                    # Save to our cache structure if possible
                    try:
                        if hasattr(self.model, 'save_pretrained'):
                            self.model.save_pretrained(str(self.cache_path))
                        else:
                            logger.info("Model2Vec model downloaded, using HuggingFace cache")
                    except Exception as e:
                        logger.warning(f"Could not save model to custom cache: {e}")
            else:
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                if self._is_model_cached():
                    self.model = SentenceTransformer(model_path, trust_remote_code=TRUST_REMOTE_CODE)
                else:
                    # Download and cache
                    self.model = SentenceTransformer(
                        self.model_name,
                        cache_folder=str(MODELS_DIR),
                        trust_remote_code=TRUST_REMOTE_CODE
                    )
                    # Move to our cache structure if needed
                    if hasattr(self.model, 'save'):
                        self.model.save(str(self.cache_path))
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def encode(self, texts: Union[str, List[str]], task_type: str = "search_document") -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        if task_type == "search_query":
            prefixed_texts = [f"search_query: {text}" for text in texts]
        else:
            prefixed_texts = [f"search_document: {text}" for text in texts]
        
        try:
            start_time = time.time()
            
            # Move to optimal device if hardware optimization is enabled
            if self.device and hasattr(self.model, 'to') and ENABLE_HARDWARE_OPTIMIZATION:
                self.model = self.model.to(self.device)
            
            embeddings = self.model.encode(prefixed_texts)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            
            # Update workload profile for optimization
            if self.hardware_optimizer:
                processing_time = time.time() - start_time
                avg_seq_length = sum(len(text.split()) for text in prefixed_texts) / len(prefixed_texts)
                memory_usage = embeddings.nbytes / (1024 * 1024)  # MB
                
                self.hardware_optimizer.update_workload_profile(
                    batch_size=len(texts),
                    sequence_length=int(avg_seq_length),
                    processing_time=processing_time,
                    memory_usage=memory_usage
                )
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    def _optimize_model(self) -> None:
        """Apply performance optimizations to the model"""
        try:
            if self.model is None:
                return
            
            # Apply torch.compile if enabled (PyTorch 2.0+)
            if TORCH_COMPILE and hasattr(torch, 'compile'):
                logger.info("Applying torch.compile optimization...")
                if hasattr(self.model, 'encode'):
                    # For sentence-transformers, compile the underlying modules
                    if hasattr(self.model, '_modules'):
                        for module in self.model._modules.values():
                            if hasattr(module, 'forward'):
                                try:
                                    module = torch.compile(module, mode='default')
                                except Exception as e:
                                    logger.warning(f"Failed to compile module: {e}")
                
                logger.info("Model optimization applied")
            
            # Enable half precision if quantization is enabled and GPU is available
            if ENABLE_QUANTIZATION and torch.cuda.is_available():
                if hasattr(self.model, 'half'):
                    logger.info("Applying half precision optimization...")
                    self.model = self.model.half()
                elif hasattr(self.model, '_modules'):
                    # For sentence-transformers
                    try:
                        for module in self.model._modules.values():
                            if hasattr(module, 'half'):
                                module.half()
                        logger.info("Half precision applied to model modules")
                    except Exception as e:
                        logger.warning(f"Failed to apply half precision: {e}")
            
            # Warm up the model with a dummy inference
            self._warmup_model()
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
    
    def _warmup_model(self) -> None:
        """Warm up the model with dummy inference"""
        try:
            logger.info("Warming up model...")
            dummy_text = ["This is a warmup text for model initialization."]
            _ = self.encode(dummy_text, task_type="search_document")
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension, using expected dimensions from config"""
        return self.expected_dimensions


