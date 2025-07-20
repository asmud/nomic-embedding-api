import logging
import time
from typing import List, Union

import numpy as np
import torch
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer

from .config import (AUTO_DEVICE_SELECTION, EMBEDDING_DIMENSIONS,
                     ENABLE_HARDWARE_OPTIMIZATION, ENABLE_QUANTIZATION,
                     MODEL_NAME, MODELS_DIR, TORCH_COMPILE, TRUST_REMOTE_CODE,
                     USE_MODEL2VEC)

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name: str = None, use_model2vec: bool = None, device: str = None):
        self.model_name = model_name or MODEL_NAME
        self.use_model2vec = use_model2vec if use_model2vec is not None else USE_MODEL2VEC
        self.model = None
        self.device = device
        self.expected_dimensions = EMBEDDING_DIMENSIONS
        self.hardware_optimizer = None
        
        # Initialize hardware optimizer if enabled
        if ENABLE_HARDWARE_OPTIMIZATION:
            self._init_hardware_optimizer()
        
        self.load_model()
        self._optimize_model()

    
    def _init_hardware_optimizer(self):
        """Initialize hardware optimizer for optimal performance"""
        try:
            from .hardware_optimizer import get_hardware_optimizer, OptimizationLevel
            
            # Use balanced optimization level as default
            self.hardware_optimizer = get_hardware_optimizer(OptimizationLevel.BALANCED)
            
            # Select optimal device if not specified
            if AUTO_DEVICE_SELECTION and not self.device:
                self.device = self.hardware_optimizer.get_optimal_device()
                logger.info(f"Auto-selected device: {self.device}")
            
        except Exception as e:
            logger.warning(f"Hardware optimizer initialization failed: {e}")

    def load_model(self) -> None:
        try:
            logger.info(f"Loading {self.model_name}")
            
            if self.use_model2vec:
                self.model = StaticModel.from_pretrained(self.model_name)
            else:
                self.model = SentenceTransformer(
                    self.model_name,
                    cache_folder=str(MODELS_DIR),
                    trust_remote_code=TRUST_REMOTE_CODE
                )
            
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
            dummy_text = ["Warmup text"]
            _ = self.encode(dummy_text, task_type="search_document")
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension, using expected dimensions from config"""
        return self.expected_dimensions


