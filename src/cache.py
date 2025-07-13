import hashlib
import logging
import threading
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np

from .config import CACHE_SIZE, ENABLE_CACHING, REDIS_ENABLED

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache for embeddings with dynamic sizing"""
    
    def __init__(self, max_size: int):
        self.initial_max_size = max_size
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.resize_count = 0
    
    def _generate_key(self, text: str, model: str, task_type: str) -> str:
        """Generate cache key from text, model, and task type"""
        content = f"{text}:{model}:{task_type}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, text: str, model: str, task_type: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if not ENABLE_CACHING:
            return None
            
        key = self._generate_key(text, model, task_type)
        
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                embedding = self.cache.pop(key)
                self.cache[key] = embedding
                self.hits += 1
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return embedding.copy()  # Return copy to avoid mutations
            else:
                self.misses += 1
                logger.debug(f"Cache miss for key: {key[:8]}...")
                return None
    
    def put(self, text: str, model: str, task_type: str, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        if not ENABLE_CACHING:
            return
            
        key = self._generate_key(text, model, task_type)
        
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Add to end
            self.cache[key] = embedding.copy()
            
            # Remove oldest if over capacity
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.evictions += 1
                logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
    
    def get_batch(self, texts: List[str], model: str, task_type: str) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """Get multiple embeddings from cache, return embeddings and indices of misses"""
        if not ENABLE_CACHING:
            return [None] * len(texts), list(range(len(texts)))
        
        embeddings = []
        miss_indices = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text, model, task_type)
            embeddings.append(embedding)
            if embedding is None:
                miss_indices.append(i)
        
        return embeddings, miss_indices
    
    def put_batch(self, texts: List[str], model: str, task_type: str, embeddings: List[np.ndarray]) -> None:
        """Store multiple embeddings in cache"""
        if not ENABLE_CACHING:
            return
            
        for text, embedding in zip(texts, embeddings):
            self.put(text, model, task_type, embedding)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("Cache cleared")
    
    def resize(self, new_max_size: int) -> None:
        """Dynamically resize the cache"""
        with self.lock:
            old_size = self.max_size
            self.max_size = max(10, new_max_size)  # Minimum size of 10
            self.resize_count += 1
            
            # Evict entries if new size is smaller
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.evictions += 1
            
            if old_size != self.max_size:
                logger.info(f"Cache resized from {old_size} to {self.max_size}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "initial_max_size": self.initial_max_size,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "resize_count": self.resize_count,
                "hit_rate": round(hit_rate, 2),
                "total_requests": total_requests,
                "enabled": ENABLE_CACHING,
                "utilization": round(len(self.cache) / self.max_size * 100, 1) if self.max_size > 0 else 0
            }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage in MB"""
        with self.lock:
            total_bytes = 0
            for embedding in self.cache.values():
                total_bytes += embedding.nbytes
            
            return {
                "cache_size_mb": round(total_bytes / (1024 * 1024), 2),
                "avg_embedding_kb": round(total_bytes / len(self.cache) / 1024, 2) if self.cache else 0
            }


class HybridCache:
    """Hybrid cache that uses both local LRU and Redis"""
    
    def __init__(self, max_size: int):
        self.local_cache = LRUCache(max_size)
        self.redis_client = None
        self.redis_stats = {"hits": 0, "misses": 0, "errors": 0}
        
        # Register with memory manager for dynamic resizing
        self._register_with_memory_manager()
        
    async def _get_redis_client(self):
        """Get Redis client if enabled"""
        if not REDIS_ENABLED:
            return None
            
        if self.redis_client is None:
            try:
                from .redis_client import get_redis_client
                self.redis_client = await get_redis_client()
            except Exception as e:
                logger.error(f"Failed to get Redis client: {e}")
                return None
        
        return self.redis_client
    
    def get(self, text: str, model: str, task_type: str) -> Optional[np.ndarray]:
        """Get embedding from local cache (sync)"""
        return self.local_cache.get(text, model, task_type)
    
    async def get_async(self, text: str, model: str, task_type: str) -> Optional[np.ndarray]:
        """Get embedding from local cache first, then Redis"""
        # Try local cache first
        local_result = self.local_cache.get(text, model, task_type)
        if local_result is not None:
            return local_result
        
        # Try Redis if enabled
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                redis_result = await redis_client.get_embeddings([text], model, task_type)
                if redis_result and len(redis_result) > 0:
                    embedding = np.array(redis_result[0])
                    # Store in local cache for faster access
                    self.local_cache.put(text, model, task_type, embedding)
                    self.redis_stats["hits"] += 1
                    return embedding
                else:
                    self.redis_stats["misses"] += 1
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
                self.redis_stats["errors"] += 1
        
        return None
    
    def put(self, text: str, model: str, task_type: str, embedding: np.ndarray) -> None:
        """Store embedding in local cache (sync)"""
        self.local_cache.put(text, model, task_type, embedding)
    
    async def put_async(self, text: str, model: str, task_type: str, embedding: np.ndarray) -> None:
        """Store embedding in both local cache and Redis"""
        # Store in local cache
        self.local_cache.put(text, model, task_type, embedding)
        
        # Store in Redis if enabled
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                await redis_client.set_embeddings([text], model, task_type, [embedding.tolist()])
            except Exception as e:
                logger.error(f"Redis cache store error: {e}")
                self.redis_stats["errors"] += 1
    
    def get_batch(self, texts: List[str], model: str, task_type: str) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """Get multiple embeddings from local cache (sync)"""
        return self.local_cache.get_batch(texts, model, task_type)
    
    async def get_batch_async(self, texts: List[str], model: str, task_type: str) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """Get multiple embeddings from local cache first, then Redis for misses"""
        # Try local cache first
        embeddings, miss_indices = self.local_cache.get_batch(texts, model, task_type)
        
        # Try Redis for misses if enabled
        if miss_indices and REDIS_ENABLED:
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    miss_texts = [texts[i] for i in miss_indices]
                    redis_result = await redis_client.get_embeddings(miss_texts, model, task_type)
                    
                    if redis_result:
                        # Fill in Redis hits
                        for i, redis_embedding in enumerate(redis_result):
                            if redis_embedding is not None:
                                miss_idx = miss_indices[i]
                                embedding = np.array(redis_embedding)
                                embeddings[miss_idx] = embedding
                                # Store in local cache
                                self.local_cache.put(texts[miss_idx], model, task_type, embedding)
                                self.redis_stats["hits"] += 1
                        
                        # Update miss indices to only include those not found in Redis
                        remaining_misses = []
                        for i, miss_idx in enumerate(miss_indices):
                            if embeddings[miss_idx] is None:
                                remaining_misses.append(miss_idx)
                            else:
                                self.redis_stats["misses"] += 1
                        miss_indices = remaining_misses
                        
                except Exception as e:
                    logger.error(f"Redis batch cache error: {e}")
                    self.redis_stats["errors"] += 1
        
        return embeddings, miss_indices
    
    def put_batch(self, texts: List[str], model: str, task_type: str, embeddings: List[np.ndarray]) -> None:
        """Store multiple embeddings in local cache (sync)"""
        self.local_cache.put_batch(texts, model, task_type, embeddings)
    
    async def put_batch_async(self, texts: List[str], model: str, task_type: str, embeddings: List[np.ndarray]) -> None:
        """Store multiple embeddings in both local cache and Redis"""
        # Store in local cache
        self.local_cache.put_batch(texts, model, task_type, embeddings)
        
        # Store in Redis if enabled
        redis_client = await self._get_redis_client()
        if redis_client:
            try:
                embeddings_list = [emb.tolist() for emb in embeddings]
                await redis_client.set_embeddings(texts, model, task_type, embeddings_list)
            except Exception as e:
                logger.error(f"Redis batch cache store error: {e}")
                self.redis_stats["errors"] += 1
    
    def clear(self) -> None:
        """Clear local cache (sync)"""
        self.local_cache.clear()
    
    async def clear_async(self, clear_redis: bool = False) -> None:
        """Clear local cache and optionally Redis"""
        self.local_cache.clear()
        
        if clear_redis:
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    await redis_client.clear_cache()
                except Exception as e:
                    logger.error(f"Redis clear error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        local_stats = self.local_cache.get_stats()
        local_memory = self.local_cache.get_memory_usage()
        
        stats = {
            "local": {**local_stats, **local_memory},
            "redis": {
                "enabled": REDIS_ENABLED,
                **self.redis_stats
            }
        }
        
        # Add Redis stats if available
        if REDIS_ENABLED:
            redis_client = await self._get_redis_client()
            if redis_client:
                try:
                    redis_info = await redis_client.get_stats()
                    stats["redis"].update(redis_info)
                except Exception as e:
                    stats["redis"]["error"] = str(e)
        
        return stats
    
    def _register_with_memory_manager(self):
        """Register cache resize callback with memory manager"""
        try:
            from .memory_manager import get_memory_manager
            from .config import ENABLE_DYNAMIC_MEMORY
            
            if ENABLE_DYNAMIC_MEMORY:
                memory_manager = get_memory_manager()
                memory_manager.register_cache_resize_callback(self.resize_cache)
                logger.info("Cache registered with memory manager for dynamic resizing")
                
        except Exception as e:
            logger.warning(f"Failed to register with memory manager: {e}")
    
    def resize_cache(self, new_size: int):
        """Resize the local cache"""
        self.local_cache.resize(new_size)


# Global cache instance
embedding_cache = HybridCache(CACHE_SIZE)


async def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics"""
    return await embedding_cache.get_stats()


def get_cache_stats_sync() -> Dict[str, Any]:
    """Get local cache statistics (sync version for compatibility)"""
    local_stats = embedding_cache.local_cache.get_stats()
    local_memory = embedding_cache.local_cache.get_memory_usage()
    return {
        "local": {**local_stats, **local_memory},
        "redis": {
            "enabled": REDIS_ENABLED,
            **embedding_cache.redis_stats
        }
    }


async def clear_cache(clear_redis: bool = False) -> None:
    """Clear global cache"""
    await embedding_cache.clear_async(clear_redis)


def clear_cache_sync() -> None:
    """Clear local cache (sync version for compatibility)"""
    embedding_cache.clear()