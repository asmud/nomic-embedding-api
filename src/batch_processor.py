import asyncio
import logging
import time
import heapq
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .config import MAX_BATCH_SIZE, BATCH_TIMEOUT_MS, MODEL_POOL_SIZE, ENABLE_DYNAMIC_MEMORY
from .cache import embedding_cache
from .model_pool import ModelPool, model_pool

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    URGENT = 0


@dataclass
class EmbeddingRequest:
    """Individual embedding request"""
    texts: List[str]
    model: str
    task_type: str
    request_id: str
    timestamp: float
    future: asyncio.Future
    priority: RequestPriority = RequestPriority.NORMAL
    user_id: Optional[str] = None
    estimated_processing_time: float = 0.0


@dataclass
class PriorityRequest:
    """Wrapper for priority queue with proper ordering"""
    priority: int
    timestamp: float
    request: EmbeddingRequest
    
    def __lt__(self, other):
        # First by priority (lower number = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        # Then by timestamp (older requests first)
        return self.timestamp < other.timestamp


@dataclass
class BatchResult:
    """Result of batch processing"""
    embeddings: List[np.ndarray]
    cache_hits: int
    cache_misses: int
    processing_time: float
    batch_size: int


class SmartBatchProcessor:
    """Smart batching processor for embedding requests with model pool and priority support"""
    
    def __init__(self):
        self.pending_requests: List[EmbeddingRequest] = []
        self.processing_lock = asyncio.Lock()
        self.priority_queue: List[PriorityRequest] = []
        self.queue_condition = asyncio.Condition()
        self.model_pool_enabled = MODEL_POOL_SIZE > 0
        
        # Dynamic batch sizing
        self.current_max_batch_size = MAX_BATCH_SIZE
        self.dynamic_memory_enabled = ENABLE_DYNAMIC_MEMORY
        
        # User rate limiting (requests per minute)
        self.user_rate_limits: Dict[str, List[float]] = {}
        self.rate_limit_window = 60.0  # 1 minute
        self.default_rate_limit = 100  # requests per minute
        
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_cache_hits": 0,
            "total_cache_misses": 0,
            "avg_batch_size": 0,
            "avg_processing_time": 0,
            "model_pool_enabled": self.model_pool_enabled,
            "priority_breakdown": {
                "urgent": 0,
                "high": 0,
                "normal": 0,
                "low": 0
            },
            "avg_queue_wait_time": 0.0,
            "dynamic_batch_size": self.current_max_batch_size,
            "initial_batch_size": MAX_BATCH_SIZE
        }
        
        # Register with memory manager for dynamic batch sizing
        if self.dynamic_memory_enabled:
            self._register_with_memory_manager()
    
    async def add_request(
        self, 
        texts: List[str], 
        model: str, 
        task_type: str, 
        request_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        user_id: Optional[str] = None
    ) -> List[np.ndarray]:
        """Add request to priority queue with rate limiting"""
        
        # Check rate limiting for user
        if user_id and not await self._check_rate_limit(user_id):
            raise Exception(f"Rate limit exceeded for user {user_id}")
        
        future = asyncio.get_event_loop().create_future()
        timestamp = time.time()
        
        # Estimate processing time based on text length
        estimated_time = self._estimate_processing_time(texts)
        
        request = EmbeddingRequest(
            texts=texts,
            model=model,
            task_type=task_type,
            request_id=request_id,
            timestamp=timestamp,
            future=future,
            priority=priority,
            user_id=user_id,
            estimated_processing_time=estimated_time
        )
        
        priority_request = PriorityRequest(
            priority=priority.value,
            timestamp=timestamp,
            request=request
        )
        
        async with self.queue_condition:
            heapq.heappush(self.priority_queue, priority_request)
            self.queue_condition.notify()
        
        self.stats["total_requests"] += 1
        self.stats["priority_breakdown"][priority.name.lower()] += 1
        
        # Wait for result
        try:
            result = await future
            queue_wait_time = time.time() - timestamp
            self._update_queue_wait_stats(queue_wait_time)
            return result
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
    
    async def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limit (async with Redis support)"""
        current_time = time.time()
        
        # Try Redis first for distributed rate limiting
        try:
            from .redis_client import get_redis_client
            redis_client = await get_redis_client()
            
            if redis_client:
                # Get user's request history from Redis
                user_requests = await redis_client.get_user_rate_limit(user_id)
                
                # Clean old requests outside the window
                user_requests = [t for t in user_requests if current_time - t < self.rate_limit_window]
                
                # Check if under limit
                if len(user_requests) >= self.default_rate_limit:
                    return False
                
                # Add current request and update Redis
                user_requests.append(current_time)
                await redis_client.set_user_rate_limit(user_id, user_requests)
                return True
                
        except Exception as e:
            logger.warning(f"Redis rate limiting failed, falling back to local: {e}")
        
        # Fallback to local rate limiting
        if user_id not in self.user_rate_limits:
            self.user_rate_limits[user_id] = []
        
        # Clean old requests outside the window
        user_requests = self.user_rate_limits[user_id]
        user_requests[:] = [t for t in user_requests if current_time - t < self.rate_limit_window]
        
        # Check if under limit
        if len(user_requests) >= self.default_rate_limit:
            return False
        
        # Add current request
        user_requests.append(current_time)
        return True
    
    def _estimate_processing_time(self, texts: List[str]) -> float:
        """Estimate processing time based on text characteristics"""
        total_chars = sum(len(text) for text in texts)
        # Simple heuristic: ~0.001 seconds per 100 characters
        return max(0.01, total_chars / 100000.0)
    
    def _update_queue_wait_stats(self, wait_time: float):
        """Update average queue wait time statistics"""
        current_avg = self.stats["avg_queue_wait_time"]
        total_requests = self.stats["total_requests"]
        self.stats["avg_queue_wait_time"] = (
            (current_avg * (total_requests - 1) + wait_time) / total_requests
        )
    
    async def process_batches(self, model_instance):
        """Main batch processing loop"""
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch, model_instance)
                else:
                    # Small delay if no requests
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[EmbeddingRequest]:
        """Collect requests into optimal batches using priority queue"""
        batch = []
        start_time = time.time()
        timeout_seconds = BATCH_TIMEOUT_MS / 1000.0
        
        async with self.queue_condition:
            # Wait for at least one request
            while not self.priority_queue:
                try:
                    await asyncio.wait_for(
                        self.queue_condition.wait(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    return []
            
            # Collect requests from priority queue
            while len(batch) < self.current_max_batch_size and self.priority_queue:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds and batch:
                    break
                
                # Get highest priority request
                priority_request = heapq.heappop(self.priority_queue)
                batch.append(priority_request.request)
                
                # For very high priority requests, prefer smaller batches for faster processing
                if priority_request.priority <= RequestPriority.HIGH.value and len(batch) >= self.current_max_batch_size // 2:
                    break
        
        return batch
    
    async def _process_batch(self, batch: List[EmbeddingRequest], model_instance):
        """Process a batch of requests"""
        start_time = time.time()
        
        try:
            # Group by model and task_type for efficiency
            grouped_requests = self._group_requests(batch)
            
            for (model, task_type), requests in grouped_requests.items():
                await self._process_group(requests, model, task_type, model_instance)
            
            processing_time = time.time() - start_time
            self._update_stats(len(batch), processing_time)
            
            logger.debug(f"Processed batch of {len(batch)} requests in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all requests in batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
    
    def _group_requests(self, batch: List[EmbeddingRequest]) -> Dict[Tuple[str, str], List[EmbeddingRequest]]:
        """Group requests by model and task_type"""
        groups = {}
        for request in batch:
            key = (request.model, request.task_type)
            if key not in groups:
                groups[key] = []
            groups[key].append(request)
        return groups
    
    async def _process_group(self, requests: List[EmbeddingRequest], model: str, task_type: str, model_instance):
        """Process a group of requests with same model and task_type"""
        # Flatten all texts from all requests
        all_texts = []
        request_text_ranges = []  # Track which texts belong to which request
        
        for request in requests:
            start_idx = len(all_texts)
            all_texts.extend(request.texts)
            end_idx = len(all_texts)
            request_text_ranges.append((start_idx, end_idx))
        
        # Check cache for all texts (try Redis-enhanced cache)
        cached_embeddings, miss_indices = await embedding_cache.get_batch_async(all_texts, model, task_type)
        
        # Process only cache misses
        if miss_indices:
            miss_texts = [all_texts[i] for i in miss_indices]
            
            # Use model pool if enabled, otherwise use single model instance
            if self.model_pool_enabled and model_pool:
                new_embeddings = await model_pool.process_with_pool(miss_texts, model, task_type)
                # Convert to list format expected by cache
                new_embeddings = [new_embeddings[i] for i in range(len(new_embeddings))]
            else:
                # Run model inference in thread pool to avoid blocking
                new_embeddings = await asyncio.to_thread(
                    self._run_model_inference,
                    model_instance,
                    miss_texts,
                    task_type
                )
            
            # Store new embeddings in cache (including Redis)
            await embedding_cache.put_batch_async(miss_texts, model, task_type, new_embeddings)
            
            # Fill in the missing embeddings
            for miss_idx, embedding in zip(miss_indices, new_embeddings):
                cached_embeddings[miss_idx] = embedding
        
        # Distribute embeddings back to original requests
        for request, (start_idx, end_idx) in zip(requests, request_text_ranges):
            request_embeddings = cached_embeddings[start_idx:end_idx]
            
            if not request.future.done():
                request.future.set_result(request_embeddings)
    
    def _run_model_inference(self, model_instance, texts: List[str], task_type: str) -> List[np.ndarray]:
        """Run model inference synchronously"""
        try:
            embeddings = model_instance.encode(texts, task_type=task_type)
            # Convert to list of individual embeddings
            return [embeddings[i] for i in range(len(embeddings))]
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            raise
    
    def _update_stats(self, batch_size: int, processing_time: float):
        """Update processing statistics"""
        self.stats["total_batches"] += 1
        
        # Update running averages
        total_batches = self.stats["total_batches"]
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * (total_batches - 1) + batch_size) / total_batches
        )
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (total_batches - 1) + processing_time) / total_batches
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            **self.stats,
            "queue_size": len(self.priority_queue),
            "max_batch_size": self.current_max_batch_size,
            "initial_max_batch_size": MAX_BATCH_SIZE,
            "batch_timeout_ms": BATCH_TIMEOUT_MS,
            "rate_limiting": {
                "active_users": len(self.user_rate_limits),
                "rate_limit_per_minute": self.default_rate_limit
            }
        }
    
    def _register_with_memory_manager(self):
        """Register batch resize callback with memory manager"""
        try:
            from .memory_manager import get_memory_manager
            
            memory_manager = get_memory_manager()
            memory_manager.register_batch_resize_callback(self.resize_batch_size)
            logger.info("Batch processor registered with memory manager for dynamic sizing")
            
        except Exception as e:
            logger.warning(f"Failed to register batch processor with memory manager: {e}")
    
    def resize_batch_size(self, new_size: int):
        """Resize the maximum batch size"""
        old_size = self.current_max_batch_size
        self.current_max_batch_size = max(1, new_size)  # Minimum batch size of 1
        
        if old_size != self.current_max_batch_size:
            logger.info(f"Batch size adjusted from {old_size} to {self.current_max_batch_size}")


# Global batch processor instance
batch_processor = SmartBatchProcessor()