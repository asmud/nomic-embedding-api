import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import torch
import gc

from .models import EmbeddingModel
from .config import CURRENT_MODEL_CONFIG

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ModelInstance:
    """Individual model instance in the pool"""
    id: str
    model: Optional[EmbeddingModel]
    device: str
    status: ModelStatus
    load_time: float
    last_used: float
    total_requests: int
    total_processing_time: float
    current_batch_size: int
    error_count: int
    memory_usage_mb: float


class ModelPool:
    """Pool of model instances for scalable inference"""
    
    def __init__(self, pool_size: int = None, enable_multi_gpu: bool = True):
        # Determine optimal pool size
        if pool_size is None:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                # 1-2 instances per GPU depending on memory
                pool_size = max(1, min(gpu_count * 2, 8))
            else:
                # CPU only - fewer instances
                pool_size = 2
        
        self.pool_size = pool_size
        self.enable_multi_gpu = enable_multi_gpu
        self.instances: List[ModelInstance] = []
        self.pool_lock = asyncio.Lock()
        self.load_balancer_index = 0
        
        # Pool statistics
        self.stats = {
            "total_requests": 0,
            "total_processing_time": 0,
            "avg_queue_time": 0,
            "active_instances": 0,
            "failed_instances": 0
        }
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.max_error_count = 5
        
        logger.info(f"Initializing model pool with {pool_size} instances, multi-GPU: {enable_multi_gpu}")
    
    async def initialize(self) -> None:
        """Initialize all model instances in the pool"""
        try:
            tasks = []
            for i in range(self.pool_size):
                device = self._get_device_for_instance(i)
                task = asyncio.create_task(self._create_model_instance(i, device))
                tasks.append(task)
            
            # Wait for all instances to load
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to load model instance {i}: {result}")
                else:
                    self.instances.append(result)
            
            active_count = len([inst for inst in self.instances if inst.status == ModelStatus.READY])
            logger.info(f"Model pool initialized: {active_count}/{self.pool_size} instances ready")
            
            if active_count == 0:
                raise RuntimeError("No model instances could be loaded")
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
        except Exception as e:
            logger.error(f"Model pool initialization failed: {e}")
            raise
    
    def _get_device_for_instance(self, instance_id: int) -> str:
        """Get the optimal device for a model instance"""
        if not torch.cuda.is_available() or not self.enable_multi_gpu:
            return "cpu"
        
        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            return "cuda:0"
        
        # Distribute instances across GPUs
        gpu_id = instance_id % gpu_count
        return f"cuda:{gpu_id}"
    
    async def _create_model_instance(self, instance_id: str, device: str) -> ModelInstance:
        """Create a single model instance"""
        start_time = time.time()
        
        try:
            logger.info(f"Loading model instance {instance_id} on device {device}")
            
            # Create model in thread to avoid blocking
            model = await asyncio.to_thread(self._load_model, device)
            
            load_time = time.time() - start_time
            memory_usage = self._get_memory_usage(device)
            
            instance = ModelInstance(
                id=f"model_{instance_id}",
                model=model,
                device=device,
                status=ModelStatus.READY,
                load_time=load_time,
                last_used=time.time(),
                total_requests=0,
                total_processing_time=0,
                current_batch_size=0,
                error_count=0,
                memory_usage_mb=memory_usage
            )
            
            logger.info(f"Model instance {instance_id} loaded in {load_time:.2f}s on {device}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create model instance {instance_id}: {e}")
            return ModelInstance(
                id=f"model_{instance_id}",
                model=None,
                device=device,
                status=ModelStatus.ERROR,
                load_time=time.time() - start_time,
                last_used=0,
                total_requests=0,
                total_processing_time=0,
                current_batch_size=0,
                error_count=1,
                memory_usage_mb=0
            )
    
    def _load_model(self, device: str) -> EmbeddingModel:
        """Load model synchronously on specified device"""
        # Temporarily set CUDA device if using GPU
        old_device = None
        if device.startswith("cuda"):
            old_device = torch.cuda.current_device()
            torch.cuda.set_device(device)
        
        try:
            model = EmbeddingModel()
            
            # Move model to specified device if needed
            if hasattr(model.model, 'to') and device != "cpu":
                model.model = model.model.to(device)
            
            return model
            
        finally:
            # Restore original device
            if old_device is not None:
                torch.cuda.set_device(old_device)
    
    def _get_memory_usage(self, device: str) -> float:
        """Get memory usage for a device in MB"""
        try:
            if device.startswith("cuda"):
                gpu_id = int(device.split(":")[1])
                memory_used = torch.cuda.memory_allocated(gpu_id) / (1024 ** 2)
                return memory_used
            else:
                # CPU memory is harder to track per model
                return 0.0
        except Exception:
            return 0.0
    
    async def get_available_instance(self) -> Optional[ModelInstance]:
        """Get an available model instance using load balancing"""
        async with self.pool_lock:
            ready_instances = [
                inst for inst in self.instances 
                if inst.status == ModelStatus.READY
            ]
            
            if not ready_instances:
                logger.warning("No ready model instances available")
                return None
            
            # Round-robin load balancing
            instance = ready_instances[self.load_balancer_index % len(ready_instances)]
            self.load_balancer_index += 1
            
            # Mark as busy
            instance.status = ModelStatus.BUSY
            instance.last_used = time.time()
            
            return instance
    
    async def release_instance(self, instance: ModelInstance, processing_time: float, success: bool) -> None:
        """Release a model instance back to the pool"""
        async with self.pool_lock:
            if success:
                instance.status = ModelStatus.READY
                instance.total_requests += 1
                instance.total_processing_time += processing_time
                instance.error_count = max(0, instance.error_count - 1)  # Reduce error count on success
            else:
                instance.error_count += 1
                if instance.error_count >= self.max_error_count:
                    logger.error(f"Instance {instance.id} exceeded max errors, marking as failed")
                    instance.status = ModelStatus.ERROR
                else:
                    instance.status = ModelStatus.READY
            
            # Update memory usage
            instance.memory_usage_mb = self._get_memory_usage(instance.device)
    
    async def process_with_pool(self, texts: List[str], model: str, task_type: str) -> List:
        """Process texts using the model pool"""
        start_time = time.time()
        
        # Get available instance
        instance = await self.get_available_instance()
        if not instance:
            raise RuntimeError("No model instances available")
        
        queue_time = time.time() - start_time
        
        try:
            # Process with the instance
            processing_start = time.time()
            embeddings = await asyncio.to_thread(
                instance.model.encode, texts, task_type
            )
            processing_time = time.time() - processing_start
            
            # Update statistics
            self.stats["total_requests"] += 1
            self.stats["total_processing_time"] += processing_time
            self.stats["avg_queue_time"] = (
                (self.stats["avg_queue_time"] * (self.stats["total_requests"] - 1) + queue_time) 
                / self.stats["total_requests"]
            )
            
            await self.release_instance(instance, processing_time, True)
            return embeddings
            
        except Exception as e:
            logger.error(f"Processing failed on instance {instance.id}: {e}")
            await self.release_instance(instance, 0, False)
            raise
    
    async def _health_monitor(self) -> None:
        """Monitor health of model instances and restart failed ones"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                async with self.pool_lock:
                    failed_instances = [
                        inst for inst in self.instances 
                        if inst.status == ModelStatus.ERROR
                    ]
                    
                    # Attempt to restart failed instances
                    for failed_instance in failed_instances:
                        if failed_instance.error_count >= self.max_error_count:
                            logger.info(f"Attempting to restart failed instance {failed_instance.id}")
                            
                            # Try to reload the model
                            try:
                                new_model = await asyncio.to_thread(
                                    self._load_model, failed_instance.device
                                )
                                failed_instance.model = new_model
                                failed_instance.status = ModelStatus.READY
                                failed_instance.error_count = 0
                                logger.info(f"Successfully restarted instance {failed_instance.id}")
                            except Exception as e:
                                logger.error(f"Failed to restart instance {failed_instance.id}: {e}")
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        ready_count = len([inst for inst in self.instances if inst.status == ModelStatus.READY])
        busy_count = len([inst for inst in self.instances if inst.status == ModelStatus.BUSY])
        error_count = len([inst for inst in self.instances if inst.status == ModelStatus.ERROR])
        
        total_memory = sum(inst.memory_usage_mb for inst in self.instances)
        avg_processing_time = (
            self.stats["total_processing_time"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0 else 0
        )
        
        return {
            "pool_size": self.pool_size,
            "instances": {
                "ready": ready_count,
                "busy": busy_count,
                "error": error_count,
                "total": len(self.instances)
            },
            "statistics": {
                **self.stats,
                "avg_processing_time": round(avg_processing_time, 4),
                "total_memory_usage_mb": round(total_memory, 2)
            },
            "devices": list(set(inst.device for inst in self.instances)),
            "instance_details": [
                {
                    "id": inst.id,
                    "device": inst.device,
                    "status": inst.status.value,
                    "requests": inst.total_requests,
                    "avg_time": (
                        round(inst.total_processing_time / inst.total_requests, 4)
                        if inst.total_requests > 0 else 0
                    ),
                    "memory_mb": round(inst.memory_usage_mb, 2),
                    "error_count": inst.error_count
                }
                for inst in self.instances
            ]
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the model pool"""
        logger.info("Shutting down model pool...")
        
        async with self.pool_lock:
            for instance in self.instances:
                instance.status = ModelStatus.SHUTDOWN
                if instance.model and hasattr(instance.model, 'model'):
                    # Clear model from memory
                    del instance.model.model
                    del instance.model
                
                # Clear GPU memory if applicable
                if instance.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            
            self.instances.clear()
            gc.collect()
        
        logger.info("Model pool shutdown complete")


# Global model pool instance
model_pool: Optional[ModelPool] = None