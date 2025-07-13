import logging
import time
import asyncio
import uuid
from typing import List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import json

from .config import LOG_LEVEL, MAX_CONCURRENT_REQUESTS, MODEL_POOL_SIZE, ENABLE_MULTI_GPU, REDIS_ENABLED, ENABLE_DYNAMIC_MEMORY, ENABLE_HARDWARE_OPTIMIZATION
from .models import EmbeddingModel
from .batch_processor import batch_processor, RequestPriority
from .cache import get_cache_stats, clear_cache
from .model_pool import ModelPool, model_pool
from .schemas import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    EmbeddingData, 
    EmbeddingUsage,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    StreamingEmbeddingChunk,
    StreamingEmbeddingResponse
)

# Configure logging from environment
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global model instance
embedding_model = None

# Global semaphore for concurrent request limiting
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, model_pool
    
    from .config import EMBEDDING_MODEL, CURRENT_MODEL_CONFIG, MODEL_POOL_SIZE, ENABLE_MULTI_GPU
    
    # Track startup tasks for proper cleanup
    startup_tasks = []
    batch_task = None
    
    try:
        logger.info("Starting embedding service...")
        logger.info(f"Model preset: {EMBEDDING_MODEL}")
        logger.info(f"Model: {CURRENT_MODEL_CONFIG['model_name']}")
        logger.info(f"Dimensions: {CURRENT_MODEL_CONFIG['dimensions']}")
        logger.info(f"Description: {CURRENT_MODEL_CONFIG['description']}")
        logger.info(f"Model pool enabled: {MODEL_POOL_SIZE > 0}, Pool size: {MODEL_POOL_SIZE}, Multi-GPU: {ENABLE_MULTI_GPU}")
        
        # Initialize model pool if enabled
        if MODEL_POOL_SIZE > 0:
            try:
                logger.info("Initializing model pool...")
                from .model_pool import model_pool
                if model_pool is None:
                    model_pool = ModelPool(pool_size=MODEL_POOL_SIZE, enable_multi_gpu=ENABLE_MULTI_GPU)
                    # Update global reference
                    import sys
                    sys.modules[__name__].model_pool = model_pool
                    sys.modules['src.model_pool'].model_pool = model_pool
                
                await model_pool.initialize()
                logger.info("Model pool initialized successfully")
                
                # For compatibility, set embedding_model to None since we use pool
                embedding_model = None
                
            except Exception as e:
                logger.error(f"Failed to initialize model pool: {e}")
                logger.info("Falling back to single model instance...")
                MODEL_POOL_SIZE = 0
        
        # Fallback to single model if pool disabled or failed
        if MODEL_POOL_SIZE == 0:
            try:
                logger.info("Loading single embedding model...")
                embedding_model = EmbeddingModel()
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise  # Re-raise to prevent startup with no models
            
        
        # Start memory monitoring if enabled
        if ENABLE_DYNAMIC_MEMORY:
            try:
                from .memory_manager import start_memory_monitoring
                await start_memory_monitoring()
                logger.info("Memory monitoring started")
            except Exception as e:
                logger.error(f"Failed to start memory monitoring: {e}")
        
        # Start hardware optimization monitoring if enabled
        if ENABLE_HARDWARE_OPTIMIZATION:
            try:
                from .hardware_optimizer import start_hardware_monitoring
                await start_hardware_monitoring()
                logger.info("Hardware optimization monitoring started")
            except Exception as e:
                logger.error(f"Failed to start hardware monitoring: {e}")
        
        # Initialize Redis client if enabled
        if REDIS_ENABLED:
            try:
                from .redis_client import get_redis_client
                redis_client = await get_redis_client()
                if redis_client:
                    logger.info("Redis client connected successfully")
                else:
                    logger.warning("Redis client failed to connect")
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
        
        # Start batch processor
        if embedding_model or model_pool:
            logger.info("Starting batch processor...")
            batch_task = asyncio.create_task(batch_processor.process_batches(embedding_model))
            startup_tasks.append(batch_task)
        
        logger.info("Embedding service startup completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        # Cleanup any partially initialized components
        await _cleanup_resources(batch_task, startup_tasks)
        raise
    
    try:
        yield
    except Exception as e:
        logger.error(f"Error during application lifetime: {e}")
    finally:
        # Graceful shutdown sequence
        logger.info("Initiating graceful shutdown of embedding service...")
        await _cleanup_resources(batch_task, startup_tasks)
        logger.info("Embedding service shutdown completed")


async def _cleanup_resources(batch_task, startup_tasks):
    """Cleanup resources during shutdown with proper error handling and timeouts"""
    shutdown_timeout = 30.0  # Total shutdown timeout
    
    try:
        # Step 1: Cancel and wait for batch processor (highest priority)
        if batch_task and not batch_task.done():
            logger.info("Stopping batch processor...")
            batch_task.cancel()
            try:
                await asyncio.wait_for(batch_task, timeout=5.0)
                logger.info("Batch processor stopped successfully")
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("Batch processor forced shutdown")
            except Exception as e:
                logger.error(f"Error stopping batch processor: {e}")
        
        # Step 2: Shutdown model pool
        try:
            global model_pool
            if model_pool:
                logger.info("Shutting down model pool...")
                await asyncio.wait_for(model_pool.shutdown(), timeout=10.0)
                logger.info("Model pool shutdown completed")
        except asyncio.TimeoutError:
            logger.warning("Model pool shutdown timed out")
        except Exception as e:
            logger.error(f"Error shutting down model pool: {e}")
        
        # Step 3: Stop monitoring services
        monitoring_tasks = []
        
        # Stop memory monitoring
        if ENABLE_DYNAMIC_MEMORY:
            try:
                from .memory_manager import stop_memory_monitoring
                task = asyncio.create_task(stop_memory_monitoring())
                monitoring_tasks.append(("memory monitoring", task))
            except Exception as e:
                logger.error(f"Error initiating memory monitoring stop: {e}")
        
        # Stop hardware monitoring
        if ENABLE_HARDWARE_OPTIMIZATION:
            try:
                from .hardware_optimizer import stop_hardware_monitoring
                task = asyncio.create_task(stop_hardware_monitoring())
                monitoring_tasks.append(("hardware monitoring", task))
            except Exception as e:
                logger.error(f"Error initiating hardware monitoring stop: {e}")
        
        # Wait for monitoring tasks with timeout
        if monitoring_tasks:
            logger.info(f"Stopping {len(monitoring_tasks)} monitoring services...")
            for name, task in monitoring_tasks:
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                    logger.info(f"{name} stopped successfully")
                except asyncio.TimeoutError:
                    logger.warning(f"{name} shutdown timed out")
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")
        
        # Step 4: Close Redis client
        try:
            from .redis_client import close_redis_client
            logger.info("Closing Redis client...")
            await asyncio.wait_for(close_redis_client(), timeout=5.0)
            logger.info("Redis client closed successfully")
        except asyncio.TimeoutError:
            logger.warning("Redis client close timed out")
        except Exception as e:
            logger.error(f"Error closing Redis client: {e}")
        
        # Step 5: Clean up any remaining startup tasks
        remaining_tasks = [task for task in startup_tasks if not task.done()]
        if remaining_tasks:
            logger.info(f"Cancelling {len(remaining_tasks)} remaining startup tasks...")
            for task in remaining_tasks:
                task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining_tasks, return_exceptions=True),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some startup tasks did not complete within timeout")
        
    except Exception as e:
        logger.error(f"Error during resource cleanup: {e}")
    
    logger.info("Resource cleanup completed")


app = FastAPI(
    title="Nomic Embedding API",
    description="OpenAI-compatible embedding API using Nomic models",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def count_tokens(text: str) -> int:
    return len(text.split())


def get_model_instance(model_name: str):
    if embedding_model:
        return embedding_model
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No embedding model available"
        )


async def stream_embeddings(texts: List[str], model: str, task_type: str, request_id: str, priority: RequestPriority = RequestPriority.NORMAL, user_id: str = None):
    """Generate streaming embeddings"""
    try:
        # Process embeddings
        embeddings = await batch_processor.add_request(
            texts=texts,
            model=model,
            task_type=task_type,
            request_id=request_id,
            priority=priority,
            user_id=user_id
        )
        
        total_tokens = sum(count_tokens(text) for text in texts)
        
        # Stream each embedding as it becomes available
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            chunk = StreamingEmbeddingChunk(
                index=i,
                embedding=embedding.tolist(),
                model=model
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            
            # Small delay to simulate streaming
            await asyncio.sleep(0.001)
        
        # Send final response with usage information
        final_response = StreamingEmbeddingResponse(
            model=model,
            usage=EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
        )
        yield f"data: {final_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_response = {
            "object": "error",
            "message": str(e),
            "type": "internal_error"
        }
        yield f"data: {json.dumps(error_response)}\n\n"


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings with async processing, batching, caching, and optional streaming"""
    async with request_semaphore:  # Limit concurrent requests
        try:
            # Validate model availability
            model_instance = get_model_instance(request.model)
            
            if isinstance(request.input, str):
                texts = [request.input]
            else:
                texts = request.input
            
            if not texts:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No input texts provided"
                )
            
            task_type = "search_document"
            if hasattr(request, 'task_type'):
                task_type = request.task_type
            
            # Generate unique request ID for tracing
            request_id = str(uuid.uuid4())
            
            # Convert priority string to enum
            priority_map = {
                "low": RequestPriority.LOW,
                "normal": RequestPriority.NORMAL,
                "high": RequestPriority.HIGH,
                "urgent": RequestPriority.URGENT
            }
            priority = priority_map.get(request.priority, RequestPriority.NORMAL)
            
            # Handle streaming vs non-streaming
            if request.stream:
                return StreamingResponse(
                    stream_embeddings(texts, request.model, task_type, request_id, priority, request.user),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "text/plain; charset=utf-8"
                    }
                )
            
            # Non-streaming response (original behavior)
            start_time = time.time()
            embeddings = await batch_processor.add_request(
                texts=texts,
                model=request.model,
                task_type=task_type,
                request_id=request_id,
                priority=priority,
                user_id=request.user
            )
            processing_time = time.time() - start_time
            
            # Build response
            embedding_data = []
            total_tokens = 0
            
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                tokens = count_tokens(text)
                total_tokens += tokens
                
                embedding_data.append(EmbeddingData(
                    object="embedding",
                    embedding=embedding.tolist(),
                    index=i
                ))
            
            usage = EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
            
            logger.debug(f"Request {request_id} processed in {processing_time:.3f}s")
            
            return EmbeddingResponse(
                object="list",
                data=embedding_data,
                model=request.model,
                usage=usage
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e)
            )


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    models = [
        {
            "id": "nomic-embed-text-v2-moe-distilled",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "nomic-ai"
        }
    ]
    
    
    return ModelsResponse(
        object="list",
        data=models
    )


@app.get("/stats")
async def get_comprehensive_stats():
    """Get comprehensive system statistics including hardware, memory, model pool, and Redis"""
    cache_stats = await get_cache_stats()
    batch_stats = batch_processor.get_stats()
    
    stats = {
        "cache": cache_stats,
        "batching": batch_stats,
        "concurrent_requests": {
            "max_concurrent": MAX_CONCURRENT_REQUESTS,
            "available_slots": request_semaphore._value
        }
    }
    
    # Add model pool stats if enabled
    if model_pool:
        stats["model_pool"] = model_pool.get_pool_stats()
    
    # Add memory management stats if enabled
    if ENABLE_DYNAMIC_MEMORY:
        try:
            from .memory_manager import get_memory_stats
            stats["memory"] = get_memory_stats()
        except Exception as e:
            stats["memory"] = {"error": str(e)}
    
    # Add hardware optimization stats if enabled
    if ENABLE_HARDWARE_OPTIMIZATION:
        try:
            from .hardware_optimizer import get_optimization_report
            stats["hardware"] = get_optimization_report()
        except Exception as e:
            stats["hardware"] = {"error": str(e)}
    
    # Note: Redis stats are already included in cache stats from the hybrid cache
    # No need to duplicate Redis information here
    
    return stats


@app.post("/clear-caches")
async def clear_caches(clear_redis: bool = False):
    """Clear the embedding caches"""
    await clear_cache(clear_redis=clear_redis)
    message = "Local cache cleared"
    if clear_redis:
        message = "Local and Redis caches cleared"
    return {"status": "success", "message": message}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not embedding_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded"
        )
    
    model_instance = embedding_model
    
    return HealthResponse(
        status="healthy",
        model=model_instance.model_name,
        embedding_dimension=model_instance.get_embedding_dimension()
    )


@app.get("/")
async def root():
    features = [
        "Async processing",
        "Smart batching", 
        "In-memory caching",
        "Concurrent request limiting",
        "Performance monitoring",
        "Streaming responses",
        "Request prioritization",
        "Rate limiting"
    ]
    
    if model_pool:
        features.extend([
            "Model pool management",
            "Multi-GPU support",
            "Load balancing",
            "Health monitoring"
        ])
    
    if REDIS_ENABLED:
        features.extend([
            "Distributed caching (Redis)",
            "Persistent rate limiting",
            "Session management"
        ])
    
    if ENABLE_DYNAMIC_MEMORY:
        features.extend([
            "Dynamic memory management",
            "Adaptive cache sizing",
            "Memory pressure monitoring",
            "Garbage collection optimization"
        ])
    
    if ENABLE_HARDWARE_OPTIMIZATION:
        features.extend([
            "Hardware optimization",
            "Auto device selection",
            "CPU/GPU utilization monitoring",
            "Dynamic worker scaling",
            "Workload profiling"
        ])
    
    return {
        "message": "Nomic Embedding API", 
        "version": "0.2.0",
        "docs": "/docs",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "models": "/v1/models", 
            "stats": "/stats",
            "clear_caches": "/clear-caches",
            "health": "/health"
        },
        "features": features,
        "scalability": {
            "model_pool_enabled": model_pool is not None,
            "pool_size": model_pool.pool_size if model_pool else 0,
            "multi_gpu": ENABLE_MULTI_GPU,
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "redis_enabled": REDIS_ENABLED,
            "dynamic_memory_enabled": ENABLE_DYNAMIC_MEMORY,
            "hardware_optimization_enabled": ENABLE_HARDWARE_OPTIMIZATION
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )