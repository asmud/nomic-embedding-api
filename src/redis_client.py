import logging
import json
import hashlib
import time
from typing import Optional, List, Dict, Any, Union
import asyncio

from .config import (
    REDIS_ENABLED,
    REDIS_URL,
    REDIS_DB,
    REDIS_KEY_PREFIX,
    REDIS_MAX_CONNECTIONS,
    REDIS_RETRY_ON_TIMEOUT,
    REDIS_CACHE_TTL,
    REDIS_SESSION_TTL
)

logger = logging.getLogger(__name__)

# Global Redis client
redis_client: Optional['RedisClient'] = None

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
    logger.info("Redis package found and imported successfully")
except ImportError as e:
    REDIS_AVAILABLE = False
    logger.error(f"Redis package not available: {e}")
    if REDIS_ENABLED:
        logger.warning("Redis requested but redis package not installed. Install with: pip install redis[hiredis]==5.2.1")


class RedisClient:
    """Async Redis client for caching and session management"""
    
    def __init__(self):
        self.pool = None
        self.client = None
        self.connected = False
        
    async def connect(self):
        """Initialize Redis connection"""
        if not REDIS_AVAILABLE:
            logger.error("Redis package not available - cannot connect")
            return False
            
        if not REDIS_ENABLED:
            logger.info("Redis disabled in configuration")
            return False
            
        try:
            logger.info(f"Attempting to connect to Redis at {REDIS_URL}, DB: {REDIS_DB}")
            
            # Create connection pool
            self.pool = redis.ConnectionPool.from_url(
                REDIS_URL,
                db=REDIS_DB,
                max_connections=REDIS_MAX_CONNECTIONS,
                retry_on_timeout=REDIS_RETRY_ON_TIMEOUT,
                decode_responses=True
            )
            
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            logger.info("Testing Redis connection with ping...")
            await self.client.ping()
            self.connected = True
            logger.info(f"Redis connected successfully to {REDIS_URL}, DB: {REDIS_DB}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {REDIS_URL}: {e}")
            logger.error(f"Redis connection error details: {type(e).__name__}: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close Redis connections"""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        self.connected = False
        logger.info("Redis disconnected")
    
    def _make_key(self, key: str, namespace: str = "cache") -> str:
        """Create prefixed Redis key"""
        return f"{REDIS_KEY_PREFIX}:{namespace}:{key}"
    
    def _hash_texts(self, texts: List[str], model: str, task_type: str) -> str:
        """Create hash key for embedding cache"""
        content = json.dumps({
            "texts": texts,
            "model": model, 
            "task_type": task_type
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_embeddings(self, texts: List[str], model: str, task_type: str) -> Optional[List[List[float]]]:
        """Get cached embeddings"""
        if not self.connected:
            return None
            
        try:
            cache_key = self._make_key(self._hash_texts(texts, model, task_type))
            cached_data = await self.client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                logger.debug(f"Redis cache hit for {len(texts)} texts")
                return data["embeddings"]
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    async def set_embeddings(self, texts: List[str], model: str, task_type: str, embeddings: List[List[float]]):
        """Cache embeddings with TTL"""
        if not self.connected:
            return
            
        try:
            cache_key = self._make_key(self._hash_texts(texts, model, task_type))
            cache_data = {
                "embeddings": embeddings,
                "model": model,
                "task_type": task_type,
                "timestamp": time.time(),
                "count": len(texts)
            }
            
            await self.client.setex(
                cache_key,
                REDIS_CACHE_TTL,
                json.dumps(cache_data, default=str)
            )
            logger.debug(f"Redis cached {len(texts)} embeddings")
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def get_user_rate_limit(self, user_id: str) -> List[float]:
        """Get user's request timestamps for rate limiting"""
        if not self.connected:
            return []
            
        try:
            rate_key = self._make_key(user_id, "rate_limit")
            timestamps_json = await self.client.get(rate_key)
            
            if timestamps_json:
                return json.loads(timestamps_json)
                
        except Exception as e:
            logger.error(f"Redis rate limit get error: {e}")
        
        return []
    
    async def set_user_rate_limit(self, user_id: str, timestamps: List[float]):
        """Update user's request timestamps for rate limiting"""
        if not self.connected:
            return
            
        try:
            rate_key = self._make_key(user_id, "rate_limit")
            await self.client.setex(
                rate_key,
                REDIS_SESSION_TTL,
                json.dumps(timestamps)
            )
            
        except Exception as e:
            logger.error(f"Redis rate limit set error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        if not self.connected:
            return {"connected": False, "redis_enabled": REDIS_ENABLED}
            
        try:
            info = await self.client.info()
            
            # Get cache statistics
            cache_pattern = f"{REDIS_KEY_PREFIX}:cache:*"
            cache_keys = await self.client.keys(cache_pattern)
            
            rate_limit_pattern = f"{REDIS_KEY_PREFIX}:rate_limit:*"
            rate_limit_keys = await self.client.keys(rate_limit_pattern)
            
            return {
                "connected": True,
                "redis_enabled": REDIS_ENABLED,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "cache_keys": len(cache_keys),
                "rate_limit_sessions": len(rate_limit_keys),
                "cache_ttl": REDIS_CACHE_TTL,
                "session_ttl": REDIS_SESSION_TTL,
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"connected": False, "error": str(e)}
    
    async def clear_cache(self, pattern: str = None):
        """Clear cache entries"""
        if not self.connected:
            return 0
            
        try:
            if pattern:
                cache_pattern = f"{REDIS_KEY_PREFIX}:{pattern}"
            else:
                cache_pattern = f"{REDIS_KEY_PREFIX}:cache:*"
                
            keys = await self.client.keys(cache_pattern)
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Redis cleared {deleted} cache entries")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Redis clear cache error: {e}")
            return 0


async def get_redis_client() -> Optional[RedisClient]:
    """Get or create Redis client instance"""
    global redis_client
    
    if not REDIS_ENABLED or not REDIS_AVAILABLE:
        return None
        
    if redis_client is None:
        redis_client = RedisClient()
        await redis_client.connect()
    
    return redis_client if redis_client.connected else None


async def close_redis_client():
    """Close Redis client connection"""
    global redis_client
    
    if redis_client:
        await redis_client.disconnect()
        redis_client = None