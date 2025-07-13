import pytest
import numpy as np
import threading
import time
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import LRUCache, HybridCache, embedding_cache, get_cache_stats, clear_cache


class TestLRUCache:
    """Test suite for LRUCache class"""
    
    def test_init(self):
        """Test LRUCache initialization"""
        cache = LRUCache(max_size=100)
        assert cache.max_size == 100
        assert cache.initial_max_size == 100
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.evictions == 0
        assert len(cache.cache) == 0
    
    def test_generate_key(self):
        """Test cache key generation"""
        cache = LRUCache(max_size=10)
        
        key1 = cache._generate_key("text", "model", "task")
        key2 = cache._generate_key("text", "model", "task")
        key3 = cache._generate_key("different", "model", "task")
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        assert len(key1) == 32  # MD5 hash length
    
    def test_get_miss(self):
        """Test cache miss"""
        cache = LRUCache(max_size=10)
        
        result = cache.get("text", "model", "task")
        
        assert result is None
        assert cache.misses == 1
        assert cache.hits == 0
    
    def test_put_and_get_hit(self):
        """Test cache put and hit"""
        cache = LRUCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0])
        
        cache.put("text", "model", "task", embedding)
        result = cache.get("text", "model", "task")
        
        assert result is not None
        assert np.array_equal(result, embedding)
        assert cache.hits == 1
        assert cache.misses == 0
        assert len(cache.cache) == 1
    
    def test_put_duplicate_updates(self):
        """Test that putting duplicate key updates value"""
        cache = LRUCache(max_size=10)
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])
        
        cache.put("text", "model", "task", embedding1)
        cache.put("text", "model", "task", embedding2)
        result = cache.get("text", "model", "task")
        
        assert np.array_equal(result, embedding2)
        assert len(cache.cache) == 1
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = LRUCache(max_size=2)
        
        # Fill cache
        cache.put("text1", "model", "task", np.array([1.0]))
        cache.put("text2", "model", "task", np.array([2.0]))
        
        # Access first item to make it most recently used
        cache.get("text1", "model", "task")
        
        # Add third item, should evict text2
        cache.put("text3", "model", "task", np.array([3.0]))
        
        assert cache.get("text1", "model", "task") is not None
        assert cache.get("text2", "model", "task") is None  # Should be evicted
        assert cache.get("text3", "model", "task") is not None
        assert cache.evictions == 1
    
    def test_clear(self):
        """Test cache clearing"""
        cache = LRUCache(max_size=10)
        
        cache.put("text1", "model", "task", np.array([1.0]))
        cache.put("text2", "model", "task", np.array([2.0]))
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.get("text1", "model", "task") is None
        assert cache.get("text2", "model", "task") is None
    
    def test_get_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=2)
        
        # Generate some activity
        cache.put("text1", "model", "task", np.array([1.0]))
        cache.get("text1", "model", "task")  # Hit
        cache.get("text2", "model", "task")  # Miss
        cache.put("text2", "model", "task", np.array([2.0]))
        cache.put("text3", "model", "task", np.array([3.0]))  # Should cause eviction
        
        stats = cache.get_stats()
        
        assert stats["size"] == 2
        assert stats["max_size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["evictions"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_resize(self):
        """Test cache resizing"""
        cache = LRUCache(max_size=2)
        
        # Fill cache
        cache.put("text1", "model", "task", np.array([1.0]))
        cache.put("text2", "model", "task", np.array([2.0]))
        
        # Resize to smaller
        cache.resize(1)
        assert cache.max_size == 1
        assert len(cache.cache) == 1  # Should evict one item
        
        # Resize to larger
        cache.resize(5)
        assert cache.max_size == 5
    
    def test_thread_safety(self):
        """Test thread safety of cache operations"""
        cache = LRUCache(max_size=100)
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(10):
                    key = f"text{thread_id}_{i}"
                    embedding = np.array([float(thread_id), float(i)])
                    
                    cache.put(key, "model", "task", embedding)
                    result = cache.get(key, "model", "task")
                    
                    if result is not None:
                        results.append((thread_id, i, result))
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0  # No thread safety errors
        assert len(results) == 50  # All operations completed
    
    def test_caching_disabled(self):
        """Test cache behavior when caching is disabled"""
        with patch('src.cache.ENABLE_CACHING', False):
            cache = LRUCache(max_size=10)
            embedding = np.array([1.0, 2.0, 3.0])
            
            cache.put("text", "model", "task", embedding)
            result = cache.get("text", "model", "task")
            
            assert result is None  # Should not cache when disabled


class TestHybridCache:
    """Test suite for HybridCache class"""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client fixture"""
        mock_client = AsyncMock()
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_client.flushdb.return_value = True
        mock_client.ping.return_value = True
        return mock_client
    
    @pytest.mark.asyncio
    async def test_init_with_redis_disabled(self):
        """Test HybridCache initialization with Redis disabled"""
        with patch('src.cache.REDIS_ENABLED', False):
            cache = HybridCache(max_size=10)
            await cache.initialize()
            
            assert cache.local_cache is not None
            assert cache.redis_client is None
    
    @pytest.mark.asyncio
    async def test_init_with_redis_enabled(self, mock_redis_client):
        """Test HybridCache initialization with Redis enabled"""
        with patch('src.cache.REDIS_ENABLED', True):
            with patch('src.cache.get_redis_client', return_value=mock_redis_client):
                cache = HybridCache(max_size=10)
                await cache.initialize()
                
                assert cache.local_cache is not None
                assert cache.redis_client is not None
    
    @pytest.mark.asyncio
    async def test_get_local_only(self):
        """Test get from local cache only"""
        with patch('src.cache.REDIS_ENABLED', False):
            cache = HybridCache(max_size=10)
            await cache.initialize()
            
            embedding = np.array([1.0, 2.0, 3.0])
            await cache.put("text", "model", "task", embedding)
            result = await cache.get("text", "model", "task")
            
            assert result is not None
            assert np.array_equal(result, embedding)
    
    @pytest.mark.asyncio
    async def test_get_with_redis_fallback(self, mock_redis_client):
        """Test get with Redis fallback"""
        # Setup Redis to return serialized embedding
        embedding = np.array([1.0, 2.0, 3.0])
        serialized = embedding.tobytes()
        mock_redis_client.get.return_value = serialized
        
        with patch('src.cache.REDIS_ENABLED', True):
            with patch('src.cache.get_redis_client', return_value=mock_redis_client):
                cache = HybridCache(max_size=10)
                await cache.initialize()
                
                result = await cache.get("text", "model", "task")
                
                assert result is not None
                assert np.array_equal(result, embedding)
                mock_redis_client.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_put_local_and_redis(self, mock_redis_client):
        """Test put to both local cache and Redis"""
        with patch('src.cache.REDIS_ENABLED', True):
            with patch('src.cache.get_redis_client', return_value=mock_redis_client):
                cache = HybridCache(max_size=10)
                await cache.initialize()
                
                embedding = np.array([1.0, 2.0, 3.0])
                await cache.put("text", "model", "task", embedding)
                
                # Check local cache
                local_result = cache.local_cache.get("text", "model", "task")
                assert local_result is not None
                
                # Check Redis was called
                mock_redis_client.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_both_caches(self, mock_redis_client):
        """Test clearing both local and Redis caches"""
        with patch('src.cache.REDIS_ENABLED', True):
            with patch('src.cache.get_redis_client', return_value=mock_redis_client):
                cache = HybridCache(max_size=10)
                await cache.initialize()
                
                # Add some data
                embedding = np.array([1.0, 2.0, 3.0])
                await cache.put("text", "model", "task", embedding)
                
                # Clear caches
                await cache.clear(clear_redis=True)
                
                # Check local cache is empty
                assert len(cache.local_cache.cache) == 0
                
                # Check Redis clear was called
                mock_redis_client.flushdb.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_stats_combined(self, mock_redis_client):
        """Test getting combined statistics"""
        mock_redis_client.info.return_value = {
            'used_memory': 1024,
            'keyspace_hits': 10,
            'keyspace_misses': 5
        }
        
        with patch('src.cache.REDIS_ENABLED', True):
            with patch('src.cache.get_redis_client', return_value=mock_redis_client):
                cache = HybridCache(max_size=10)
                await cache.initialize()
                
                stats = await cache.get_stats()
                
                assert "local" in stats
                assert "redis" in stats
                assert "combined_hit_rate" in stats
    
    @pytest.mark.asyncio
    async def test_redis_error_handling(self):
        """Test Redis error handling"""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Redis connection failed")
        
        with patch('src.cache.REDIS_ENABLED', True):
            with patch('src.cache.get_redis_client', return_value=mock_client):
                cache = HybridCache(max_size=10)
                await cache.initialize()
                
                # Should fall back to local cache only
                embedding = np.array([1.0, 2.0, 3.0])
                await cache.put("text", "model", "task", embedding)
                
                # Local cache should still work
                local_result = cache.local_cache.get("text", "model", "task")
                assert local_result is not None


class TestCacheIntegration:
    """Test cache integration and global functions"""
    
    def test_global_embedding_cache_instance(self):
        """Test global embedding_cache instance"""
        assert embedding_cache is not None
        assert hasattr(embedding_cache, 'get')
        assert hasattr(embedding_cache, 'put')
        assert hasattr(embedding_cache, 'clear')
    
    @pytest.mark.asyncio
    async def test_get_cache_stats_function(self):
        """Test get_cache_stats global function"""
        stats = await get_cache_stats()
        
        assert isinstance(stats, dict)
        # Should contain at least local cache stats
        assert "local" in stats or "size" in stats
    
    @pytest.mark.asyncio
    async def test_clear_cache_function(self):
        """Test clear_cache global function"""
        # Add some data to cache
        if hasattr(embedding_cache, 'put'):
            embedding = np.array([1.0, 2.0, 3.0])
            await embedding_cache.put("test", "model", "task", embedding)
        
        # Clear cache
        await clear_cache(clear_redis=False)
        
        # Cache should be empty (if it supports checking)
        if hasattr(embedding_cache, 'local_cache'):
            assert len(embedding_cache.local_cache.cache) == 0
    
    def test_cache_with_different_task_types(self):
        """Test cache behavior with different task types"""
        cache = LRUCache(max_size=10)
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])
        
        cache.put("same text", "model", "search_document", embedding1)
        cache.put("same text", "model", "search_query", embedding2)
        
        result1 = cache.get("same text", "model", "search_document")
        result2 = cache.get("same text", "model", "search_query")
        
        assert np.array_equal(result1, embedding1)
        assert np.array_equal(result2, embedding2)
        assert len(cache.cache) == 2  # Different keys
    
    def test_cache_with_different_models(self):
        """Test cache behavior with different models"""
        cache = LRUCache(max_size=10)
        embedding1 = np.array([1.0, 2.0, 3.0])
        embedding2 = np.array([4.0, 5.0, 6.0])
        
        cache.put("same text", "model1", "task", embedding1)
        cache.put("same text", "model2", "task", embedding2)
        
        result1 = cache.get("same text", "model1", "task")
        result2 = cache.get("same text", "model2", "task")
        
        assert np.array_equal(result1, embedding1)
        assert np.array_equal(result2, embedding2)
        assert len(cache.cache) == 2  # Different keys


class TestCachePerformance:
    """Test cache performance characteristics"""
    
    def test_large_cache_operations(self):
        """Test cache with large number of operations"""
        cache = LRUCache(max_size=1000)
        
        start_time = time.time()
        
        # Add 1000 items
        for i in range(1000):
            embedding = np.array([float(i)] * 10)
            cache.put(f"text{i}", "model", "task", embedding)
        
        # Access all items
        for i in range(1000):
            result = cache.get(f"text{i}", "model", "task")
            assert result is not None
        
        end_time = time.time()
        operation_time = end_time - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert operation_time < 1.0  # Less than 1 second for 2000 operations
        assert cache.hits == 1000
        assert cache.misses == 0
    
    def test_memory_efficiency(self):
        """Test cache memory usage with large embeddings"""
        cache = LRUCache(max_size=10)
        
        # Add large embeddings
        for i in range(10):
            large_embedding = np.random.random(1000)  # 1000-dimensional
            cache.put(f"text{i}", "model", "task", large_embedding)
        
        assert len(cache.cache) == 10
        
        # Access all items to ensure they're still valid
        for i in range(10):
            result = cache.get(f"text{i}", "model", "task")
            assert result is not None
            assert result.shape == (1000,)


class TestCacheEdgeCases:
    """Test cache edge cases and error conditions"""
    
    def test_zero_size_cache(self):
        """Test cache with zero size"""
        cache = LRUCache(max_size=0)
        embedding = np.array([1.0, 2.0, 3.0])
        
        cache.put("text", "model", "task", embedding)
        result = cache.get("text", "model", "task")
        
        assert result is None  # Should not store anything
        assert len(cache.cache) == 0
    
    def test_negative_size_cache(self):
        """Test cache with negative size (should handle gracefully)"""
        cache = LRUCache(max_size=-1)
        # Should not crash, may treat as 0 or handle differently
        assert cache.max_size >= 0  # Implementation should normalize
    
    def test_empty_text_caching(self):
        """Test caching with empty text"""
        cache = LRUCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0])
        
        cache.put("", "model", "task", embedding)
        result = cache.get("", "model", "task")
        
        assert np.array_equal(result, embedding)
    
    def test_unicode_text_caching(self):
        """Test caching with Unicode text"""
        cache = LRUCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0])
        unicode_text = "Hello 世界! 🌍 ñáéíóú"
        
        cache.put(unicode_text, "model", "task", embedding)
        result = cache.get(unicode_text, "model", "task")
        
        assert np.array_equal(result, embedding)
    
    def test_very_long_text_caching(self):
        """Test caching with very long text"""
        cache = LRUCache(max_size=10)
        embedding = np.array([1.0, 2.0, 3.0])
        long_text = "Very long text " * 1000  # Very long string
        
        cache.put(long_text, "model", "task", embedding)
        result = cache.get(long_text, "model", "task")
        
        assert np.array_equal(result, embedding)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])