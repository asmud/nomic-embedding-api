"""
Test utilities and helper functions.
"""

import asyncio
import contextlib
import functools
import time
import threading
from typing import List, Dict, Any, Callable, Optional
import numpy as np
from unittest.mock import MagicMock, patch


class MockAsyncIterator:
    """Mock async iterator for testing streaming responses."""
    
    def __init__(self, items):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        await asyncio.sleep(0.001)  # Simulate async delay
        return item


class ModelMockFactory:
    """Factory for creating consistent model mocks."""
    
    @staticmethod
    def create_embedding_model(dimensions: int = 768, model_name: str = "test-model"):
        """Create a mock embedding model."""
        model = MagicMock()
        model.model_name = model_name
        model.get_embedding_dimension.return_value = dimensions
        
        def mock_encode(texts, task_type="search_document"):
            if isinstance(texts, str):
                texts = [texts]
            batch_size = len(texts)
            return np.random.random((batch_size, dimensions)).astype(np.float32)
        
        model.encode.side_effect = mock_encode
        return model
    
    @staticmethod
    def create_failing_model(error_type: type = Exception, error_message: str = "Model failed"):
        """Create a model that always fails."""
        model = MagicMock()
        model.encode.side_effect = error_type(error_message)
        model.get_embedding_dimension.return_value = 768
        model.model_name = "failing-model"
        return model
    
    @staticmethod
    def create_slow_model(delay: float = 1.0, dimensions: int = 768):
        """Create a model with artificial delay."""
        model = MagicMock()
        model.model_name = "slow-model"
        model.get_embedding_dimension.return_value = dimensions
        
        def slow_encode(texts, task_type="search_document"):
            time.sleep(delay)
            if isinstance(texts, str):
                texts = [texts]
            batch_size = len(texts)
            return np.random.random((batch_size, dimensions)).astype(np.float32)
        
        model.encode.side_effect = slow_encode
        return model


class CacheMockFactory:
    """Factory for creating cache mocks."""
    
    @staticmethod
    def create_cache_mock(hit_rate: float = 0.5):
        """Create a cache mock with configurable hit rate."""
        cache = MagicMock()
        call_count = [0]
        
        def mock_get(text, model, task_type):
            call_count[0] += 1
            if call_count[0] % int(1/hit_rate) == 0:
                # Cache hit
                return np.random.random((768,)).astype(np.float32)
            return None  # Cache miss
        
        cache.get.side_effect = mock_get
        cache.put = MagicMock()
        cache.clear = MagicMock()
        
        def mock_stats():
            total_calls = call_count[0]
            hits = int(total_calls * hit_rate)
            misses = total_calls - hits
            return {
                'size': 50,
                'max_size': 100,
                'hits': hits,
                'misses': misses,
                'hit_rate': hit_rate,
                'evictions': 0
            }
        
        cache.get_stats.side_effect = mock_stats
        return cache
    
    @staticmethod
    def create_failing_cache():
        """Create a cache that always fails."""
        cache = MagicMock()
        cache.get.side_effect = Exception("Cache failed")
        cache.put.side_effect = Exception("Cache failed")
        cache.clear.side_effect = Exception("Cache failed")
        cache.get_stats.return_value = {'error': 'Cache unavailable'}
        return cache


class RequestMockFactory:
    """Factory for creating request mocks."""
    
    @staticmethod
    def create_embedding_request(
        texts: List[str] = None,
        model: str = "test-model",
        task_type: str = "search_document",
        priority: str = "normal",
        user: str = None,
        stream: bool = False
    ):
        """Create a mock embedding request."""
        if texts is None:
            texts = ["test text"]
        
        return {
            "input": texts,
            "model": model,
            "task_type": task_type,
            "priority": priority,
            "user": user,
            "stream": stream
        }
    
    @staticmethod
    def create_batch_requests(count: int = 10, batch_size: int = 5):
        """Create multiple mock requests for batch testing."""
        requests = []
        for i in range(count):
            texts = [f"Batch {i} text {j}" for j in range(batch_size)]
            request = RequestMockFactory.create_embedding_request(
                texts=texts,
                model=f"model-{i % 3}"  # Vary models
            )
            requests.append(request)
        return requests


class PerformanceTimer:
    """Timer utility for performance testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.lap_times = []
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer and return elapsed time."""
        self.end_time = time.time()
        return self.elapsed
    
    def lap(self):
        """Record a lap time."""
        if self.start_time is None:
            return None
        lap_time = time.time() - self.start_time
        self.lap_times.append(lap_time)
        return lap_time
    
    @property
    def elapsed(self):
        """Get elapsed time."""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class ConcurrencyHelper:
    """Helper for testing concurrent operations."""
    
    @staticmethod
    def run_concurrent_tasks(tasks: List[Callable], max_workers: int = 10):
        """Run tasks concurrently and return results."""
        import concurrent.futures
        
        results = []
        errors = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(e)
        
        return results, errors
    
    @staticmethod
    async def run_concurrent_async_tasks(tasks: List[Callable], semaphore_limit: int = 10):
        """Run async tasks concurrently with semaphore limiting."""
        semaphore = asyncio.Semaphore(semaphore_limit)
        
        async def limited_task(task):
            async with semaphore:
                return await task()
        
        results = await asyncio.gather(
            *[limited_task(task) for task in tasks],
            return_exceptions=True
        )
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        return successful_results, errors


class MemoryMonitor:
    """Monitor memory usage during tests."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
        self.samples = []
    
    def start(self):
        """Start memory monitoring."""
        import psutil
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        return self
    
    def sample(self):
        """Take a memory sample."""
        import psutil
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        self.samples.append(current_memory)
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        return current_memory
    
    def stop(self):
        """Stop monitoring and return summary."""
        import psutil
        process = psutil.Process()
        self.final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'initial_mb': self.initial_memory,
            'final_mb': self.final_memory,
            'peak_mb': self.peak_memory,
            'growth_mb': self.final_memory - self.initial_memory,
            'samples': self.samples
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class ResponseValidator:
    """Validate API responses."""
    
    @staticmethod
    def validate_embedding_response(response_data: Dict[str, Any], expected_count: int = 1):
        """Validate embedding response structure."""
        assert "object" in response_data
        assert "data" in response_data
        assert "model" in response_data
        assert "usage" in response_data
        
        assert response_data["object"] == "list"
        assert len(response_data["data"]) == expected_count
        
        for i, embedding_data in enumerate(response_data["data"]):
            assert embedding_data["object"] == "embedding"
            assert "embedding" in embedding_data
            assert "index" in embedding_data
            assert embedding_data["index"] == i
            assert isinstance(embedding_data["embedding"], list)
            assert len(embedding_data["embedding"]) > 0
        
        usage = response_data["usage"]
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
    
    @staticmethod
    def validate_models_response(response_data: Dict[str, Any]):
        """Validate models list response."""
        assert "object" in response_data
        assert "data" in response_data
        assert response_data["object"] == "list"
        assert isinstance(response_data["data"], list)
        
        for model in response_data["data"]:
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model
            assert model["object"] == "model"
    
    @staticmethod
    def validate_health_response(response_data: Dict[str, Any]):
        """Validate health check response."""
        assert "status" in response_data
        assert "model" in response_data
        assert "embedding_dimension" in response_data
        assert response_data["status"] == "healthy"
        assert isinstance(response_data["embedding_dimension"], int)
        assert response_data["embedding_dimension"] > 0


class ConfigurationHelper:
    """Helper for managing test configurations."""
    
    @staticmethod
    @contextlib.contextmanager
    def temporary_config(**config_overrides):
        """Temporarily override configuration values."""
        import os
        original_values = {}
        
        # Set new values and store originals
        for key, value in config_overrides.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = str(value)
        
        try:
            # Reload config module if available
            try:
                import importlib
                import src.config
                importlib.reload(src.config)
            except ImportError:
                pass
            
            yield
        finally:
            # Restore original values
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value
            
            # Reload config again to restore
            try:
                import importlib
                import src.config
                importlib.reload(src.config)
            except ImportError:
                pass


class LoadTestHelper:
    """Helper for load testing."""
    
    @staticmethod
    def create_load_test_requests(
        num_requests: int,
        texts_per_request: int = 1,
        vary_sizes: bool = True
    ) -> List[Dict[str, Any]]:
        """Create requests for load testing."""
        requests = []
        
        for i in range(num_requests):
            if vary_sizes:
                # Vary the number of texts and their lengths
                num_texts = max(1, (i % 5) + 1)
                text_length = max(5, (i % 20) + 5)
            else:
                num_texts = texts_per_request
                text_length = 10
            
            texts = [
                f"Load test {i} text {j} " + "word " * text_length
                for j in range(num_texts)
            ]
            
            request = {
                "input": texts[0] if len(texts) == 1 else texts,
                "model": "nomic-embed-text-v2-moe-distilled",
                "priority": ["low", "normal", "high"][i % 3],
                "user": f"load_test_user_{i % 10}"
            }
            requests.append(request)
        
        return requests
    
    @staticmethod
    def analyze_load_test_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze load test results."""
        if not results:
            return {"error": "No results to analyze"}
        
        response_times = [r.get("response_time", 0) for r in results if r.get("success")]
        success_count = sum(1 for r in results if r.get("success"))
        error_count = len(results) - success_count
        
        if response_times:
            import statistics
            
            analysis = {
                "total_requests": len(results),
                "successful_requests": success_count,
                "failed_requests": error_count,
                "success_rate": success_count / len(results),
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
            }
            
            if len(response_times) >= 4:
                analysis["p95_response_time"] = statistics.quantiles(response_times, n=20)[18]
                analysis["p99_response_time"] = statistics.quantiles(response_times, n=100)[98]
            
            return analysis
        else:
            return {
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(results),
                "success_rate": 0.0,
                "error": "No successful requests"
            }


class RetryHelper:
    """Helper for retry logic in tests."""
    
    @staticmethod
    def retry_on_failure(
        func: Callable,
        max_attempts: int = 3,
        delay: float = 0.1,
        exceptions: tuple = (Exception,)
    ):
        """Retry a function on failure."""
        for attempt in range(max_attempts):
            try:
                return func()
            except exceptions as e:
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(delay)
        
        return None  # Should not reach here
    
    @staticmethod
    async def async_retry_on_failure(
        func: Callable,
        max_attempts: int = 3,
        delay: float = 0.1,
        exceptions: tuple = (Exception,)
    ):
        """Async version of retry on failure."""
        for attempt in range(max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            except exceptions as e:
                if attempt == max_attempts - 1:
                    raise e
                await asyncio.sleep(delay)
        
        return None  # Should not reach here


def skip_if_no_gpu():
    """Decorator to skip tests if no GPU is available."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import torch
                if not torch.cuda.is_available():
                    import pytest
                    pytest.skip("GPU not available")
                return func(*args, **kwargs)
            except ImportError:
                import pytest
                pytest.skip("PyTorch not available")
        
        return wrapper
    return decorator


def require_model():
    """Decorator to skip tests if model loading is disabled."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import os
            if os.environ.get("SKIP_MODEL_TESTS", "false").lower() == "true":
                import pytest
                pytest.skip("Model tests disabled")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def timeout_after(seconds: float):
    """Decorator to add timeout to test functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Test timed out after {seconds} seconds")
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                return func(*args, **kwargs)
            finally:
                # Clean up
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator