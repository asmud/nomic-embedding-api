"""
Test configuration and shared fixtures for pytest.
"""

import pytest
import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_model():
    """Mock embedding model fixture."""
    model = MagicMock()
    model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    model.get_embedding_dimension.return_value = 768
    model.model_name = "test-model"
    return model


@pytest.fixture
def mock_cache():
    """Mock cache fixture."""
    cache = MagicMock()
    cache.get.return_value = None  # Default to cache miss
    cache.put = MagicMock()
    cache.clear = MagicMock()
    cache.get_stats.return_value = {
        'size': 10,
        'max_size': 100,
        'hits': 5,
        'misses': 15,
        'hit_rate': 0.25
    }
    return cache


@pytest.fixture
def mock_redis_client():
    """Mock Redis client fixture."""
    redis_client = MagicMock()
    redis_client.get.return_value = None
    redis_client.set.return_value = True
    redis_client.delete.return_value = 1
    redis_client.flushdb.return_value = True
    redis_client.ping.return_value = True
    redis_client.info.return_value = {
        'used_memory': 1024,
        'keyspace_hits': 10,
        'keyspace_misses': 5
    }
    return redis_client


@pytest.fixture
def temporary_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "This is a test sentence.",
        "Another test sentence with different content.",
        "A third sentence for testing purposes.",
        "Short text.",
        "This is a much longer sentence that contains significantly more words and should test the system's ability to handle varying text lengths effectively.",
        "Text with special characters: café, naïve, résumé, 世界, 🌍",
        "Numbers and symbols: 123, $456.78, test@example.com, https://example.com",
    ]


@pytest.fixture
def mock_environment_variables():
    """Context manager for mocking environment variables."""
    class MockEnvVars:
        def __init__(self):
            self.original_env = os.environ.copy()
        
        def set(self, **kwargs):
            for key, value in kwargs.items():
                os.environ[key] = str(value)
        
        def restore(self):
            os.environ.clear()
            os.environ.update(self.original_env)
    
    mock_env = MockEnvVars()
    yield mock_env
    mock_env.restore()


@pytest.fixture
def disable_hardware_detection():
    """Disable hardware detection for testing."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.cuda.device_count', return_value=0):
            with patch('os.cpu_count', return_value=4):
                yield


@pytest.fixture
def enable_gpu_mocking():
    """Enable GPU mocking for testing."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=2):
            yield


@pytest.fixture
def mock_model_loading():
    """Mock model loading to avoid downloading actual models."""
    def mock_sentence_transformer(*args, **kwargs):
        model = MagicMock()
        model.encode.return_value = np.random.random((1, 768))
        model.get_sentence_embedding_dimension.return_value = 768
        return model
    
    def mock_static_model(*args, **kwargs):
        model = MagicMock()
        model.encode.return_value = np.random.random((1, 768))
        return model
    
    with patch('sentence_transformers.SentenceTransformer', side_effect=mock_sentence_transformer):
        with patch('model2vec.StaticModel.from_pretrained', side_effect=mock_static_model):
            yield


@pytest.fixture
def mock_psutil():
    """Mock psutil for memory and CPU monitoring."""
    mock = MagicMock()
    
    # Mock virtual_memory
    mock.virtual_memory.return_value = MagicMock(
        total=16 * 1024 * 1024 * 1024,  # 16GB
        available=8 * 1024 * 1024 * 1024,  # 8GB available
        used=8 * 1024 * 1024 * 1024,  # 8GB used
        percent=50.0
    )
    
    # Mock cpu_percent
    mock.cpu_percent.return_value = 25.0
    
    # Mock Process
    process_mock = MagicMock()
    process_mock.memory_info.return_value = MagicMock(
        rss=1024 * 1024 * 1024  # 1GB
    )
    mock.Process.return_value = process_mock
    
    with patch('psutil.virtual_memory', mock.virtual_memory):
        with patch('psutil.cpu_percent', mock.cpu_percent):
            with patch('psutil.Process', mock.Process):
                yield mock


@pytest.fixture
def reset_global_state():
    """Reset global state between tests."""
    # Clear any global caches or state
    import gc
    gc.collect()
    
    yield
    
    # Cleanup after test
    gc.collect()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Automatically set up test environment for all tests."""
    # Set test-specific environment variables
    test_env = {
        'EMBEDDING_MODEL': 'nomic-moe-768',
        'LOG_LEVEL': 'ERROR',  # Reduce log noise in tests
        'ENABLE_CACHING': 'true',
        'CACHE_SIZE': '100',
        'MAX_BATCH_SIZE': '16',
        'BATCH_TIMEOUT_MS': '100',
        'MAX_CONCURRENT_REQUESTS': '10',
        'MODEL_POOL_SIZE': '0',  # Disable model pool for most tests
        'REDIS_ENABLED': 'false',
        'ENABLE_DYNAMIC_MEMORY': 'false',
        'ENABLE_HARDWARE_OPTIMIZATION': 'false',
        'TRUST_REMOTE_CODE': 'true',
        'ENABLE_QUANTIZATION': 'false',
        'TORCH_COMPILE': 'false',
    }
    
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "requires_model: mark test as requiring actual model")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test paths and names
    for item in items:
        # Mark slow tests
        if "performance" in item.nodeid or "load" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.performance)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark API tests
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        
        # Mark unit tests (default for most tests)
        if not any(marker.name in ["integration", "performance", "api"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


@pytest.fixture
def benchmark_timer():
    """Simple benchmark timer for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    import tracemalloc
    
    class MemoryTracker:
        def __init__(self):
            self.start_memory = None
            self.end_memory = None
        
        def start(self):
            tracemalloc.start()
            self.start_memory = tracemalloc.get_traced_memory()
        
        def stop(self):
            self.end_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return self.memory_usage
        
        @property
        def memory_usage(self):
            if self.start_memory is None or self.end_memory is None:
                return None
            return {
                'current': self.end_memory[0],
                'peak': self.end_memory[1],
                'growth': self.end_memory[0] - self.start_memory[0]
            }
    
    return MemoryTracker()


# Test data generators
@pytest.fixture
def generate_random_embeddings():
    """Generate random embeddings for testing."""
    def _generate(num_embeddings, dimensions=768):
        return np.random.random((num_embeddings, dimensions)).astype(np.float32)
    return _generate


@pytest.fixture
def generate_test_texts():
    """Generate test texts of various lengths."""
    def _generate(count=10, min_words=5, max_words=50):
        import random
        texts = []
        for i in range(count):
            num_words = random.randint(min_words, max_words)
            words = [f"word{j}" for j in range(num_words)]
            texts.append(" ".join(words))
        return texts
    return _generate


# Database and cache cleanup
@pytest.fixture(autouse=True)
def cleanup_caches():
    """Clean up caches between tests."""
    yield
    
    # Clear any global caches
    try:
        from src.cache import embedding_cache
        if hasattr(embedding_cache, 'clear'):
            embedding_cache.clear()
    except ImportError:
        pass


# Error injection helpers
@pytest.fixture
def error_injector():
    """Helper for injecting errors in tests."""
    class ErrorInjector:
        def __init__(self):
            self.patches = []
        
        def inject_model_error(self, error_type=Exception, message="Injected error"):
            def failing_encode(*args, **kwargs):
                raise error_type(message)
            
            mock = MagicMock()
            mock.encode.side_effect = failing_encode
            return mock
        
        def inject_network_error(self):
            import requests
            def failing_request(*args, **kwargs):
                raise requests.ConnectionError("Network error")
            return failing_request
        
        def cleanup(self):
            for p in self.patches:
                p.stop()
            self.patches.clear()
    
    injector = ErrorInjector()
    yield injector
    injector.cleanup()


# Skip markers for missing dependencies
def pytest_runtest_setup(item):
    """Setup for individual tests."""
    # Skip tests that require optional dependencies
    if item.get_closest_marker("requires_gpu"):
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")
    
    if item.get_closest_marker("requires_model"):
        # Skip if we can't load actual models (e.g., in CI)
        if os.environ.get("SKIP_MODEL_TESTS", "false").lower() == "true":
            pytest.skip("Model tests disabled")


# Parallel test configuration
def pytest_xdist_worker_id(worker_id):
    """Configure worker-specific settings for parallel testing."""
    if worker_id is not None:
        # Set worker-specific ports and IDs to avoid conflicts
        worker_num = int(worker_id.replace("gw", ""))
        os.environ["TEST_WORKER_ID"] = str(worker_num)
        os.environ["TEST_PORT_OFFSET"] = str(worker_num * 10)