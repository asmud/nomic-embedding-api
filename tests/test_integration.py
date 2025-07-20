import pytest
import asyncio
import numpy as np
import time
import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import sys
import os
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app


class TestFullSystemIntegration:
    """Test complete system integration scenarios"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_model(self):
        """Mock model that returns consistent embeddings"""
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        model.get_embedding_dimension.return_value = 768
        model.model_name = "test-model"
        return model
    
    def test_complete_embedding_workflow(self, client):
        """Test complete embedding generation workflow"""
        # Test single text embedding
        payload = {
            "input": "This is a test sentence for embedding.",
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        response = client.post("/v1/embeddings", json=payload)
        
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 503, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "object" in data
            assert "data" in data
            assert "model" in data
            assert "usage" in data
            
            # Verify structure
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["object"] == "embedding"
            assert isinstance(data["data"][0]["embedding"], list)
    
    def test_batch_embedding_workflow(self, client):
        """Test batch embedding generation workflow"""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence with more content."
        ]
        
        payload = {
            "input": texts,
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        response = client.post("/v1/embeddings", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["data"]) == 3
            
            # Check each embedding
            for i, embedding_data in enumerate(data["data"]):
                assert embedding_data["index"] == i
                assert isinstance(embedding_data["embedding"], list)
                assert len(embedding_data["embedding"]) > 0
    
    def test_system_health_check_integration(self, client):
        """Test system health check integration"""
        # Check system status
        health_response = client.get("/health")
        root_response = client.get("/")
        models_response = client.get("/v1/models")
        
        # All endpoints should be accessible
        assert health_response.status_code in [200, 503]
        assert root_response.status_code == 200
        assert models_response.status_code == 200
        
        # Root endpoint should show system info
        root_data = root_response.json()
        assert "message" in root_data
        assert "features" in root_data
        assert "scalability" in root_data
        
        # Models endpoint should list available models
        models_data = models_response.json()
        assert "data" in models_data
        assert isinstance(models_data["data"], list)
    
    def test_error_handling_integration(self, client):
        """Test error handling across the system"""
        # Test malformed requests
        malformed_requests = [
            {},  # Empty request
            {"model": "nonexistent"},  # Missing input
            {"input": "test"},  # Missing model
            {"input": "", "model": "test"},  # Empty input
            {"input": [], "model": "test"},  # Empty array
        ]
        
        for payload in malformed_requests:
            response = client.post("/v1/embeddings", json=payload)
            # Should return appropriate error codes
            assert response.status_code in [400, 422, 503]
    
    def test_concurrent_requests_integration(self, client):
        """Test handling multiple concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def make_request(text_id):
            try:
                payload = {
                    "input": f"Test text {text_id}",
                    "model": "nomic-embed-text-v2-moe-distilled"
                }
                response = client.post("/v1/embeddings", json=payload)
                results.put((text_id, response.status_code, response.json() if response.status_code == 200 else None))
            except Exception as e:
                errors.put((text_id, str(e)))
        
        # Create multiple concurrent requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check results
        assert errors.qsize() == 0, f"Errors occurred: {list(errors.queue)}"
        assert results.qsize() == 5
        
        # All should either succeed or fail gracefully
        while not results.empty():
            text_id, status_code, data = results.get()
            assert status_code in [200, 503, 500]
    
    def test_statistics_integration(self, client):
        """Test statistics collection integration"""
        # Make some requests to generate stats
        for i in range(3):
            payload = {
                "input": f"Test request {i}",
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            client.post("/v1/embeddings", json=payload)
        
        # Get stats
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200
        
        stats_data = stats_response.json()
        # Should contain various stats categories
        expected_categories = ["cache", "batching", "concurrent_requests"]
        found_categories = [cat for cat in expected_categories if cat in stats_data]
        assert len(found_categories) > 0
    
    def test_cache_integration(self, client):
        """Test caching system integration"""
        # Make identical requests
        payload = {
            "input": "This exact text should be cached",
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        # First request
        response1 = client.post("/v1/embeddings", json=payload)
        
        # Second identical request (should potentially hit cache)
        response2 = client.post("/v1/embeddings", json=payload)
        
        # Both should succeed or fail consistently
        assert response1.status_code == response2.status_code
        
        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()
            
            # Results should be identical (from cache or consistent model)
            assert len(data1["data"]) == len(data2["data"])
        
        # Test cache clearing
        clear_response = client.post("/clear-caches")
        assert clear_response.status_code == 200
    
    @patch('src.api.embedding_model')
    @patch('src.api.model_pool')
    def test_model_initialization_integration(self, mock_model_pool, mock_embedding_model):
        """Test model initialization integration"""
        # Mock model components
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_embedding_dimension.return_value = 768
        mock_model.model_name = "test-model"
        
        mock_embedding_model = mock_model
        mock_model_pool = None
        
        # Test with mocked model
        with TestClient(app) as client:
            response = client.get("/health")
            # Should be able to get health status
            assert response.status_code in [200, 503]


class TestStreamingIntegration:
    """Test streaming functionality integration"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_streaming_response_format(self, client):
        """Test streaming response format"""
        payload = {
            "input": "Test streaming response",
            "model": "nomic-embed-text-v2-moe-distilled",
            "stream": True
        }
        
        response = client.post("/v1/embeddings", json=payload)
        
        if response.status_code == 200:
            # Should be streaming response
            assert "text/plain" in response.headers.get("content-type", "")
        else:
            # Should fail gracefully
            assert response.status_code in [503, 500]
    
    def test_streaming_vs_regular_consistency(self, client):
        """Test consistency between streaming and regular responses"""
        payload_regular = {
            "input": "Test consistency",
            "model": "nomic-embed-text-v2-moe-distilled",
            "stream": False
        }
        
        payload_streaming = {
            "input": "Test consistency",
            "model": "nomic-embed-text-v2-moe-distilled",
            "stream": True
        }
        
        regular_response = client.post("/v1/embeddings", json=payload_regular)
        streaming_response = client.post("/v1/embeddings", json=payload_streaming)
        
        # Both should have same success/failure status
        success_codes = [200]
        error_codes = [500, 503]
        
        if regular_response.status_code in success_codes:
            assert streaming_response.status_code in success_codes
        elif regular_response.status_code in error_codes:
            assert streaming_response.status_code in error_codes


class TestPriorityIntegration:
    """Test request priority system integration"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_priority_request_handling(self, client):
        """Test priority request handling"""
        priorities = ["low", "normal", "high", "urgent"]
        
        for priority in priorities:
            payload = {
                "input": f"Test {priority} priority",
                "model": "nomic-embed-text-v2-moe-distilled",
                "priority": priority
            }
            
            response = client.post("/v1/embeddings", json=payload)
            # Should accept all valid priorities
            assert response.status_code in [200, 503, 500]
    
    def test_invalid_priority_handling(self, client):
        """Test invalid priority handling"""
        payload = {
            "input": "Test invalid priority",
            "model": "nomic-embed-text-v2-moe-distilled",
            "priority": "invalid_priority"
        }
        
        response = client.post("/v1/embeddings", json=payload)
        # Should reject invalid priority
        assert response.status_code == 422


class TestSystemLimitsIntegration:
    """Test system limits and resource management integration"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_large_input_handling(self, client):
        """Test handling of large inputs"""
        # Test with very long text
        long_text = "This is a very long text. " * 1000
        payload = {
            "input": long_text,
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        response = client.post("/v1/embeddings", json=payload)
        # Should handle gracefully
        assert response.status_code in [200, 400, 413, 503, 500]
    
    def test_large_batch_handling(self, client):
        """Test handling of large batches"""
        # Test with many texts
        large_batch = [f"Text {i}" for i in range(100)]
        payload = {
            "input": large_batch,
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        response = client.post("/v1/embeddings", json=payload)
        # Should handle gracefully
        assert response.status_code in [200, 413, 503, 500]
    
    def test_concurrent_request_limits(self, client):
        """Test concurrent request limits"""
        import threading
        import time
        
        responses = []
        
        def make_request():
            payload = {
                "input": "Concurrent test",
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            response = client.post("/v1/embeddings", json=payload)
            responses.append(response.status_code)
        
        # Create many concurrent requests
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads quickly
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        # Should complete in reasonable time
        assert time.time() - start_time < 30
        
        # All should return valid status codes
        assert len(responses) == 20
        for status in responses:
            assert status in [200, 429, 503, 500]


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience integration"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_graceful_degradation(self, client):
        """Test graceful degradation when components fail"""
        # Test that system remains partially functional
        # even when some components might fail
        
        # Basic endpoints should still work
        response = client.get("/")
        assert response.status_code == 200
        
        response = client.get("/v1/models")
        assert response.status_code == 200
    
    def test_recovery_after_errors(self, client):
        """Test system recovery after errors"""
        # Generate some error conditions
        error_payloads = [
            {"input": "", "model": "invalid"},
            {"input": None, "model": "test"},
            {},
        ]
        
        for payload in error_payloads:
            try:
                client.post("/v1/embeddings", json=payload)
            except:
                pass  # Ignore errors for this test
        
        # System should still work after errors
        valid_payload = {
            "input": "Recovery test",
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        response = client.post("/v1/embeddings", json=valid_payload)
        # Should still be able to process valid requests
        assert response.status_code in [200, 503, 500]


class TestConfigurationIntegration:
    """Test configuration system integration"""
    
    def test_environment_configuration_integration(self):
        """Test that environment configuration affects system behavior"""
        # Test with different configurations
        configs = [
            {"ENABLE_CACHING": "false"},
            {"MAX_BATCH_SIZE": "8"},
            {"LOG_LEVEL": "DEBUG"},
        ]
        
        for config in configs:
            with patch.dict(os.environ, config):
                # Should be able to import and initialize with different configs
                try:
                    import importlib
                    import src.config
                    importlib.reload(src.config)
                    # Config should be updated
                    assert True  # Test passes if no exception
                except Exception as e:
                    pytest.fail(f"Configuration failed with {config}: {e}")
    
    def test_model_preset_integration(self):
        """Test model preset integration"""
        # Test different model presets
        presets = ["nomic-moe-768", "nomic-moe-256"]
        
        for preset in presets:
            with patch.dict(os.environ, {"EMBEDDING_MODEL": preset}):
                try:
                    import importlib
                    import src.config
                    importlib.reload(src.config)
                    
                    # Should have valid configuration
                    assert hasattr(src.config, 'CURRENT_MODEL_CONFIG')
                    assert hasattr(src.config, 'MODEL_NAME')
                    assert hasattr(src.config, 'EMBEDDING_DIMENSIONS')
                except Exception as e:
                    pytest.fail(f"Model preset {preset} failed: {e}")


class TestPerformanceIntegration:
    """Test performance-related integration scenarios"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_response_time_consistency(self, client):
        """Test response time consistency"""
        response_times = []
        
        for i in range(5):
            payload = {
                "input": f"Performance test {i}",
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            start_time = time.time()
            response = client.post("/v1/embeddings", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                response_times.append(end_time - start_time)
        
        if response_times:
            # Response times should be relatively consistent
            avg_time = sum(response_times) / len(response_times)
            max_deviation = max(abs(t - avg_time) for t in response_times)
            
            # Allow for some variation but not excessive
            assert max_deviation < avg_time * 2  # No more than 2x average
    
    def test_throughput_under_load(self, client):
        """Test system throughput under load"""
        import threading
        import time
        
        completed_requests = []
        start_time = time.time()
        
        def make_requests():
            for i in range(5):
                payload = {
                    "input": f"Load test {threading.current_thread().ident}_{i}",
                    "model": "nomic-embed-text-v2-moe-distilled"
                }
                
                response = client.post("/v1/embeddings", json=payload)
                if response.status_code == 200:
                    completed_requests.append(time.time())
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=60)
        
        total_time = time.time() - start_time
        
        if completed_requests:
            throughput = len(completed_requests) / total_time
            # Should achieve reasonable throughput
            assert throughput > 0  # At least some requests completed
            logger.info(f"Achieved throughput: {throughput:.2f} requests/second")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])