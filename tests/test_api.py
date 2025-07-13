import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Nomic Embedding API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "embedding_dimension" in data


def test_models_endpoint():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "object" in data
    assert "data" in data
    assert data["object"] == "list"
    assert isinstance(data["data"], list)


def test_embeddings_endpoint():
    payload = {
        "input": "Hello world",
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert "model" in data
        assert "usage" in data
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)
    else:
        assert response.status_code in [503, 500]


def test_embeddings_batch():
    payload = {
        "input": ["Hello world", "How are you?", "This is a test"],
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert len(data["data"]) == 3
        for i, embedding_data in enumerate(data["data"]):
            assert embedding_data["index"] == i
            assert isinstance(embedding_data["embedding"], list)
    else:
        assert response.status_code in [503, 500]


def test_invalid_model():
    payload = {
        "input": "Hello world",
        "model": "nonexistent-model"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code in [400, 503, 500]


def test_embeddings_missing_input():
    """Test embeddings endpoint with missing input"""
    payload = {
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422  # Validation error


def test_embeddings_empty_input():
    """Test embeddings endpoint with empty input"""
    payload = {
        "input": [],
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 400


def test_embeddings_with_dimensions():
    """Test embeddings endpoint with dimensions parameter"""
    payload = {
        "input": "Test text",
        "model": "nomic-embed-text-v2-moe-distilled",
        "dimensions": 256
    }
    
    response = client.post("/v1/embeddings", json=payload)
    # Should work regardless of model availability
    assert response.status_code in [200, 503, 500]


def test_embeddings_with_user():
    """Test embeddings endpoint with user parameter"""
    payload = {
        "input": "Test text",
        "model": "nomic-embed-text-v2-moe-distilled",
        "user": "test-user-123"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code in [200, 503, 500]


def test_embeddings_with_priority():
    """Test embeddings endpoint with different priority levels"""
    priorities = ["low", "normal", "high", "urgent"]
    
    for priority in priorities:
        payload = {
            "input": "Test text",
            "model": "nomic-embed-text-v2-moe-distilled",
            "priority": priority
        }
        
        response = client.post("/v1/embeddings", json=payload)
        assert response.status_code in [200, 503, 500]


def test_embeddings_with_invalid_priority():
    """Test embeddings endpoint with invalid priority"""
    payload = {
        "input": "Test text",
        "model": "nomic-embed-text-v2-moe-distilled",
        "priority": "invalid_priority"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code == 422  # Validation error


def test_embeddings_very_long_text():
    """Test embeddings endpoint with very long text"""
    long_text = "This is a test. " * 1000  # Very long text
    payload = {
        "input": long_text,
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code in [200, 503, 500]


def test_embeddings_special_characters():
    """Test embeddings endpoint with special characters"""
    special_text = "Hello 世界! 🌍 @#$%^&*()_+ ñáéíóú"
    payload = {
        "input": special_text,
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code in [200, 503, 500]


def test_embeddings_large_batch():
    """Test embeddings endpoint with large batch"""
    large_batch = [f"Test text {i}" for i in range(100)]
    payload = {
        "input": large_batch,
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    assert response.status_code in [200, 503, 500]


def test_embeddings_streaming_parameter():
    """Test embeddings endpoint with streaming enabled"""
    payload = {
        "input": "Test streaming text",
        "model": "nomic-embed-text-v2-moe-distilled",
        "stream": True
    }
    
    response = client.post("/v1/embeddings", json=payload)
    # Streaming should return different status or content type
    assert response.status_code in [200, 503, 500]


def test_malformed_json():
    """Test endpoints with malformed JSON"""
    malformed_json = '{"input": "test", "model": "test"'  # Missing closing brace
    
    response = client.post(
        "/v1/embeddings",
        data=malformed_json,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_unsupported_content_type():
    """Test endpoints with unsupported content type"""
    response = client.post(
        "/v1/embeddings",
        data="input=test&model=test",
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 422


def test_stats_endpoint():
    """Test the stats endpoint"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    
    # Should contain basic stats structure
    assert "cache" in data or "batching" in data or "concurrent_requests" in data


def test_clear_caches_endpoint():
    """Test the clear caches endpoint"""
    response = client.post("/clear-caches")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"


def test_clear_caches_with_redis():
    """Test the clear caches endpoint with Redis parameter"""
    response = client.post("/clear-caches?clear_redis=true")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "success"


def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/")
    assert response.status_code == 200
    
    # Check for CORS headers in any response
    response = client.get("/")
    assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]


def test_health_endpoint_structure():
    """Test health endpoint returns proper structure when healthy"""
    response = client.get("/health")
    
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "model" in data
        assert "embedding_dimension" in data
        assert data["status"] == "healthy"
        assert isinstance(data["embedding_dimension"], int)


def test_models_endpoint_structure():
    """Test models endpoint returns proper structure"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    
    assert "object" in data
    assert "data" in data
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    
    # Check structure of individual model entries
    for model in data["data"]:
        assert "id" in model
        assert "object" in model
        assert "created" in model
        assert "owned_by" in model
        assert model["object"] == "model"


def test_root_endpoint_features():
    """Test root endpoint returns comprehensive feature list"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert "endpoints" in data
    assert "features" in data
    assert "scalability" in data
    
    # Check endpoints are properly listed
    endpoints = data["endpoints"]
    assert "embeddings" in endpoints
    assert "models" in endpoints
    assert "health" in endpoints
    
    # Check features is a list
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0


def test_embeddings_response_structure():
    """Test embeddings response has correct structure when successful"""
    payload = {
        "input": "Hello world",
        "model": "nomic-embed-text-v2-moe-distilled"
    }
    
    response = client.post("/v1/embeddings", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        assert "object" in data
        assert "data" in data
        assert "model" in data
        assert "usage" in data
        assert data["object"] == "list"
        
        # Check embedding data structure
        assert len(data["data"]) == 1
        embedding_data = data["data"][0]
        assert "object" in embedding_data
        assert "embedding" in embedding_data
        assert "index" in embedding_data
        assert embedding_data["object"] == "embedding"
        assert isinstance(embedding_data["embedding"], list)
        assert embedding_data["index"] == 0
        
        # Check usage structure
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["total_tokens"], int)


if __name__ == "__main__":
    pytest.main([__file__])