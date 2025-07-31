# Quick Start Guide

Get up and running with Nomic Embedding API in minutes.

## Prerequisites

- Docker (recommended) OR Python 3.11+
- 4GB+ RAM (8GB+ recommended)
- Internet connection for model downloads
- Git (for cloning the repository)

## Getting Started

First, clone the repository:

```bash
# Clone the repository
git clone https://github.com/asmud/nomic-embedding-api.git
cd nomic-embedding-api
```

## Option 1: Docker (Recommended)

### Basic Setup

```bash
# Build the image
docker build -t nomic-embedding-api .

# Run the container
docker run -d \
  --name nomic-embedding \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  nomic-embedding-api

# Check if it's running
curl http://localhost:8000/health
```

### With GPU Support

```bash
# Build with GPU support
docker build --build-arg INSTALL_NVIDIA=true -t nomic-embedding-api .

# Run with GPU
docker run -d \
  --name nomic-embedding \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  nomic-embedding-api
```

### Production Setup

```bash
# Multi-worker deployment
docker run -d \
  --name nomic-embedding \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e USE_GUNICORN=true \
  -e GUNICORN_WORKERS=4 \
  -e EMBEDDING_MODEL=nomic-moe-768 \
  nomic-embedding-api
```

## Option 2: Local Installation

```bash
# After cloning the repository (see above)
cd nomic-embedding-api

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py

# Or for development with auto-reload
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

## First API Call

Once the server is running, test it:

```bash
# Health check
curl http://localhost:8000/health

# Create embeddings (using latest model)
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, world!",
    "model": "nomic-v1.5"
  }'
```

## API Examples

### Single Text Embedding

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a sample text for embedding",
    "model": "nomic-v1.5"
  }'
```

### Batch Embeddings

```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "First text to embed",
      "Second text to embed", 
      "Third text to embed"
    ],
    "model": "nomic-v1.5"
  }'
```

### List Available Models

```bash
curl http://localhost:8000/v1/models
```

## Python Client Examples

### Using Requests

```python
import requests

# Single embedding
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": "Hello world",
    "model": "nomic-v1.5"
})

result = response.json()
embedding = result["data"][0]["embedding"]
print(f"Embedding dimension: {len(embedding)}")

# Batch embeddings
response = requests.post("http://localhost:8000/v1/embeddings", json={
    "input": ["Text 1", "Text 2", "Text 3"],
    "model": "nomic-v1.5"
})

result = response.json()
embeddings = [item["embedding"] for item in result["data"]]
print(f"Created {len(embeddings)} embeddings")
```

### Using OpenAI Client

```python
from openai import OpenAI

# Configure client for local server
client = OpenAI(
    api_key="dummy-key",  # Required but not used
    base_url="http://localhost:8000/v1"
)

# Create embeddings
response = client.embeddings.create(
    input="Hello world",
    model="nomic-v1.5"
)

embedding = response.data[0].embedding
print(f"Embedding: {embedding[:5]}...")  # First 5 dimensions
```

## Model Configuration

### Available Models

Set the model using environment variables:

```bash
# Latest model (768 dimensions) - RECOMMENDED
export EMBEDDING_MODEL=nomic-v1.5

# High accuracy model (768 dimensions, MoE)
export EMBEDDING_MODEL=nomic-moe-768

# Speed-optimized model (256 dimensions)
export EMBEDDING_MODEL=nomic-moe-256

# Run with specific model
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e EMBEDDING_MODEL=nomic-v1.5 \
  nomic-embedding-api
```

### Model Presets

| Preset | Model | Dimensions | Speed | Use Case |
|--------|-------|------------|-------|----------|
| `nomic-v1.5` | `asmud/nomic-embed-indonesian` | 768 | **Latest & Best** | **Recommended for all use cases** |
| `nomic-moe-768` | `nomic-ai/nomic-embed-text-v2-moe` | 768 | Fast | High accuracy applications |
| `nomic-moe-256` | `Abdelkareem/nomic-embed-text-v2-moe_distilled` | 256 | Ultra Fast | Speed-critical applications |

## Performance Tuning

### Basic Configuration

```bash
# Increase concurrent requests
-e MAX_CONCURRENT_REQUESTS=200

# Enable caching
-e ENABLE_CACHING=true
-e CACHE_SIZE=5000

# Model pool for scaling
-e MODEL_POOL_SIZE=2
```

### Redis Caching

```bash
# With Redis for distributed caching
docker run -d --name redis redis:alpine

docker run -d \
  --name nomic-embedding \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e REDIS_ENABLED=true \
  -e REDIS_URL=redis://redis:6379 \
  --link redis \
  nomic-embedding-api
```

## Monitoring

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Health & Stats

```bash
# Health check
curl http://localhost:8000/health

# System statistics
curl http://localhost:8000/stats

# Available endpoints
curl http://localhost:8000/
```

### Logs

```bash
# View container logs
docker logs -f nomic-embedding

# Set log level
-e LOG_LEVEL=DEBUG
```

## Testing

```bash
# Run the test client
python example_client.py

# Run unit tests
pytest tests/

# Performance tests
pytest tests/test_performance.py -v
```

## Troubleshooting

### Common Issues

1. **Port already in use**: Change port with `-p 8001:8000`
2. **Out of memory**: Reduce `MODEL_POOL_SIZE` or use `nomic-moe-256` model
3. **Slow first request**: Models download on first use (normal)

### Performance Tips

1. **Use Docker**: More consistent performance
2. **Mount models volume**: Avoids re-downloading models
3. **Enable caching**: Significant speedup for repeated texts
4. **Use GPU**: Add `--gpus all` for GPU acceleration
5. **Tune workers**: Start with `GUNICORN_WORKERS=4`

### Getting Help

- Check logs: `docker logs nomic-embedding`
- Test health: `curl http://localhost:8000/health`
- View stats: `curl http://localhost:8000/stats`
- See full installation guide: [INSTALLATION.md](INSTALLATION.md)

## Next Steps

- **Production**: See [INSTALLATION.md](INSTALLATION.md) for production deployment
- **Configuration**: Copy `.env.example` to `.env` and customize
- **Integration**: Use the OpenAI-compatible API in your applications
- **Scaling**: Enable model pooling and Redis caching for high throughput