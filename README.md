# Nomic Embedding API

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal.svg)

A high-performance, OpenAI-compatible REST API server for Nomic embedding models with optimized caching, smart batching, and multi-GPU support.

## Overview

Production-ready embedding service that provides OpenAI-compatible endpoints using the latest Nomic embedding models. Built with FastAPI and optimized for performance and reliability.

## Key Features

- **🔌 OpenAI-Compatible API** - Drop-in replacement for OpenAI embedding endpoints
- **⚡ High Performance** - Smart batching, optimized caching, and hardware acceleration
- **🎯 Latest Models** - Support for Nomic v1.5, v2-MoE, and distilled models
- **🏭 Production Ready** - Health checks, monitoring, graceful shutdown
- **🚀 Advanced Features** - Model pooling, Redis caching, dynamic memory management
- **🐳 Container Ready** - Optimized Docker deployment with HuggingFace cache

## Quick Start

```bash
# Local Installation
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

## API Usage

```bash
# Create embeddings with latest model
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "nomic-v1.5"}'

# List available models
curl "http://localhost:8000/v1/models"

# Using OpenAI Python client
from openai import OpenAI
client = OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")
response = client.embeddings.create(input="Hello world", model="nomic-v1.5")
```

## Architecture

- **FastAPI** - Modern, async web framework with automatic OpenAPI docs
- **HuggingFace Integration** - Standardized model caching and loading
- **Smart Batching** - Efficient request processing with configurable timeouts
- **Model Pooling** - Scalable inference with health monitoring
- **Hybrid Caching** - In-memory + optional Redis distributed caching
- **Hardware Optimization** - Auto device selection and memory management

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Getting started guide with examples
- **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation instructions
- **API Docs** - Interactive documentation at `/docs` when running
- **Configuration** - See `.env.example` for all options
- **Issues** - Report bugs at [GitHub Issues](https://github.com/asmud/nomic-embedding-api/issues)
- **Discussions** - Ask questions at [GitHub Discussions](https://github.com/asmud/nomic-embedding-api/discussions)

## Models Supported

| Model Preset | HuggingFace Model | Dimensions | Library | Use Case |
|--------------|-------------------|------------|---------|----------|
| `nomic-v1.5` | `asmud/nomic-embed-indonesian` | 768 | SentenceTransformers | **Latest & Best** - Improved performance |
| `nomic-moe-768` | `nomic-ai/nomic-embed-text-v2-moe` | 768 | SentenceTransformers | High accuracy applications |
| `nomic-moe-256` | `Abdelkareem/nomic-embed-text-v2-moe_distilled` | 256 | Model2Vec | Speed-optimized applications |

Use `/v1/models` endpoint to see all available models with detailed information.

## Requirements

- Python 3.11+
- Docker (for containerized deployment)
- 4GB+ RAM (8GB+ recommended)
- Optional: NVIDIA GPU for acceleration

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. 🍴 Fork the repository
2. 🌟 Star this repository if you find it useful
3. 🔧 Create a feature branch (`git checkout -b feature/amazing-feature`)
4. ✅ Run tests: `pytest tests/`
5. 📝 Commit your changes (`git commit -m 'Add amazing feature'`)
6. 🚀 Push to the branch (`git push origin feature/amazing-feature`)
7. 🎯 Open a Pull Request

### Getting Help

- 📚 **Documentation**: Check our [QUICKSTART.md](QUICKSTART.md) and [INSTALLATION.md](INSTALLATION.md)

---

**Need help?** Check [QUICKSTART.md](QUICKSTART.md) for examples or [INSTALLATION.md](INSTALLATION.md) for setup details.