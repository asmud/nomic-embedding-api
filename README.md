# Nomic Embedding API

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Docker](https://img.shields.io/badge/docker-supported-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal.svg)

A high-performance, OpenAI-compatible REST API server for Nomic embedding models with advanced features like smart batching, caching, and multi-GPU support.

## Overview

Fast, production-ready embedding service that provides OpenAI-compatible endpoints using state-of-the-art Nomic embedding models. Built with FastAPI and optimized for scalability.

## Key Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI embedding endpoints
- **High Performance** - Smart batching, caching, and Model2Vec optimization
- **Multiple Models** - Support for various Nomic embedding models (256D, 768D)
- **Production Ready** - Multi-worker support, health checks, monitoring
- **Advanced Features** - Model pooling, Redis caching, hardware optimization
- **Container Ready** - Optimized Docker builds with multi-stage architecture

## Quick Start

```bash
# Local Installation
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

## API Usage

```bash
# Create embeddings
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "nomic-embed-text-v2-moe-distilled"}'

# Using OpenAI Python client
from openai import OpenAI
client = OpenAI(api_key="dummy", base_url="http://localhost:8000/v1")
response = client.embeddings.create(input="Hello world", model="nomic-embed-text-v2-moe-distilled")
```

## Architecture

- **FastAPI** - Modern, fast web framework
- **Multi-stage Docker** - Optimized container builds
- **Smart Batching** - Efficient request processing
- **Model Pooling** - Scalable inference with multiple model instances
- **Hybrid Caching** - In-memory + Redis distributed caching
- **Hardware Optimization** - Auto device selection and performance tuning

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Getting started guide with examples
- **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation instructions
- **API Docs** - Interactive documentation at `/docs` when running
- **Configuration** - See `.env.example` for all options
- **Issues** - Report bugs at [GitHub Issues](https://github.com/asmud/nomic-embedding-api/issues)
- **Discussions** - Ask questions at [GitHub Discussions](https://github.com/asmud/nomic-embedding-api/discussions)

## Models Supported

| Model | Dimensions | Speed | Use Case |
|-------|------------|-------|----------|
| `nomic-embed-text-v2-moe-distilled` | 256/768 | Ultra Fast | Real-time applications |
| `nomic-embed-text-v2-moe` | 768 | Fast | High accuracy tasks |

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