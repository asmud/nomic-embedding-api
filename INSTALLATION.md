# Installation Guide

Comprehensive installation instructions for Nomic Embedding API with different deployment options.

## System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: 10GB free space (for models)
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB for production)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional)
- **Storage**: 20GB+ SSD

### Software Dependencies
- **Docker**: 20.10+ (for container deployment)
- **Python**: 3.11+ (for local installation)
- **Git**: For cloning the repository

## Installation Methods

## Method 1: Docker Installation (Recommended)

### Basic Docker Setup

```bash
# Clone the repository
git clone <repository-url>
cd nomic-embedding-v2-custom

# Build the Docker image
docker build -t nomic-embedding-api .

# Run the container
docker run -d \
  --name nomic-embedding \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  nomic-embedding-api

# Verify installation
curl http://localhost:8000/health
```

### Docker with Build Arguments

```bash
# Build with performance optimizations
docker build \
  --build-arg INSTALL_PERFORMANCE=true \
  -t nomic-embedding-api .

# Build with GPU support
docker build \
  --build-arg INSTALL_NVIDIA=true \
  -t nomic-embedding-api .

# Build with both optimizations
docker build \
  --build-arg INSTALL_PERFORMANCE=true \
  --build-arg INSTALL_NVIDIA=true \
  -t nomic-embedding-api .
```

### Multi-platform Docker Build

```bash
# For ARM64 (Apple Silicon) and AMD64
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t nomic-embedding-api \
  --push .
```

## Method 2: Local Python Installation

### Python Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd nomic-embedding-v2-custom

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Core Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install from pyproject.toml
pip install -e .
```

### Optional Dependencies

```bash
# Performance optimizations (requires build tools)
pip install -e ".[performance]"

# GPU support
pip install -e ".[nvidia]"

# Development tools
pip install -e ".[dev]"

# All optional dependencies
pip install -e ".[all]"
```

### Build Tools Installation

For performance optimizations, you may need build tools:

#### macOS
```bash
# Install Xcode command line tools
xcode-select --install

# Or install via Homebrew
brew install cmake
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential cmake python3-dev
```

#### CentOS/RHEL
```bash
sudo yum groupinstall "Development Tools"
sudo yum install cmake python3-devel
```

#### Windows
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

# Or use chocolatey
choco install visualstudio2022buildtools
```

## Method 3: Development Installation

### For Contributors

```bash
# Clone with development setup
git clone <repository-url>
cd nomic-embedding-v2-custom

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest tests/
```

### IDE Setup

For VS Code, install recommended extensions:
- Python
- Docker
- Pylance
- Black Formatter

## Configuration

### Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit the `.env` file with your configuration:

```bash
# Model Configuration
EMBEDDING_MODEL=nomic-v1.5                 # Latest model (recommended)
# EMBEDDING_MODEL=nomic-moe-768            # Alternative: High accuracy
# EMBEDDING_MODEL=nomic-moe-256            # Alternative: Speed optimized

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Model Storage (uses standard HuggingFace cache format)
HF_HOME=./models
TRANSFORMERS_CACHE=./models
SENTENCE_TRANSFORMERS_HOME=./models

# Performance Settings
MAX_CONCURRENT_REQUESTS=50
ENABLE_CACHING=true
CACHE_SIZE=1000

# Model Pool (for scaling)
MODEL_POOL_SIZE=0                      # 0 = disabled, >0 = number of model instances
ENABLE_MULTI_GPU=false

# Redis (optional distributed caching)
REDIS_ENABLED=false
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# Hardware Optimization
ENABLE_HARDWARE_OPTIMIZATION=true
AUTO_DEVICE_SELECTION=true

# Memory Management
ENABLE_DYNAMIC_MEMORY=false
MEMORY_THRESHOLD_MB=6000
CACHE_CLEANUP_INTERVAL=300

# Production Settings
ENVIRONMENT=production
USE_GUNICORN=false                     # Set to true for multi-worker mode
GUNICORN_WORKERS=auto                  # Number of workers or "auto"
GUNICORN_TIMEOUT=300
```

### Model Presets

The application supports simplified model configuration:

```bash
# Latest model (768 dimensions, best performance) - RECOMMENDED
EMBEDDING_MODEL=nomic-v1.5

# High accuracy model (768 dimensions, MoE architecture)
EMBEDDING_MODEL=nomic-moe-768

# Speed-optimized model (256 dimensions, Model2Vec)
EMBEDDING_MODEL=nomic-moe-256
```

Model details:
- **nomic-v1.5**: `nomic-ai/nomic-embed-text-v1.5` (SentenceTransformers, 768D) - **Latest & Best**
- **nomic-moe-768**: `nomic-ai/nomic-embed-text-v2-moe` (SentenceTransformers, 768D)
- **nomic-moe-256**: `Abdelkareem/nomic-embed-text-v2-moe_distilled` (Model2Vec, 256D)

## Production Deployment

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  nomic-embedding:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - EMBEDDING_MODEL=nomic-moe-768
      - USE_GUNICORN=true
      - GUNICORN_WORKERS=4
      - MAX_CONCURRENT_REQUESTS=200
      - ENABLE_CACHING=true
      - REDIS_ENABLED=true
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

Deploy with:
```bash
docker-compose up -d
```

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nomic-embedding
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nomic-embedding
  template:
    metadata:
      labels:
        app: nomic-embedding
    spec:
      containers:
      - name: nomic-embedding
        image: nomic-embedding-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: EMBEDDING_MODEL
          value: "nomic-moe-768"
        - name: USE_GUNICORN
          value: "true"
        - name: GUNICORN_WORKERS
          value: "4"
        - name: MAX_CONCURRENT_REQUESTS
          value: "200"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nomic-embedding-service
spec:
  selector:
    app: nomic-embedding
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### Systemd Service (Linux)

Create `/etc/systemd/system/nomic-embedding.service`:

```ini
[Unit]
Description=Nomic Embedding API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/nomic-embedding
Environment=PATH=/opt/nomic-embedding/venv/bin
ExecStart=/opt/nomic-embedding/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable nomic-embedding
sudo systemctl start nomic-embedding
```

## GPU Support

### NVIDIA Docker Setup

```bash
# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### GPU-enabled Container

```bash
# Build with GPU support
docker build --build-arg INSTALL_NVIDIA=true -t nomic-embedding-api .

# Run with GPU
docker run -d \
  --name nomic-embedding \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e EMBEDDING_MODEL=nomic-moe-768 \
  nomic-embedding-api
```

## Performance Optimization

### Memory Settings

```bash
# For systems with limited memory
EMBEDDING_MODEL=nomic-moe-256
MODEL_POOL_SIZE=1
MAX_CONCURRENT_REQUESTS=25
CACHE_SIZE=500

# For high-memory systems
EMBEDDING_MODEL=nomic-moe-768
MODEL_POOL_SIZE=4
MAX_CONCURRENT_REQUESTS=200
CACHE_SIZE=5000
```

### CPU Optimization

```bash
# Single worker (default)
USE_GUNICORN=false

# Multi-worker (CPU intensive workloads)
USE_GUNICORN=true
GUNICORN_WORKERS=auto  # Uses CPU count

# Custom worker count
GUNICORN_WORKERS=4
```

### Disk I/O Optimization

```bash
# Use SSD for model storage
-v /path/to/ssd/models:/app/models

# Enable tmpfs for temporary files
--tmpfs /tmp:rw,noexec,nosuid,size=2g
```

## Troubleshooting

### Common Installation Issues

#### Build Errors
```bash
# Missing build tools
sudo apt install build-essential python3-dev

# macOS missing Xcode tools
xcode-select --install
```

#### Memory Issues
```bash
# Reduce memory usage
EMBEDDING_MODEL=nomic-moe-256
MODEL_POOL_SIZE=0
MAX_CONCURRENT_REQUESTS=10
```

#### Permission Issues
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER ./models
```

### Performance Issues

#### Slow Model Loading
```bash
# Pre-download models
python -c "from src.models import EmbeddingModel; EmbeddingModel()"
```

#### High Memory Usage
```bash
# Enable memory management
ENABLE_DYNAMIC_MEMORY=true
MEMORY_THRESHOLD_MB=4000
```

#### Poor Performance
```bash
# Enable hardware optimization
ENABLE_HARDWARE_OPTIMIZATION=true
AUTO_DEVICE_SELECTION=true
OPTIMIZATION_LEVEL=aggressive
```

### Verification

#### Test Installation
```bash
# Health check
curl http://localhost:8000/health

# Test embedding
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "nomic-v1.5"}'

# Check system stats
curl http://localhost:8000/stats
```

#### Performance Test
```bash
# Run performance tests
pytest tests/test_performance.py -v

# Load test with custom client
python example_client.py
```

## Upgrading

### Docker Upgrade
```bash
# Pull latest changes
git pull origin main

# Rebuild image
docker build -t nomic-embedding-api .

# Stop old container
docker stop nomic-embedding
docker rm nomic-embedding

# Start new container
docker run -d \
  --name nomic-embedding \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  nomic-embedding-api
```

### Local Upgrade
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart nomic-embedding
```

## Security

### Production Security

```bash
# Run as non-root user
USER appuser

# Limit container capabilities
--cap-drop=ALL
--cap-add=NET_BIND_SERVICE

# Use secrets for sensitive data
docker secret create redis_password /path/to/password.txt
```

### Firewall Configuration
```bash
# Allow only necessary ports
sudo ufw allow 8000/tcp
sudo ufw enable
```

## Monitoring

### Health Monitoring
```bash
# Add health check to monitoring
*/5 * * * * curl -f http://localhost:8000/health || systemctl restart nomic-embedding
```

### Log Management
```bash
# Configure log rotation
/app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

## Support

- **Documentation**: Check `/docs` endpoint when server is running
- **Issues**: Report bugs via GitHub issues
- **Configuration**: See `.env.example` for all options
- **Performance**: Run `pytest tests/test_performance.py` for benchmarks