import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Models cache directory
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Model presets with automatic configuration
MODEL_PRESETS = {
    "nomic-moe-768": {
        "model_name": "nomic-ai/nomic-embed-text-v2-moe",
        "use_model2vec": False,
        "dimensions": 768,
        "description": "Full Nomic MoE model (768 dimensions, high quality)"
    },
    "nomic-v1.5": {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "use_model2vec": False,
        "dimensions": 768,
        "description": "Full Nomic v1.5 model (768 dimensions, new improved)"
    },
    "nomic-moe-256": {
        "model_name": "Abdelkareem/nomic-embed-text-v2-moe_distilled", 
        "use_model2vec": True,
        "dimensions": 256,
        "description": "Distilled Nomic model (256 dimensions, fast)"
    }
}

# Environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-moe-768").lower()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"

# Performance configuration
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 32))
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", 50))
CACHE_SIZE = int(os.getenv("CACHE_SIZE", 1000))
ENABLE_QUANTIZATION = os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true"
TORCH_COMPILE = os.getenv("TORCH_COMPILE", "false").lower() == "true"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 100))
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

# Model pool configuration
MODEL_POOL_SIZE = int(os.getenv("MODEL_POOL_SIZE", 0))  # 0 = auto-detect
ENABLE_MULTI_GPU = os.getenv("ENABLE_MULTI_GPU", "true").lower() == "true"
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 30))
MAX_MODEL_ERROR_COUNT = int(os.getenv("MAX_MODEL_ERROR_COUNT", 5))

# Redis configuration (optional)
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "nomic_embedding")
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", 20))
REDIS_RETRY_ON_TIMEOUT = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", 3600))  # 1 hour
REDIS_SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", 300))  # 5 minutes

# Memory management configuration
ENABLE_DYNAMIC_MEMORY = os.getenv("ENABLE_DYNAMIC_MEMORY", "true").lower() == "true"
MEMORY_MONITORING_INTERVAL = float(os.getenv("MEMORY_MONITORING_INTERVAL", 30.0))
MEMORY_PRESSURE_HIGH_PERCENT = float(os.getenv("MEMORY_PRESSURE_HIGH_PERCENT", 85.0))
MEMORY_PRESSURE_CRITICAL_PERCENT = float(os.getenv("MEMORY_PRESSURE_CRITICAL_PERCENT", 95.0))
MIN_AVAILABLE_MEMORY_MB = float(os.getenv("MIN_AVAILABLE_MEMORY_MB", 512.0))
ENABLE_GC_OPTIMIZATION = os.getenv("ENABLE_GC_OPTIMIZATION", "true").lower() == "true"

# Hardware optimization configuration
ENABLE_HARDWARE_OPTIMIZATION = os.getenv("ENABLE_HARDWARE_OPTIMIZATION", "true").lower() == "true"
AUTO_DEVICE_SELECTION = os.getenv("AUTO_DEVICE_SELECTION", "true").lower() == "true"

# Validate and get model configuration
if EMBEDDING_MODEL not in MODEL_PRESETS:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Unknown EMBEDDING_MODEL '{EMBEDDING_MODEL}'. Available options: {list(MODEL_PRESETS.keys())}")
    logger.warning(f"Falling back to 'nomic-moe-768'")
    EMBEDDING_MODEL = "nomic-moe-768"

# Get current model configuration
CURRENT_MODEL_CONFIG = MODEL_PRESETS[EMBEDDING_MODEL]
MODEL_NAME = CURRENT_MODEL_CONFIG["model_name"]
USE_MODEL2VEC = CURRENT_MODEL_CONFIG["use_model2vec"]
EMBEDDING_DIMENSIONS = CURRENT_MODEL_CONFIG["dimensions"]


# Cache settings
CACHE_DIR = os.getenv("HF_HOME", str(MODELS_DIR))
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODELS_DIR)