#!/usr/bin/env python3

import sys
import os
import logging

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api import app
from src.server import run_production_server
from src.config import LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting Nomic Embedding API...")
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
        
        # Use production server with graceful shutdown
        run_production_server(app)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)