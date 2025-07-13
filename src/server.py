import signal
import sys
import asyncio
import logging
import os
import multiprocessing
from typing import Optional
import uvicorn
from uvicorn.config import LOGGING_CONFIG

try:
    import gunicorn.app.base
    import gunicorn.config
    GUNICORN_AVAILABLE = True
except ImportError:
    GUNICORN_AVAILABLE = False

from .config import HOST, PORT, LOG_LEVEL

logger = logging.getLogger(__name__)

class GracefulShutdownServer:
    """Production-ready server with graceful shutdown handling"""
    
    def __init__(self):
        self.server: Optional[uvicorn.Server] = None
        self.shutdown_event = asyncio.Event()
        self.tasks = set()
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        # Handle SIGTERM (Docker, Kubernetes)
        signal.signal(signal.SIGTERM, signal_handler)
        # Handle SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)
        
        if hasattr(signal, 'SIGHUP'):
            # Handle SIGHUP (process manager reload)
            signal.signal(signal.SIGHUP, signal_handler)
    
    async def shutdown(self):
        """Perform graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        # Signal all components to shutdown
        self.shutdown_event.set()
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete or timeout
        if self.tasks:
            logger.info(f"Waiting for {len(self.tasks)} tasks to complete...")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout")
        
        # Shutdown the server
        if self.server:
            logger.info("Shutting down HTTP server...")
            self.server.should_exit = True
            await self.server.shutdown()
        
        logger.info("Graceful shutdown completed")
    
    def get_uvicorn_config(self, app) -> uvicorn.Config:
        """Get production-ready uvicorn configuration"""
        
        # Determine number of workers based on environment
        workers = int(os.getenv("UVICORN_WORKERS", "1"))
        
        # Enhanced logging configuration
        log_config = LOGGING_CONFIG.copy()
        log_config["formatters"]["default"]["fmt"] = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_config["formatters"]["access"]["fmt"] = (
            '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s'
        )
        
        # Base configuration parameters
        config_params = {
            "app": app,
            "host": HOST,
            "port": PORT,
            "log_level": LOG_LEVEL.lower(),
            "log_config": log_config,
            "access_log": True,
            "use_colors": False,  # Better for production logs
            
            # Connection settings
            "limit_concurrency": int(os.getenv("UVICORN_LIMIT_CONCURRENCY", "1000")),
            "limit_max_requests": int(os.getenv("UVICORN_LIMIT_MAX_REQUESTS", "10000")),
            "timeout_keep_alive": int(os.getenv("UVICORN_TIMEOUT_KEEP_ALIVE", "5")),
            "timeout_graceful_shutdown": int(os.getenv("UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN", "30")),
            
            # SSL settings (if certificates are provided)
            "ssl_keyfile": os.getenv("SSL_KEYFILE"),
            "ssl_certfile": os.getenv("SSL_CERTFILE"),
            "ssl_ca_certs": os.getenv("SSL_CA_CERTS"),
            
            # HTTP settings
            "h11_max_incomplete_event_size": int(os.getenv("H11_MAX_INCOMPLETE_EVENT_SIZE", "16384")),
            
            # Development vs Production
            "reload": False,  # Always False for production
        }
        
        # Only add worker settings if multiple workers are requested
        # Note: Multiple workers are typically handled by process managers like gunicorn
        # For single-process deployment, we don't use workers parameter
        if workers > 1:
            logger.warning("Multiple workers requested but uvicorn.Config doesn't support workers parameter")
            logger.warning("Use gunicorn with uvicorn workers for multi-process deployment")
            logger.info("Continuing with single-process configuration")
        
        config = uvicorn.Config(**config_params)
        return config
    
    async def serve(self, app):
        """Start the server with graceful shutdown support"""
        try:
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Create server configuration
            config = self.get_uvicorn_config(app)
            self.server = uvicorn.Server(config)
            
            logger.info(f"Starting server on {HOST}:{PORT}")
            logger.info(f"Log level: {LOG_LEVEL}")
            logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'production')}")
            
            # Start the server
            await self.server.serve()
            
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            logger.info("Server stopped")

class GunicornApplication(gunicorn.app.base.BaseApplication):
    """Custom Gunicorn application for FastAPI"""
    
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()
    
    def load_config(self):
        """Load gunicorn configuration"""
        config = {key: value for key, value in self.options.items()
                 if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)
    
    def load(self):
        """Return the application"""
        return self.application


def get_server_config():
    """Determine server configuration based on environment variables"""
    use_gunicorn = os.getenv("USE_GUNICORN", "false").lower() == "true"
    workers_env = os.getenv("GUNICORN_WORKERS", "auto")
    
    # Determine final configuration based on two conditions:
    # 1. USE_GUNICORN=true with any GUNICORN_WORKERS value
    # 2. GUNICORN_WORKERS > 1 (auto-enable gunicorn)
    
    if use_gunicorn:
        # USE_GUNICORN=true - use gunicorn with specified workers
        if workers_env == "auto":
            workers = multiprocessing.cpu_count()
        else:
            try:
                workers = int(workers_env)
            except ValueError:
                logger.warning(f"Invalid GUNICORN_WORKERS value: {workers_env}, using auto")
                workers = multiprocessing.cpu_count()
        
        logger.info(f"USE_GUNICORN=true, using gunicorn mode with {workers} workers")
        
    else:
        # USE_GUNICORN=false - respect the explicit setting, always use single worker uvicorn
        workers = 1  # Force single worker when USE_GUNICORN=false
        if workers_env != "auto" and workers_env != "1":
            logger.warning(f"USE_GUNICORN=false but GUNICORN_WORKERS={workers_env} specified. Ignoring worker count, using single worker uvicorn mode")
        else:
            logger.info("USE_GUNICORN=false, using single worker uvicorn mode")
    
    return {
        "use_gunicorn": use_gunicorn,
        "workers": workers,
        "timeout": int(os.getenv("GUNICORN_TIMEOUT", "300")),
        "keepalive": int(os.getenv("GUNICORN_KEEPALIVE", "2")),
        "max_requests": int(os.getenv("GUNICORN_MAX_REQUESTS", "10000")),
        "max_requests_jitter": int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "1000"))
    }


def run_gunicorn_server(app):
    """Run server with gunicorn for multi-worker support"""
    if not GUNICORN_AVAILABLE:
        logger.error("Gunicorn not available. Install with: pip install gunicorn")
        raise ImportError("Gunicorn not available")
    
    config = get_server_config()
    
    # Determine worker temp directory based on platform
    import platform
    if platform.system() == "Linux" and os.path.exists("/dev/shm"):
        worker_tmp_dir = "/dev/shm"  # Use shared memory on Linux
    else:
        worker_tmp_dir = None  # Use default temp directory on other platforms
    
    # Gunicorn configuration options
    options = {
        "bind": f"{HOST}:{PORT}",
        "workers": config["workers"],
        "worker_class": "uvicorn.workers.UvicornWorker",
        "timeout": config["timeout"],
        "keepalive": config["keepalive"],
        "max_requests": config["max_requests"],
        "max_requests_jitter": config["max_requests_jitter"],
        "preload_app": True,  # Load app before forking workers
        "log_level": LOG_LEVEL.lower(),
        "access_log_format": '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
        "capture_output": True,
    }
    
    # Only set worker_tmp_dir if available
    if worker_tmp_dir:
        options["worker_tmp_dir"] = worker_tmp_dir
    
    logger.info(f"Starting Gunicorn server with {config['workers']} workers")
    logger.info(f"Binding to {HOST}:{PORT}")
    
    try:
        GunicornApplication(app, options).run()
    except Exception as e:
        logger.error(f"Gunicorn server failed: {e}")
        raise


def run_uvicorn_server(app):
    """Run server with uvicorn for single-worker support"""
    logger.info("Starting Uvicorn server (single process)")
    try:
        asyncio.run(start_production_server(app))
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Uvicorn server failed: {e}")
        raise


# Global server instance
graceful_server = GracefulShutdownServer()

async def start_production_server(app):
    """Start production server with graceful shutdown"""
    await graceful_server.serve(app)

def run_production_server(app):
    """Run production server with auto-detection (blocking)"""
    try:
        config = get_server_config()
        
        logger.info(f"Server configuration: workers={config['workers']}, gunicorn={config['use_gunicorn']}")
        
        if config["use_gunicorn"]:
            run_gunicorn_server(app)
        else:
            run_uvicorn_server(app)
            
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)