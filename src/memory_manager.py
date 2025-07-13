import gc
import logging
import psutil
import threading
import time
import asyncio
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import torch

from .config import CACHE_SIZE, MAX_BATCH_SIZE

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """Memory pressure levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    cache_size_mb: float
    gpu_memory_mb: float
    pressure_level: MemoryPressure
    
    
@dataclass
class MemoryThresholds:
    """Memory pressure thresholds"""
    medium_percent: float = 70.0
    high_percent: float = 85.0
    critical_percent: float = 95.0
    min_available_mb: float = 512.0


class MemoryManager:
    """Dynamic memory management with adaptive sizing"""
    
    def __init__(self, 
                 initial_cache_size: int = None,
                 initial_batch_size: int = None,
                 monitoring_interval: float = None):
        
        # Import here to avoid circular imports
        from .config import CACHE_SIZE, MAX_BATCH_SIZE, MEMORY_MONITORING_INTERVAL
        
        self.initial_cache_size = initial_cache_size or CACHE_SIZE
        self.initial_batch_size = initial_batch_size or MAX_BATCH_SIZE
        self.monitoring_interval = monitoring_interval or MEMORY_MONITORING_INTERVAL
        
        # Current dynamic values
        self.current_cache_size = self.initial_cache_size
        self.current_batch_size = self.initial_batch_size
        
        # Memory thresholds
        from .config import (
            MEMORY_PRESSURE_HIGH_PERCENT, 
            MEMORY_PRESSURE_CRITICAL_PERCENT,
            MIN_AVAILABLE_MEMORY_MB
        )
        
        self.thresholds = MemoryThresholds(
            high_percent=MEMORY_PRESSURE_HIGH_PERCENT,
            critical_percent=MEMORY_PRESSURE_CRITICAL_PERCENT,
            min_available_mb=MIN_AVAILABLE_MEMORY_MB
        )
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.stats_lock = threading.RLock()
        
        # Callbacks for dynamic adjustments
        self.cache_resize_callbacks: List[Callable[[int], None]] = []
        self.batch_resize_callbacks: List[Callable[[int], None]] = []
        
        # Statistics
        self.stats_history: List[MemoryStats] = []
        self.max_history = 100
        
        # GC optimization
        self.gc_threshold_adjustments = 0
        self.last_gc_time = time.time()
        self.gc_interval = 60.0  # seconds
        
        logger.info(f"Memory manager initialized - Cache: {initial_cache_size}, Batch: {initial_batch_size}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            total_mb = memory.total / (1024 ** 2)
            available_mb = memory.available / (1024 ** 2)
            used_mb = memory.used / (1024 ** 2)
            percent_used = memory.percent
            
            # Cache memory (estimate)
            cache_size_mb = self._estimate_cache_memory()
            
            # GPU memory
            gpu_memory_mb = self._get_gpu_memory()
            
            # Determine pressure level
            pressure_level = self._calculate_pressure_level(percent_used, available_mb)
            
            return MemoryStats(
                total_mb=total_mb,
                available_mb=available_mb,
                used_mb=used_mb,
                percent_used=percent_used,
                cache_size_mb=cache_size_mb,
                gpu_memory_mb=gpu_memory_mb,
                pressure_level=pressure_level
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0, 0, MemoryPressure.LOW)
    
    def _estimate_cache_memory(self) -> float:
        """Estimate memory used by caches"""
        try:
            from .cache import embedding_cache
            if hasattr(embedding_cache, 'local_cache'):
                memory_info = embedding_cache.local_cache.get_memory_usage()
                return memory_info.get('cache_size_mb', 0)
            return 0
        except Exception:
            return 0
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB"""
        try:
            if torch.cuda.is_available():
                total_gpu_memory = 0
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
                    total_gpu_memory += memory_allocated
                return total_gpu_memory
            return 0
        except Exception:
            return 0
    
    def _calculate_pressure_level(self, percent_used: float, available_mb: float) -> MemoryPressure:
        """Calculate memory pressure level"""
        if (percent_used >= self.thresholds.critical_percent or 
            available_mb < self.thresholds.min_available_mb):
            return MemoryPressure.CRITICAL
        elif percent_used >= self.thresholds.high_percent:
            return MemoryPressure.HIGH
        elif percent_used >= self.thresholds.medium_percent:
            return MemoryPressure.MEDIUM
        else:
            return MemoryPressure.LOW
    
    def start_monitoring(self):
        """Start memory monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_memory())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory monitoring stopped")
    
    async def _monitor_memory(self):
        """Main memory monitoring loop"""
        while self.monitoring_active:
            try:
                stats = self.get_memory_stats()
                
                with self.stats_lock:
                    self.stats_history.append(stats)
                    if len(self.stats_history) > self.max_history:
                        self.stats_history.pop(0)
                
                # Adjust settings based on memory pressure
                await self._adjust_for_memory_pressure(stats)
                
                # Periodic garbage collection
                self._maybe_run_gc()
                
                # Log memory status if pressure is high
                if stats.pressure_level in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
                    logger.warning(f"Memory pressure {stats.pressure_level.value}: "
                                 f"{stats.percent_used:.1f}% used, "
                                 f"{stats.available_mb:.1f}MB available")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _adjust_for_memory_pressure(self, stats: MemoryStats):
        """Adjust cache and batch sizes based on memory pressure"""
        old_cache_size = self.current_cache_size
        old_batch_size = self.current_batch_size
        
        if stats.pressure_level == MemoryPressure.CRITICAL:
            # Aggressive reduction
            self.current_cache_size = max(100, int(self.current_cache_size * 0.5))
            self.current_batch_size = max(4, int(self.current_batch_size * 0.5))
            # Force garbage collection
            await asyncio.to_thread(gc.collect)
            
        elif stats.pressure_level == MemoryPressure.HIGH:
            # Moderate reduction
            self.current_cache_size = max(200, int(self.current_cache_size * 0.7))
            self.current_batch_size = max(8, int(self.current_batch_size * 0.7))
            
        elif stats.pressure_level == MemoryPressure.MEDIUM:
            # Slight reduction
            self.current_cache_size = max(300, int(self.current_cache_size * 0.9))
            self.current_batch_size = max(16, int(self.current_batch_size * 0.9))
            
        elif stats.pressure_level == MemoryPressure.LOW:
            # Gradual recovery towards initial values
            target_cache = min(self.initial_cache_size, int(self.current_cache_size * 1.1))
            target_batch = min(self.initial_batch_size, int(self.current_batch_size * 1.1))
            
            self.current_cache_size = target_cache
            self.current_batch_size = target_batch
        
        # Notify callbacks if values changed
        if old_cache_size != self.current_cache_size:
            for callback in self.cache_resize_callbacks:
                try:
                    callback(self.current_cache_size)
                except Exception as e:
                    logger.error(f"Cache resize callback error: {e}")
        
        if old_batch_size != self.current_batch_size:
            for callback in self.batch_resize_callbacks:
                try:
                    callback(self.current_batch_size)
                except Exception as e:
                    logger.error(f"Batch resize callback error: {e}")
        
        if old_cache_size != self.current_cache_size or old_batch_size != self.current_batch_size:
            logger.info(f"Memory adjustment: Cache {old_cache_size}->{self.current_cache_size}, "
                       f"Batch {old_batch_size}->{self.current_batch_size}")
    
    def _maybe_run_gc(self):
        """Run garbage collection if needed"""
        current_time = time.time()
        if current_time - self.last_gc_time > self.gc_interval:
            try:
                # Get memory before GC
                before_stats = self.get_memory_stats()
                
                # Run garbage collection
                collected = gc.collect()
                
                # Get memory after GC
                after_stats = self.get_memory_stats()
                
                memory_freed = before_stats.used_mb - after_stats.used_mb
                
                if collected > 0 or memory_freed > 10:  # Log if significant
                    logger.info(f"GC collected {collected} objects, "
                               f"freed {memory_freed:.1f}MB")
                
                self.last_gc_time = current_time
                
            except Exception as e:
                logger.error(f"Garbage collection error: {e}")
    
    def optimize_gc_thresholds(self):
        """Optimize garbage collection thresholds based on usage"""
        try:
            # Get current thresholds
            thresholds = gc.get_threshold()
            
            # Adjust based on memory pressure history
            with self.stats_lock:
                if len(self.stats_history) >= 10:
                    recent_pressure = [s.pressure_level for s in self.stats_history[-10:]]
                    high_pressure_count = sum(1 for p in recent_pressure 
                                            if p in [MemoryPressure.HIGH, MemoryPressure.CRITICAL])
                    
                    if high_pressure_count > 5:  # Frequent high pressure
                        # More aggressive GC
                        new_thresholds = (
                            max(400, thresholds[0] - 100),
                            max(5, thresholds[1] - 2),
                            max(5, thresholds[2] - 2)
                        )
                        gc.set_threshold(*new_thresholds)
                        self.gc_threshold_adjustments += 1
                        logger.info(f"GC thresholds adjusted to {new_thresholds}")
                        
        except Exception as e:
            logger.error(f"GC threshold optimization error: {e}")
    
    def register_cache_resize_callback(self, callback: Callable[[int], None]):
        """Register callback for cache size changes"""
        self.cache_resize_callbacks.append(callback)
    
    def register_batch_resize_callback(self, callback: Callable[[int], None]):
        """Register callback for batch size changes"""
        self.batch_resize_callbacks.append(callback)
    
    def force_memory_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.info("Starting forced memory cleanup")
        
        try:
            # Clear caches if possible
            from .cache import embedding_cache
            embedding_cache.clear()
            
            # GPU memory cleanup
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.empty_cache()
            
            # Run garbage collection multiple times
            for _ in range(3):
                gc.collect()
            
            logger.info("Forced memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Forced memory cleanup error: {e}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        stats = self.get_memory_stats()
        
        # Calculate average pressure over recent history
        with self.stats_lock:
            recent_history = self.stats_history[-20:] if self.stats_history else []
            
        avg_percent_used = sum(s.percent_used for s in recent_history) / len(recent_history) if recent_history else 0
        
        return {
            "current": {
                "total_mb": round(stats.total_mb, 1),
                "used_mb": round(stats.used_mb, 1),
                "available_mb": round(stats.available_mb, 1),
                "percent_used": round(stats.percent_used, 1),
                "cache_size_mb": round(stats.cache_size_mb, 1),
                "gpu_memory_mb": round(stats.gpu_memory_mb, 1),
                "pressure_level": stats.pressure_level.value
            },
            "dynamic_sizing": {
                "initial_cache_size": self.initial_cache_size,
                "current_cache_size": self.current_cache_size,
                "initial_batch_size": self.initial_batch_size,
                "current_batch_size": self.current_batch_size,
                "cache_size_ratio": round(self.current_cache_size / self.initial_cache_size, 2),
                "batch_size_ratio": round(self.current_batch_size / self.initial_batch_size, 2)
            },
            "monitoring": {
                "active": self.monitoring_active,
                "interval_seconds": self.monitoring_interval,
                "history_size": len(self.stats_history),
                "avg_percent_used": round(avg_percent_used, 1)
            },
            "gc_info": {
                "threshold_adjustments": self.gc_threshold_adjustments,
                "thresholds": gc.get_threshold(),
                "counts": gc.get_count()
            },
            "thresholds": {
                "medium_percent": self.thresholds.medium_percent,
                "high_percent": self.thresholds.high_percent,
                "critical_percent": self.thresholds.critical_percent,
                "min_available_mb": self.thresholds.min_available_mb
            }
        }


# Global memory manager instance
memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager"""
    global memory_manager
    
    if memory_manager is None:
        memory_manager = MemoryManager()
    
    return memory_manager


async def start_memory_monitoring():
    """Start global memory monitoring"""
    manager = get_memory_manager()
    manager.start_monitoring()


async def stop_memory_monitoring():
    """Stop global memory monitoring"""
    if memory_manager:
        await memory_manager.stop_monitoring()


def get_memory_stats() -> Dict[str, Any]:
    """Get global memory statistics"""
    manager = get_memory_manager()
    return manager.get_memory_report()