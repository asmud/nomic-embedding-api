import asyncio
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
import psutil
import torch

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Hardware device types"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    

class OptimizationLevel(Enum):
    """Optimization aggressiveness levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class CPUInfo:
    """CPU hardware information"""
    cores_physical: int
    cores_logical: int
    frequency_max: float
    frequency_current: float
    cache_l1: int
    cache_l2: int
    cache_l3: int
    architecture: str
    features: List[str]
    utilization_percent: float
    temperature: Optional[float] = None


@dataclass
class GPUInfo:
    """GPU hardware information"""
    device_id: int
    name: str
    memory_total: int
    memory_free: int
    memory_used: int
    utilization_percent: float
    temperature: Optional[float] = None
    power_usage: Optional[float] = None
    compute_capability: Optional[Tuple[int, int]] = None


@dataclass
class WorkloadProfile:
    """Workload characteristics for optimization"""
    avg_batch_size: float
    avg_sequence_length: float
    avg_processing_time: float
    memory_usage_mb: float
    cpu_intensive: bool
    gpu_intensive: bool
    io_bound: bool


class HardwareOptimizer:
    """Comprehensive CPU/GPU optimization system"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.cpu_info: Optional[CPUInfo] = None
        self.gpu_info: List[GPUInfo] = []
        self.optimal_workers = self._calculate_optimal_workers()
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.workload_profile = WorkloadProfile(0, 0, 0, 0, False, False, False)
        
        # Optimization settings
        self.cpu_affinity_enabled = False
        self.numa_aware = False
        self.thread_pinning_enabled = False
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.stats_lock = threading.RLock()
        
        logger.info(f"Hardware optimizer initialized with {optimization_level.value} mode")
        self._initialize_hardware_detection()
    
    def _initialize_hardware_detection(self):
        """Detect and analyze hardware capabilities"""
        try:
            self.cpu_info = self._detect_cpu_info()
            self.gpu_info = self._detect_gpu_info()
            
            logger.info(f"Hardware detected: {self.cpu_info.cores_physical}C/{self.cpu_info.cores_logical}T CPU, "
                       f"{len(self.gpu_info)} GPU(s)")
            
            # Configure optimization based on hardware
            self._configure_optimization()
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
    
    def _detect_cpu_info(self) -> CPUInfo:
        """Detect CPU hardware information"""
        try:
            # Basic CPU info
            cores_physical = psutil.cpu_count(logical=False) or 1
            cores_logical = psutil.cpu_count(logical=True) or 1
            
            # CPU frequency
            freq_info = psutil.cpu_freq()
            freq_max = freq_info.max if freq_info else 0
            freq_current = freq_info.current if freq_info else 0
            
            # CPU utilization
            utilization = psutil.cpu_percent(interval=1)
            
            # Try to get cache info (Linux)
            cache_l1 = cache_l2 = cache_l3 = 0
            try:
                import subprocess
                lscpu = subprocess.check_output(['lscpu'], text=True)
                for line in lscpu.split('\n'):
                    if 'L1d cache' in line:
                        cache_l1 = self._parse_cache_size(line)
                    elif 'L2 cache' in line:
                        cache_l2 = self._parse_cache_size(line)
                    elif 'L3 cache' in line:
                        cache_l3 = self._parse_cache_size(line)
            except:
                pass
            
            # CPU features
            features = []
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                features = cpu_info.get('flags', [])
                architecture = cpu_info.get('arch', 'unknown')
            except ImportError:
                architecture = 'unknown'
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    cpu_temps = temps.get('coretemp', [])
                    if cpu_temps:
                        temperature = cpu_temps[0].current
            except:
                pass
            
            return CPUInfo(
                cores_physical=cores_physical,
                cores_logical=cores_logical,
                frequency_max=freq_max,
                frequency_current=freq_current,
                cache_l1=cache_l1,
                cache_l2=cache_l2,
                cache_l3=cache_l3,
                architecture=architecture,
                features=features,
                utilization_percent=utilization,
                temperature=temperature
            )
            
        except Exception as e:
            logger.error(f"CPU detection error: {e}")
            return CPUInfo(1, 1, 0, 0, 0, 0, 0, 'unknown', [], 0)
    
    def _parse_cache_size(self, line: str) -> int:
        """Parse cache size from lscpu output"""
        try:
            parts = line.split()
            for part in parts:
                if 'K' in part:
                    return int(part.replace('K', '')) * 1024
                elif 'M' in part:
                    return int(part.replace('M', '')) * 1024 * 1024
        except:
            pass
        return 0
    
    def _detect_gpu_info(self) -> List[GPUInfo]:
        """Detect GPU hardware information"""
        gpu_list = []
        
        if not torch.cuda.is_available():
            return gpu_list
        
        try:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                
                # Memory info
                memory_total = props.total_memory
                memory_free = memory_total - torch.cuda.memory_allocated(i)
                memory_used = torch.cuda.memory_allocated(i)
                
                # Utilization (approximate)
                utilization = (memory_used / memory_total * 100) if memory_total > 0 else 0
                
                # Compute capability
                compute_capability = (props.major, props.minor)
                
                # Temperature and power (if available via nvidia-ml-py)
                temperature = power_usage = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                except ImportError:
                    pass
                except Exception:
                    pass
                
                gpu_info = GPUInfo(
                    device_id=i,
                    name=props.name,
                    memory_total=memory_total,
                    memory_free=memory_free,
                    memory_used=memory_used,
                    utilization_percent=utilization,
                    temperature=temperature,
                    power_usage=power_usage,
                    compute_capability=compute_capability
                )
                
                gpu_list.append(gpu_info)
                
        except Exception as e:
            logger.error(f"GPU detection error: {e}")
        
        return gpu_list
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads"""
        try:
            cpu_cores = multiprocessing.cpu_count()
            
            if self.optimization_level == OptimizationLevel.CONSERVATIVE:
                return max(1, cpu_cores // 2)
            elif self.optimization_level == OptimizationLevel.BALANCED:
                return cpu_cores
            else:  # AGGRESSIVE
                return cpu_cores * 2
                
        except Exception:
            return 4  # Safe default
    
    def _configure_optimization(self):
        """Configure optimization based on hardware and level"""
        try:
            # Configure thread pool
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.optimal_workers,
                thread_name_prefix="embedding_worker"
            )
            
            # Configure PyTorch settings
            self._configure_pytorch()
            
            # Configure CPU affinity if aggressive optimization
            if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                self._configure_cpu_affinity()
            
            logger.info(f"Optimization configured: {self.optimal_workers} workers, "
                       f"PyTorch threads: {torch.get_num_threads()}")
            
        except Exception as e:
            logger.error(f"Optimization configuration error: {e}")
    
    def _configure_pytorch(self):
        """Configure PyTorch for optimal performance"""
        try:
            # Set optimal number of threads
            if self.cpu_info:
                if self.optimization_level == OptimizationLevel.CONSERVATIVE:
                    num_threads = max(1, self.cpu_info.cores_physical)
                elif self.optimization_level == OptimizationLevel.BALANCED:
                    num_threads = self.cpu_info.cores_logical
                else:  # AGGRESSIVE
                    num_threads = self.cpu_info.cores_logical
                
                torch.set_num_threads(num_threads)
                torch.set_num_interop_threads(max(1, num_threads // 2))
            
            # Enable optimizations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # MKL optimizations (Intel CPUs)
            if 'mkl' in torch.__config__.parallel_info():
                os.environ['MKL_NUM_THREADS'] = str(torch.get_num_threads())
                os.environ['MKL_DYNAMIC'] = 'FALSE'
            
            # OpenMP optimizations
            os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())
            os.environ['OMP_DYNAMIC'] = 'FALSE'
            
        except Exception as e:
            logger.error(f"PyTorch configuration error: {e}")
    
    def _configure_cpu_affinity(self):
        """Configure CPU affinity for performance"""
        try:
            if self.cpu_info and self.cpu_info.cores_physical > 1:
                # Pin process to specific cores if aggressive optimization
                process = psutil.Process()
                available_cores = list(range(self.cpu_info.cores_physical))
                process.cpu_affinity(available_cores)
                self.cpu_affinity_enabled = True
                logger.info(f"CPU affinity set to cores: {available_cores}")
                
        except Exception as e:
            logger.warning(f"CPU affinity configuration failed: {e}")
    
    def get_optimal_device(self, memory_required: int = 0) -> str:
        """Select optimal device for inference"""
        try:
            # Check GPU availability and memory
            if self.gpu_info:
                best_gpu = None
                best_score = -1
                
                for gpu in self.gpu_info:
                    # Skip if insufficient memory
                    if memory_required > 0 and gpu.memory_free < memory_required:
                        continue
                    
                    # Score based on free memory and low utilization
                    score = (gpu.memory_free / gpu.memory_total) * (100 - gpu.utilization_percent)
                    
                    if score > best_score:
                        best_score = score
                        best_gpu = gpu
                
                if best_gpu:
                    return f"cuda:{best_gpu.device_id}"
            
            # Fallback to CPU
            return "cpu"
            
        except Exception as e:
            logger.error(f"Device selection error: {e}")
            return "cpu"
    
    def get_optimal_batch_size(self, model_memory_mb: float, sequence_length: int) -> int:
        """Calculate optimal batch size based on hardware"""
        try:
            if self.gpu_info:
                # GPU-based calculation
                gpu = max(self.gpu_info, key=lambda g: g.memory_free)
                available_memory_mb = gpu.memory_free / (1024 * 1024)
                
                # Estimate memory per sample (rough heuristic)
                memory_per_sample = model_memory_mb * (sequence_length / 512)  # Normalize to 512 tokens
                
                # Conservative estimate (leave 20% buffer)
                max_batch_size = int((available_memory_mb * 0.8) / memory_per_sample)
                
                # Clamp to reasonable range
                return max(1, min(max_batch_size, 64))
            else:
                # CPU-based calculation
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
                memory_per_sample = model_memory_mb * (sequence_length / 512)
                
                max_batch_size = int((available_memory_mb * 0.5) / memory_per_sample)
                return max(1, min(max_batch_size, 32))
                
        except Exception as e:
            logger.error(f"Batch size calculation error: {e}")
            return 8  # Safe default
    
    async def start_monitoring(self):
        """Start hardware utilization monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_hardware())
        logger.info("Hardware monitoring started")
    
    async def stop_monitoring(self):
        """Stop hardware utilization monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Hardware monitoring stopped")
    
    async def _monitor_hardware(self):
        """Monitor hardware utilization and performance"""
        while self.monitoring_active:
            try:
                # Update hardware stats
                self.cpu_info = self._detect_cpu_info()
                self.gpu_info = self._detect_gpu_info()
                
                # Record performance metrics
                with self.stats_lock:
                    perf_data = {
                        "timestamp": time.time(),
                        "cpu_utilization": self.cpu_info.utilization_percent,
                        "cpu_temperature": self.cpu_info.temperature,
                        "gpu_utilization": [g.utilization_percent for g in self.gpu_info],
                        "gpu_memory_usage": [g.memory_used / g.memory_total * 100 for g in self.gpu_info],
                        "gpu_temperature": [g.temperature for g in self.gpu_info if g.temperature],
                        "optimal_workers": self.optimal_workers
                    }
                    
                    self.performance_history.append(perf_data)
                    if len(self.performance_history) > 100:  # Keep last 100 records
                        self.performance_history.pop(0)
                
                # Dynamic optimization adjustments
                await self._dynamic_optimization()
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Hardware monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _dynamic_optimization(self):
        """Dynamically adjust optimization based on current workload"""
        try:
            if not self.performance_history:
                return
            
            recent_data = self.performance_history[-5:]  # Last 5 measurements
            avg_cpu_util = sum(d["cpu_utilization"] for d in recent_data) / len(recent_data)
            
            # Adjust worker count based on CPU utilization
            if avg_cpu_util > 90 and self.optimization_level == OptimizationLevel.AGGRESSIVE:
                # Reduce workers if CPU is overloaded
                new_workers = max(1, self.optimal_workers - 1)
                if new_workers != self.optimal_workers:
                    self.optimal_workers = new_workers
                    await self._reconfigure_thread_pool()
                    logger.info(f"Reduced workers to {new_workers} due to high CPU utilization")
            
            elif avg_cpu_util < 50 and self.optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
                # Increase workers if CPU is underutilized
                max_workers = self.cpu_info.cores_logical * 2 if self.optimization_level == OptimizationLevel.AGGRESSIVE else self.cpu_info.cores_logical
                new_workers = min(max_workers, self.optimal_workers + 1)
                if new_workers != self.optimal_workers:
                    self.optimal_workers = new_workers
                    await self._reconfigure_thread_pool()
                    logger.info(f"Increased workers to {new_workers} due to low CPU utilization")
            
        except Exception as e:
            logger.error(f"Dynamic optimization error: {e}")
    
    async def _reconfigure_thread_pool(self):
        """Reconfigure thread pool with new worker count"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.optimal_workers,
                thread_name_prefix="embedding_worker"
            )
            
        except Exception as e:
            logger.error(f"Thread pool reconfiguration error: {e}")
    
    def update_workload_profile(self, batch_size: int, sequence_length: int, processing_time: float, memory_usage: float):
        """Update workload profile for optimization"""
        try:
            # Update running averages
            alpha = 0.1  # Smoothing factor
            
            self.workload_profile.avg_batch_size = (
                (1 - alpha) * self.workload_profile.avg_batch_size + 
                alpha * batch_size
            )
            
            self.workload_profile.avg_sequence_length = (
                (1 - alpha) * self.workload_profile.avg_sequence_length + 
                alpha * sequence_length
            )
            
            self.workload_profile.avg_processing_time = (
                (1 - alpha) * self.workload_profile.avg_processing_time + 
                alpha * processing_time
            )
            
            self.workload_profile.memory_usage_mb = (
                (1 - alpha) * self.workload_profile.memory_usage_mb + 
                alpha * memory_usage
            )
            
            # Classify workload characteristics
            self.workload_profile.cpu_intensive = processing_time > 0.1  # > 100ms
            self.workload_profile.gpu_intensive = memory_usage > 1000  # > 1GB
            self.workload_profile.io_bound = processing_time < 0.01  # < 10ms
            
        except Exception as e:
            logger.error(f"Workload profile update error: {e}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        try:
            with self.stats_lock:
                recent_perf = self.performance_history[-10:] if self.performance_history else []
            
            avg_cpu_util = sum(d["cpu_utilization"] for d in recent_perf) / len(recent_perf) if recent_perf else 0
            avg_gpu_util = []
            if recent_perf and recent_perf[0]["gpu_utilization"]:
                num_gpus = len(recent_perf[0]["gpu_utilization"])
                for i in range(num_gpus):
                    gpu_utils = [d["gpu_utilization"][i] for d in recent_perf if len(d["gpu_utilization"]) > i]
                    avg_gpu_util.append(sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0)
            
            return {
                "hardware": {
                    "cpu": {
                        "cores_physical": self.cpu_info.cores_physical if self.cpu_info else 0,
                        "cores_logical": self.cpu_info.cores_logical if self.cpu_info else 0,
                        "current_utilization": round(avg_cpu_util, 1),
                        "temperature": self.cpu_info.temperature if self.cpu_info else None,
                        "frequency_mhz": round(self.cpu_info.frequency_current, 0) if self.cpu_info else 0
                    },
                    "gpu": [
                        {
                            "device_id": gpu.device_id,
                            "name": gpu.name,
                            "memory_total_gb": round(gpu.memory_total / (1024**3), 1),
                            "memory_used_gb": round(gpu.memory_used / (1024**3), 1),
                            "utilization_percent": round(avg_gpu_util[i] if i < len(avg_gpu_util) else 0, 1),
                            "temperature": gpu.temperature
                        }
                        for i, gpu in enumerate(self.gpu_info)
                    ]
                },
                "optimization": {
                    "level": self.optimization_level.value,
                    "optimal_workers": self.optimal_workers,
                    "pytorch_threads": torch.get_num_threads(),
                    "cpu_affinity_enabled": self.cpu_affinity_enabled,
                    "monitoring_active": self.monitoring_active
                },
                "workload_profile": {
                    "avg_batch_size": round(self.workload_profile.avg_batch_size, 1),
                    "avg_sequence_length": round(self.workload_profile.avg_sequence_length, 1),
                    "avg_processing_time_ms": round(self.workload_profile.avg_processing_time * 1000, 1),
                    "memory_usage_mb": round(self.workload_profile.memory_usage_mb, 1),
                    "cpu_intensive": self.workload_profile.cpu_intensive,
                    "gpu_intensive": self.workload_profile.gpu_intensive,
                    "io_bound": self.workload_profile.io_bound
                },
                "performance_history_size": len(self.performance_history)
            }
            
        except Exception as e:
            logger.error(f"Optimization report error: {e}")
            return {"error": str(e)}
    
    def shutdown(self):
        """Shutdown optimizer and cleanup resources"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            logger.info("Hardware optimizer shut down")
            
        except Exception as e:
            logger.error(f"Optimizer shutdown error: {e}")


# Global optimizer instance
hardware_optimizer: Optional[HardwareOptimizer] = None


def get_hardware_optimizer(optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> HardwareOptimizer:
    """Get or create global hardware optimizer"""
    global hardware_optimizer
    
    if hardware_optimizer is None:
        hardware_optimizer = HardwareOptimizer(optimization_level)
    
    return hardware_optimizer


async def start_hardware_monitoring():
    """Start global hardware monitoring"""
    optimizer = get_hardware_optimizer()
    await optimizer.start_monitoring()


async def stop_hardware_monitoring():
    """Stop global hardware monitoring"""
    if hardware_optimizer:
        await hardware_optimizer.stop_monitoring()


def get_optimization_report() -> Dict[str, Any]:
    """Get global optimization report"""
    if hardware_optimizer:
        return hardware_optimizer.get_optimization_report()
    return {"error": "Hardware optimizer not initialized"}