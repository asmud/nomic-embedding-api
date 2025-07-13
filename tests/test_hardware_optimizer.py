import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock hardware detection modules
sys.modules['py_cpuinfo'] = MagicMock()
sys.modules['pynvml'] = MagicMock()

# Mock hardware optimizer components for testing
class MockOptimizationLevel:
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class MockHardwareOptimizer:
    def __init__(self, optimization_level=MockOptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.is_monitoring = False
        self.device_info = {
            'cpu_count': 8,
            'gpu_count': 1,
            'total_memory_gb': 16,
            'gpu_memory_gb': 8
        }
        self.workload_profile = {
            'avg_batch_size': 16,
            'avg_sequence_length': 128,
            'avg_processing_time': 0.1,
            'memory_usage_mb': 256
        }
        self.optimization_history = []
    
    def get_optimal_device(self):
        if self.device_info['gpu_count'] > 0:
            return "cuda:0"
        return "cpu"
    
    def update_workload_profile(self, batch_size, sequence_length, processing_time, memory_usage):
        self.workload_profile.update({
            'avg_batch_size': batch_size,
            'avg_sequence_length': sequence_length,
            'avg_processing_time': processing_time,
            'memory_usage_mb': memory_usage
        })
    
    def get_optimization_report(self):
        return {
            'optimization_level': self.optimization_level,
            'device_info': self.device_info,
            'workload_profile': self.workload_profile,
            'recommendations': self._generate_recommendations(),
            'optimization_history': self.optimization_history
        }
    
    def _generate_recommendations(self):
        recommendations = []
        
        if self.workload_profile['avg_processing_time'] > 0.5:
            recommendations.append("Consider using GPU acceleration")
        
        if self.workload_profile['memory_usage_mb'] > 1000:
            recommendations.append("Consider reducing batch size")
        
        if self.workload_profile['avg_batch_size'] < 8:
            recommendations.append("Consider increasing batch size for better throughput")
        
        return recommendations
    
    async def start_hardware_monitoring(self):
        self.is_monitoring = True
    
    async def stop_hardware_monitoring(self):
        self.is_monitoring = False


class TestOptimizationLevel:
    """Test optimization level enum"""
    
    def test_optimization_levels(self):
        """Test optimization level constants"""
        assert MockOptimizationLevel.CONSERVATIVE == "conservative"
        assert MockOptimizationLevel.BALANCED == "balanced"
        assert MockOptimizationLevel.AGGRESSIVE == "aggressive"


class TestHardwareOptimizer:
    """Test hardware optimizer functionality"""
    
    @pytest.fixture
    def hardware_optimizer(self):
        """Hardware optimizer fixture"""
        return MockHardwareOptimizer()
    
    def test_hardware_optimizer_init(self, hardware_optimizer):
        """Test hardware optimizer initialization"""
        assert hardware_optimizer.optimization_level == MockOptimizationLevel.BALANCED
        assert hardware_optimizer.is_monitoring is False
        assert 'cpu_count' in hardware_optimizer.device_info
        assert 'gpu_count' in hardware_optimizer.device_info
        assert hardware_optimizer.workload_profile is not None
    
    def test_device_selection_with_gpu(self):
        """Test optimal device selection with GPU available"""
        optimizer = MockHardwareOptimizer()
        optimizer.device_info['gpu_count'] = 2
        
        device = optimizer.get_optimal_device()
        assert device == "cuda:0"
    
    def test_device_selection_cpu_only(self):
        """Test optimal device selection with CPU only"""
        optimizer = MockHardwareOptimizer()
        optimizer.device_info['gpu_count'] = 0
        
        device = optimizer.get_optimal_device()
        assert device == "cpu"
    
    def test_workload_profile_update(self, hardware_optimizer):
        """Test updating workload profile"""
        hardware_optimizer.update_workload_profile(
            batch_size=32,
            sequence_length=256,
            processing_time=0.2,
            memory_usage=512
        )
        
        profile = hardware_optimizer.workload_profile
        assert profile['avg_batch_size'] == 32
        assert profile['avg_sequence_length'] == 256
        assert profile['avg_processing_time'] == 0.2
        assert profile['memory_usage_mb'] == 512
    
    def test_optimization_report_structure(self, hardware_optimizer):
        """Test optimization report structure"""
        report = hardware_optimizer.get_optimization_report()
        
        required_keys = [
            'optimization_level',
            'device_info',
            'workload_profile',
            'recommendations',
            'optimization_history'
        ]
        
        for key in required_keys:
            assert key in report
        
        assert isinstance(report['recommendations'], list)
        assert isinstance(report['optimization_history'], list)
    
    def test_recommendations_generation(self, hardware_optimizer):
        """Test recommendation generation based on workload"""
        # Set up scenario with slow processing
        hardware_optimizer.update_workload_profile(
            batch_size=16,
            sequence_length=128,
            processing_time=0.8,  # Slow
            memory_usage=256
        )
        
        report = hardware_optimizer.get_optimization_report()
        recommendations = report['recommendations']
        
        # Should recommend GPU acceleration for slow processing
        gpu_rec = any("GPU" in rec for rec in recommendations)
        assert gpu_rec is True
    
    def test_recommendations_high_memory(self, hardware_optimizer):
        """Test recommendations for high memory usage"""
        hardware_optimizer.update_workload_profile(
            batch_size=32,
            sequence_length=256,
            processing_time=0.1,
            memory_usage=1500  # High memory
        )
        
        report = hardware_optimizer.get_optimization_report()
        recommendations = report['recommendations']
        
        # Should recommend reducing batch size
        batch_rec = any("batch size" in rec and "reducing" in rec for rec in recommendations)
        assert batch_rec is True
    
    def test_recommendations_small_batch(self, hardware_optimizer):
        """Test recommendations for small batch sizes"""
        hardware_optimizer.update_workload_profile(
            batch_size=4,  # Small batch
            sequence_length=128,
            processing_time=0.1,
            memory_usage=128
        )
        
        report = hardware_optimizer.get_optimization_report()
        recommendations = report['recommendations']
        
        # Should recommend increasing batch size
        batch_rec = any("increasing" in rec and "batch size" in rec for rec in recommendations)
        assert batch_rec is True
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, hardware_optimizer):
        """Test hardware monitoring lifecycle"""
        assert hardware_optimizer.is_monitoring is False
        
        await hardware_optimizer.start_hardware_monitoring()
        assert hardware_optimizer.is_monitoring is True
        
        await hardware_optimizer.stop_hardware_monitoring()
        assert hardware_optimizer.is_monitoring is False


class TestHardwareDetection:
    """Test hardware detection functionality"""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_gpu_detection(self, mock_device_count, mock_cuda_available):
        """Test GPU detection"""
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 2
        
        def detect_gpu_info():
            if mock_cuda_available():
                return {
                    'available': True,
                    'count': mock_device_count(),
                    'devices': [f"cuda:{i}" for i in range(mock_device_count())]
                }
            return {'available': False, 'count': 0, 'devices': []}
        
        gpu_info = detect_gpu_info()
        assert gpu_info['available'] is True
        assert gpu_info['count'] == 2
        assert gpu_info['devices'] == ["cuda:0", "cuda:1"]
    
    @patch('os.cpu_count')
    def test_cpu_detection(self, mock_cpu_count):
        """Test CPU detection"""
        mock_cpu_count.return_value = 8
        
        def detect_cpu_info():
            return {
                'count': mock_cpu_count(),
                'architecture': 'x86_64'  # Mock architecture
            }
        
        cpu_info = detect_cpu_info()
        assert cpu_info['count'] == 8
        assert 'architecture' in cpu_info
    
    def test_memory_detection(self):
        """Test memory detection simulation"""
        def detect_memory_info():
            # Simulate memory detection
            return {
                'total_gb': 16,
                'available_gb': 12,
                'type': 'DDR4'
            }
        
        memory_info = detect_memory_info()
        assert memory_info['total_gb'] > 0
        assert memory_info['available_gb'] <= memory_info['total_gb']


class TestPerformanceOptimization:
    """Test performance optimization strategies"""
    
    def test_batch_size_optimization(self):
        """Test batch size optimization based on hardware"""
        def optimize_batch_size(device_type, memory_gb, optimization_level):
            base_batch_size = 16
            
            # Adjust based on device
            if device_type == "cuda":
                multiplier = 2.0  # GPUs can handle larger batches
            else:
                multiplier = 1.0
            
            # Adjust based on memory
            if memory_gb >= 16:
                memory_multiplier = 1.5
            elif memory_gb >= 8:
                memory_multiplier = 1.0
            else:
                memory_multiplier = 0.5
            
            # Adjust based on optimization level
            level_multipliers = {
                "conservative": 0.75,
                "balanced": 1.0,
                "aggressive": 1.25
            }
            
            level_multiplier = level_multipliers.get(optimization_level, 1.0)
            
            optimal_size = int(base_batch_size * multiplier * memory_multiplier * level_multiplier)
            return max(1, min(optimal_size, 128))  # Clamp between 1 and 128
        
        # Test various scenarios
        gpu_high_mem_aggressive = optimize_batch_size("cuda", 32, "aggressive")
        assert gpu_high_mem_aggressive > 32
        
        cpu_low_mem_conservative = optimize_batch_size("cpu", 4, "conservative")
        assert cpu_low_mem_conservative <= 16
        
        balanced_scenario = optimize_batch_size("cuda", 16, "balanced")
        assert 16 <= balanced_scenario <= 64
    
    def test_thread_optimization(self):
        """Test thread count optimization"""
        def optimize_thread_count(cpu_count, workload_type, optimization_level):
            if workload_type == "cpu_intensive":
                base_ratio = 1.0  # One thread per core
            else:
                base_ratio = 2.0  # More threads for I/O bound work
            
            level_adjustments = {
                "conservative": 0.75,
                "balanced": 1.0,
                "aggressive": 1.5
            }
            
            adjustment = level_adjustments.get(optimization_level, 1.0)
            optimal_threads = int(cpu_count * base_ratio * adjustment)
            
            return max(1, min(optimal_threads, cpu_count * 4))  # Reasonable bounds
        
        # Test scenarios
        cpu_intensive = optimize_thread_count(8, "cpu_intensive", "balanced")
        assert cpu_intensive == 8
        
        io_bound = optimize_thread_count(8, "io_bound", "aggressive")
        assert io_bound > 8
        
        conservative = optimize_thread_count(8, "cpu_intensive", "conservative")
        assert conservative < 8
    
    def test_memory_optimization(self):
        """Test memory optimization strategies"""
        def optimize_memory_usage(total_memory_gb, current_usage_gb, optimization_level):
            # Calculate safe memory limits
            safe_limits = {
                "conservative": 0.6,  # Use only 60% of memory
                "balanced": 0.75,     # Use 75% of memory
                "aggressive": 0.9     # Use 90% of memory
            }
            
            limit_ratio = safe_limits.get(optimization_level, 0.75)
            max_safe_usage = total_memory_gb * limit_ratio
            
            if current_usage_gb > max_safe_usage:
                return {
                    "action": "reduce_usage",
                    "target_gb": max_safe_usage,
                    "reduction_needed_gb": current_usage_gb - max_safe_usage
                }
            else:
                return {
                    "action": "maintain",
                    "headroom_gb": max_safe_usage - current_usage_gb
                }
        
        # Test over-usage scenario
        result = optimize_memory_usage(16, 14, "conservative")
        assert result["action"] == "reduce_usage"
        
        # Test safe usage scenario
        result = optimize_memory_usage(16, 8, "balanced")
        assert result["action"] == "maintain"
        assert result["headroom_gb"] > 0


class TestWorkloadProfiling:
    """Test workload profiling and analysis"""
    
    def test_workload_classification(self):
        """Test workload classification based on metrics"""
        def classify_workload(avg_processing_time, memory_usage_mb, batch_size):
            if avg_processing_time > 1.0:
                compute_intensity = "high"
            elif avg_processing_time > 0.1:
                compute_intensity = "medium"
            else:
                compute_intensity = "low"
            
            if memory_usage_mb > 1000:
                memory_intensity = "high"
            elif memory_usage_mb > 500:
                memory_intensity = "medium"
            else:
                memory_intensity = "low"
            
            if batch_size > 32:
                throughput_demand = "high"
            elif batch_size > 8:
                throughput_demand = "medium"
            else:
                throughput_demand = "low"
            
            return {
                "compute_intensity": compute_intensity,
                "memory_intensity": memory_intensity,
                "throughput_demand": throughput_demand
            }
        
        # Test high-intensity workload
        high_workload = classify_workload(2.0, 1500, 64)
        assert high_workload["compute_intensity"] == "high"
        assert high_workload["memory_intensity"] == "high"
        assert high_workload["throughput_demand"] == "high"
        
        # Test low-intensity workload
        low_workload = classify_workload(0.05, 200, 4)
        assert low_workload["compute_intensity"] == "low"
        assert low_workload["memory_intensity"] == "low"
        assert low_workload["throughput_demand"] == "low"
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis"""
        def analyze_performance_trend(history):
            if len(history) < 2:
                return "insufficient_data"
            
            recent_times = [entry["processing_time"] for entry in history[-5:]]
            avg_recent = sum(recent_times) / len(recent_times)
            
            older_times = [entry["processing_time"] for entry in history[-10:-5]]
            if not older_times:
                return "stable"
            
            avg_older = sum(older_times) / len(older_times)
            
            change_ratio = avg_recent / avg_older
            
            if change_ratio > 1.1:
                return "degrading"
            elif change_ratio < 0.9:
                return "improving"
            else:
                return "stable"
        
        # Test improving trend
        improving_history = [
            {"processing_time": 1.0, "timestamp": i} for i in range(5)
        ] + [
            {"processing_time": 0.8, "timestamp": i + 5} for i in range(5)
        ]
        
        trend = analyze_performance_trend(improving_history)
        assert trend == "improving"
        
        # Test degrading trend
        degrading_history = [
            {"processing_time": 0.8, "timestamp": i} for i in range(5)
        ] + [
            {"processing_time": 1.2, "timestamp": i + 5} for i in range(5)
        ]
        
        trend = analyze_performance_trend(degrading_history)
        assert trend == "degrading"


class TestOptimizationStrategies:
    """Test different optimization strategies"""
    
    def test_conservative_optimization(self):
        """Test conservative optimization strategy"""
        def conservative_optimize(current_config):
            optimized = current_config.copy()
            
            # Conservative changes - small, safe adjustments
            if optimized["batch_size"] > 8:
                optimized["batch_size"] = max(8, optimized["batch_size"] - 4)
            
            if optimized["num_workers"] > 2:
                optimized["num_workers"] = max(2, optimized["num_workers"] - 1)
            
            # Don't enable aggressive features
            optimized["use_mixed_precision"] = False
            optimized["use_torch_compile"] = False
            
            return optimized
        
        config = {
            "batch_size": 32,
            "num_workers": 8,
            "use_mixed_precision": True,
            "use_torch_compile": True
        }
        
        optimized = conservative_optimize(config)
        assert optimized["batch_size"] <= 28
        assert optimized["num_workers"] <= 7
        assert optimized["use_mixed_precision"] is False
    
    def test_aggressive_optimization(self):
        """Test aggressive optimization strategy"""
        def aggressive_optimize(current_config, hardware_info):
            optimized = current_config.copy()
            
            # Aggressive changes - maximize performance
            if hardware_info["gpu_available"]:
                optimized["batch_size"] = min(128, optimized["batch_size"] * 2)
                optimized["use_mixed_precision"] = True
                optimized["use_torch_compile"] = True
            
            optimized["num_workers"] = min(
                hardware_info["cpu_count"],
                optimized["num_workers"] * 2
            )
            
            return optimized
        
        config = {
            "batch_size": 16,
            "num_workers": 4,
            "use_mixed_precision": False,
            "use_torch_compile": False
        }
        
        hardware = {
            "gpu_available": True,
            "cpu_count": 16
        }
        
        optimized = aggressive_optimize(config, hardware)
        assert optimized["batch_size"] >= 32
        assert optimized["num_workers"] >= 8
        assert optimized["use_mixed_precision"] is True
    
    def test_adaptive_optimization(self):
        """Test adaptive optimization based on performance feedback"""
        def adaptive_optimize(current_config, performance_history):
            if not performance_history:
                return current_config
            
            optimized = current_config.copy()
            recent_performance = performance_history[-3:]
            avg_time = sum(p["processing_time"] for p in recent_performance) / len(recent_performance)
            
            # Adapt based on recent performance
            if avg_time > 0.5:  # Too slow
                if optimized["batch_size"] > 4:
                    optimized["batch_size"] //= 2
                optimized["num_workers"] = min(8, optimized["num_workers"] + 1)
            elif avg_time < 0.1:  # Fast enough, can push harder
                optimized["batch_size"] = min(64, optimized["batch_size"] * 2)
            
            return optimized
        
        config = {"batch_size": 16, "num_workers": 4}
        
        # Test with slow performance
        slow_history = [{"processing_time": 0.8} for _ in range(3)]
        optimized = adaptive_optimize(config, slow_history)
        assert optimized["batch_size"] <= 8
        
        # Test with fast performance
        fast_history = [{"processing_time": 0.05} for _ in range(3)]
        optimized = adaptive_optimize(config, fast_history)
        assert optimized["batch_size"] >= 32


class TestHardwareMonitoring:
    """Test hardware monitoring functionality"""
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self):
        """Test hardware monitoring loop"""
        optimizer = MockHardwareOptimizer()
        monitoring_data = []
        
        async def mock_monitoring_loop():
            await optimizer.start_hardware_monitoring()
            
            for i in range(3):
                # Simulate monitoring data collection
                data = {
                    "timestamp": time.time(),
                    "cpu_usage": 50 + i * 10,
                    "memory_usage": 60 + i * 5,
                    "gpu_usage": 30 + i * 15
                }
                monitoring_data.append(data)
                await asyncio.sleep(0.001)
            
            await optimizer.stop_hardware_monitoring()
        
        await mock_monitoring_loop()
        
        assert len(monitoring_data) == 3
        assert all("timestamp" in data for data in monitoring_data)
        assert optimizer.is_monitoring is False
    
    def test_performance_alerting(self):
        """Test performance alerting system"""
        def check_performance_alerts(metrics):
            alerts = []
            
            if metrics.get("cpu_usage", 0) > 90:
                alerts.append("High CPU usage detected")
            
            if metrics.get("memory_usage", 0) > 85:
                alerts.append("High memory usage detected")
            
            if metrics.get("gpu_usage", 0) > 95:
                alerts.append("High GPU usage detected")
            
            if metrics.get("processing_time", 0) > 2.0:
                alerts.append("Slow processing time detected")
            
            return alerts
        
        # Test normal metrics
        normal_metrics = {
            "cpu_usage": 50,
            "memory_usage": 60,
            "gpu_usage": 70,
            "processing_time": 0.2
        }
        alerts = check_performance_alerts(normal_metrics)
        assert len(alerts) == 0
        
        # Test high usage metrics
        high_metrics = {
            "cpu_usage": 95,
            "memory_usage": 90,
            "gpu_usage": 98,
            "processing_time": 3.0
        }
        alerts = check_performance_alerts(high_metrics)
        assert len(alerts) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])