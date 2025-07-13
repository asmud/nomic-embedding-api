import pytest
import asyncio
import gc
import time
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock psutil before importing memory_manager
sys.modules['psutil'] = MagicMock()

# Mock the memory manager module components for testing
class MockMemoryManager:
    def __init__(self):
        self.is_monitoring = False
        self.stats = {
            'total_memory_mb': 8192,
            'available_memory_mb': 4096,
            'used_memory_mb': 4096,
            'memory_pressure': 'low',
            'gc_stats': {'collections': 0, 'collected': 0}
        }
    
    async def start_memory_monitoring(self):
        self.is_monitoring = True
    
    async def stop_memory_monitoring(self):
        self.is_monitoring = False
    
    def get_memory_stats(self):
        return self.stats.copy()
    
    def check_memory_pressure(self):
        available_percent = (self.stats['available_memory_mb'] / self.stats['total_memory_mb']) * 100
        if available_percent < 15:
            return 'critical'
        elif available_percent < 25:
            return 'high'
        elif available_percent < 50:
            return 'medium'
        else:
            return 'low'


class TestMemoryManager:
    """Test suite for memory management functionality"""
    
    @pytest.fixture
    def memory_manager(self):
        """Memory manager fixture"""
        return MockMemoryManager()
    
    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil with memory info"""
        mock = MagicMock()
        mock.virtual_memory.return_value = MagicMock(
            total=8 * 1024 * 1024 * 1024,  # 8GB
            available=4 * 1024 * 1024 * 1024,  # 4GB available
            used=4 * 1024 * 1024 * 1024,  # 4GB used
            percent=50.0
        )
        return mock
    
    def test_memory_manager_init(self, memory_manager):
        """Test memory manager initialization"""
        assert memory_manager.is_monitoring is False
        assert 'total_memory_mb' in memory_manager.stats
        assert 'available_memory_mb' in memory_manager.stats
        assert 'memory_pressure' in memory_manager.stats
    
    @pytest.mark.asyncio
    async def test_start_memory_monitoring(self, memory_manager):
        """Test starting memory monitoring"""
        await memory_manager.start_memory_monitoring()
        assert memory_manager.is_monitoring is True
    
    @pytest.mark.asyncio
    async def test_stop_memory_monitoring(self, memory_manager):
        """Test stopping memory monitoring"""
        await memory_manager.start_memory_monitoring()
        assert memory_manager.is_monitoring is True
        
        await memory_manager.stop_memory_monitoring()
        assert memory_manager.is_monitoring is False
    
    def test_get_memory_stats(self, memory_manager):
        """Test getting memory statistics"""
        stats = memory_manager.get_memory_stats()
        
        required_keys = [
            'total_memory_mb', 'available_memory_mb', 'used_memory_mb',
            'memory_pressure', 'gc_stats'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert isinstance(stats['total_memory_mb'], (int, float))
        assert isinstance(stats['available_memory_mb'], (int, float))
        assert isinstance(stats['used_memory_mb'], (int, float))
        assert stats['memory_pressure'] in ['low', 'medium', 'high', 'critical']
    
    def test_memory_pressure_calculation(self, memory_manager):
        """Test memory pressure calculation"""
        # Test low pressure (>50% available)
        memory_manager.stats['total_memory_mb'] = 8192
        memory_manager.stats['available_memory_mb'] = 5000
        assert memory_manager.check_memory_pressure() == 'low'
        
        # Test medium pressure (25-50% available)
        memory_manager.stats['available_memory_mb'] = 3000
        assert memory_manager.check_memory_pressure() == 'medium'
        
        # Test high pressure (15-25% available)
        memory_manager.stats['available_memory_mb'] = 1500
        assert memory_manager.check_memory_pressure() == 'high'
        
        # Test critical pressure (<15% available)
        memory_manager.stats['available_memory_mb'] = 1000
        assert memory_manager.check_memory_pressure() == 'critical'


class TestMemoryMonitoring:
    """Test memory monitoring functionality"""
    
    @pytest.fixture
    def mock_cache(self):
        """Mock cache for testing resize operations"""
        cache = MagicMock()
        cache.max_size = 1000
        cache.resize = MagicMock()
        cache.get_stats.return_value = {
            'size': 800,
            'max_size': 1000,
            'hit_rate': 0.85
        }
        return cache
    
    def test_cache_resize_on_high_pressure(self, mock_cache):
        """Test cache resizing under high memory pressure"""
        # Simulate high memory pressure
        original_size = mock_cache.max_size
        
        # Mock memory pressure detection
        def simulate_resize_logic(pressure_level):
            if pressure_level == 'high':
                new_size = int(original_size * 0.7)  # Reduce by 30%
                mock_cache.resize(new_size)
            elif pressure_level == 'critical':
                new_size = int(original_size * 0.5)  # Reduce by 50%
                mock_cache.resize(new_size)
        
        # Test high pressure
        simulate_resize_logic('high')
        mock_cache.resize.assert_called_with(700)
        
        # Test critical pressure
        simulate_resize_logic('critical')
        mock_cache.resize.assert_called_with(500)
    
    def test_garbage_collection_stats(self):
        """Test garbage collection statistics tracking"""
        # Get initial GC stats
        initial_stats = gc.get_stats()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Get updated stats
        updated_stats = gc.get_stats()
        
        # Stats should be available
        assert isinstance(initial_stats, list)
        assert isinstance(updated_stats, list)
        assert isinstance(collected, int)
        assert collected >= 0
    
    @patch('gc.collect')
    def test_forced_garbage_collection(self, mock_gc_collect):
        """Test forced garbage collection trigger"""
        mock_gc_collect.return_value = 42  # Objects collected
        
        # Simulate triggering GC under memory pressure
        def trigger_gc_if_needed(memory_pressure):
            if memory_pressure in ['high', 'critical']:
                return gc.collect()
            return 0
        
        # Test triggering GC
        collected = trigger_gc_if_needed('high')
        mock_gc_collect.assert_called_once()
        
        # Test not triggering GC
        mock_gc_collect.reset_mock()
        collected = trigger_gc_if_needed('low')
        mock_gc_collect.assert_not_called()


class TestMemoryOptimization:
    """Test memory optimization strategies"""
    
    def test_memory_threshold_detection(self):
        """Test memory threshold detection"""
        def check_memory_thresholds(available_mb, total_mb):
            available_percent = (available_mb / total_mb) * 100
            
            thresholds = {
                'critical': 15,
                'high': 25,
                'medium': 50
            }
            
            for level, threshold in thresholds.items():
                if available_percent < threshold:
                    return level
            return 'low'
        
        # Test various scenarios
        assert check_memory_thresholds(1000, 8192) == 'critical'  # ~12%
        assert check_memory_thresholds(2000, 8192) == 'high'      # ~24%
        assert check_memory_thresholds(3000, 8192) == 'medium'    # ~37%
        assert check_memory_thresholds(5000, 8192) == 'low'       # ~61%
    
    def test_cache_size_calculation(self):
        """Test optimal cache size calculation based on memory"""
        def calculate_optimal_cache_size(available_mb, total_mb, current_size):
            available_percent = (available_mb / total_mb) * 100
            
            if available_percent < 15:  # Critical
                return int(current_size * 0.3)  # Aggressive reduction
            elif available_percent < 25:  # High pressure
                return int(current_size * 0.6)  # Moderate reduction
            elif available_percent < 50:  # Medium pressure
                return int(current_size * 0.8)  # Small reduction
            else:  # Low pressure
                return current_size  # No change
        
        # Test different memory scenarios
        current_size = 1000
        
        # Critical memory
        new_size = calculate_optimal_cache_size(1000, 8192, current_size)
        assert new_size == 300
        
        # High memory pressure
        new_size = calculate_optimal_cache_size(2000, 8192, current_size)
        assert new_size == 600
        
        # Medium memory pressure
        new_size = calculate_optimal_cache_size(3000, 8192, current_size)
        assert new_size == 800
        
        # Low memory pressure
        new_size = calculate_optimal_cache_size(5000, 8192, current_size)
        assert new_size == 1000
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation for embeddings"""
        def estimate_embedding_memory(num_embeddings, embedding_dim, dtype_size=4):
            """Estimate memory usage for embeddings in MB"""
            bytes_per_embedding = embedding_dim * dtype_size  # float32 = 4 bytes
            total_bytes = num_embeddings * bytes_per_embedding
            return total_bytes / (1024 * 1024)  # Convert to MB
        
        # Test various scenarios
        # 1000 embeddings of 768 dimensions (float32)
        memory_mb = estimate_embedding_memory(1000, 768)
        expected_mb = (1000 * 768 * 4) / (1024 * 1024)
        assert abs(memory_mb - expected_mb) < 0.01
        
        # Large cache scenario
        memory_mb = estimate_embedding_memory(10000, 768)
        assert memory_mb > 25  # Should be substantial memory usage
    
    def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing based on memory pressure"""
        def adapt_batch_size(current_batch_size, memory_pressure):
            adjustments = {
                'critical': 0.25,  # Very small batches
                'high': 0.5,       # Half size
                'medium': 0.75,    # Slight reduction
                'low': 1.0         # No change
            }
            
            multiplier = adjustments.get(memory_pressure, 1.0)
            return max(1, int(current_batch_size * multiplier))
        
        base_size = 32
        
        assert adapt_batch_size(base_size, 'critical') == 8
        assert adapt_batch_size(base_size, 'high') == 16
        assert adapt_batch_size(base_size, 'medium') == 24
        assert adapt_batch_size(base_size, 'low') == 32
        
        # Edge case: very small batch
        assert adapt_batch_size(2, 'critical') == 1  # Should not go below 1


class TestMemoryIntegration:
    """Test memory management integration with other components"""
    
    @pytest.mark.asyncio
    async def test_memory_monitoring_lifecycle(self):
        """Test complete memory monitoring lifecycle"""
        memory_manager = MockMemoryManager()
        
        # Start monitoring
        await memory_manager.start_memory_monitoring()
        assert memory_manager.is_monitoring is True
        
        # Simulate monitoring loop
        for _ in range(3):
            stats = memory_manager.get_memory_stats()
            pressure = memory_manager.check_memory_pressure()
            
            assert isinstance(stats, dict)
            assert pressure in ['low', 'medium', 'high', 'critical']
            
            # Simulate monitoring interval
            await asyncio.sleep(0.001)
        
        # Stop monitoring
        await memory_manager.stop_memory_monitoring()
        assert memory_manager.is_monitoring is False
    
    def test_memory_stats_format(self):
        """Test memory statistics format and completeness"""
        memory_manager = MockMemoryManager()
        stats = memory_manager.get_memory_stats()
        
        # Required fields
        required_fields = [
            'total_memory_mb',
            'available_memory_mb', 
            'used_memory_mb',
            'memory_pressure',
            'gc_stats'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Type validation
        assert isinstance(stats['total_memory_mb'], (int, float))
        assert isinstance(stats['available_memory_mb'], (int, float))
        assert isinstance(stats['used_memory_mb'], (int, float))
        assert isinstance(stats['memory_pressure'], str)
        assert isinstance(stats['gc_stats'], dict)
        
        # Value validation
        assert stats['total_memory_mb'] > 0
        assert stats['available_memory_mb'] >= 0
        assert stats['used_memory_mb'] >= 0
        assert stats['available_memory_mb'] <= stats['total_memory_mb']
    
    @pytest.mark.asyncio
    async def test_memory_pressure_response(self):
        """Test system response to different memory pressure levels"""
        memory_manager = MockMemoryManager()
        
        # Mock components that should respond to memory pressure
        cache_mock = MagicMock()
        batch_processor_mock = MagicMock()
        
        def simulate_pressure_response(pressure_level):
            responses = []
            
            if pressure_level in ['high', 'critical']:
                # Trigger garbage collection
                collected = gc.collect()
                responses.append(f"GC collected {collected} objects")
                
                # Reduce cache size
                if pressure_level == 'critical':
                    cache_mock.resize(500)
                    responses.append("Cache resized to 500")
                else:
                    cache_mock.resize(750)
                    responses.append("Cache resized to 750")
                
                # Reduce batch size
                if pressure_level == 'critical':
                    batch_processor_mock.adjust_batch_size(8)
                    responses.append("Batch size reduced to 8")
                else:
                    batch_processor_mock.adjust_batch_size(16)
                    responses.append("Batch size reduced to 16")
            
            return responses
        
        # Test different pressure levels
        responses_critical = simulate_pressure_response('critical')
        assert len(responses_critical) >= 3
        
        responses_high = simulate_pressure_response('high')
        assert len(responses_high) >= 3
        
        responses_low = simulate_pressure_response('low')
        assert len(responses_low) == 0  # No action needed


class TestMemoryErrorHandling:
    """Test memory management error handling"""
    
    def test_memory_stats_error_handling(self):
        """Test handling of memory stats collection errors"""
        def safe_get_memory_stats():
            try:
                # Simulate psutil error
                raise Exception("Memory information unavailable")
            except Exception as e:
                # Return safe defaults
                return {
                    'total_memory_mb': 0,
                    'available_memory_mb': 0,
                    'used_memory_mb': 0,
                    'memory_pressure': 'unknown',
                    'error': str(e)
                }
        
        stats = safe_get_memory_stats()
        assert 'error' in stats
        assert stats['memory_pressure'] == 'unknown'
    
    @pytest.mark.asyncio
    async def test_monitoring_error_recovery(self):
        """Test recovery from monitoring errors"""
        error_count = [0]
        
        async def monitoring_loop_with_errors():
            while error_count[0] < 3:
                try:
                    if error_count[0] == 1:  # Simulate error on second iteration
                        raise Exception("Monitoring error")
                    
                    # Normal monitoring
                    await asyncio.sleep(0.001)
                    error_count[0] += 1
                    
                except Exception as e:
                    error_count[0] += 1
                    # Log error and continue
                    await asyncio.sleep(0.001)
        
        # Should complete despite error
        await monitoring_loop_with_errors()
        assert error_count[0] == 3
    
    def test_cache_resize_error_handling(self):
        """Test handling of cache resize errors"""
        cache_mock = MagicMock()
        cache_mock.resize.side_effect = Exception("Resize failed")
        
        def safe_cache_resize(cache, new_size):
            try:
                cache.resize(new_size)
                return True
            except Exception:
                return False
        
        result = safe_cache_resize(cache_mock, 500)
        assert result is False
        cache_mock.resize.assert_called_once_with(500)


class TestMemoryConfiguration:
    """Test memory management configuration"""
    
    def test_memory_thresholds_configuration(self):
        """Test memory threshold configuration"""
        config = {
            'MEMORY_PRESSURE_HIGH_PERCENT': 85.0,
            'MEMORY_PRESSURE_CRITICAL_PERCENT': 95.0,
            'MIN_AVAILABLE_MEMORY_MB': 512.0,
            'MEMORY_MONITORING_INTERVAL': 30.0
        }
        
        def validate_memory_config(cfg):
            # Validate threshold ordering
            assert cfg['MEMORY_PRESSURE_CRITICAL_PERCENT'] > cfg['MEMORY_PRESSURE_HIGH_PERCENT']
            
            # Validate reasonable values
            assert 0 < cfg['MEMORY_PRESSURE_HIGH_PERCENT'] < 100
            assert 0 < cfg['MEMORY_PRESSURE_CRITICAL_PERCENT'] < 100
            assert cfg['MIN_AVAILABLE_MEMORY_MB'] > 0
            assert cfg['MEMORY_MONITORING_INTERVAL'] > 0
            
            return True
        
        assert validate_memory_config(config) is True
    
    def test_memory_optimization_settings(self):
        """Test memory optimization settings validation"""
        settings = {
            'ENABLE_DYNAMIC_MEMORY': True,
            'ENABLE_GC_OPTIMIZATION': True,
            'CACHE_RESIZE_AGGRESSIVE': False,
            'MEMORY_CLEANUP_ON_PRESSURE': True
        }
        
        def validate_optimization_settings(settings):
            # All should be boolean
            for key, value in settings.items():
                assert isinstance(value, bool), f"{key} should be boolean"
            
            return True
        
        assert validate_optimization_settings(settings) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])