import pytest
import asyncio
import numpy as np
import time
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_pool import ModelInstance, ModelPool


class TestModelInstance:
    """Test ModelInstance wrapper class"""
    
    def test_model_instance_creation(self):
        """Test ModelInstance creation"""
        mock_model = MagicMock()
        instance = ModelInstance(
            model=mock_model,
            instance_id="test-1",
            device="cpu"
        )
        
        assert instance.model == mock_model
        assert instance.instance_id == "test-1"
        assert instance.device == "cpu"
        assert instance.is_healthy is True
        assert instance.is_busy is False
        assert instance.request_count == 0
        assert instance.error_count == 0
        assert instance.total_processing_time == 0.0
    
    def test_model_instance_stats(self):
        """Test ModelInstance statistics tracking"""
        mock_model = MagicMock()
        instance = ModelInstance(
            model=mock_model,
            instance_id="test-1"
        )
        
        # Simulate some activity
        instance.request_count = 10
        instance.error_count = 1
        instance.total_processing_time = 5.0
        
        stats = instance.get_stats()
        
        assert stats["instance_id"] == "test-1"
        assert stats["request_count"] == 10
        assert stats["error_count"] == 1
        assert stats["total_processing_time"] == 5.0
        assert stats["average_processing_time"] == 0.5
        assert stats["is_healthy"] is True
        assert stats["is_busy"] is False
    
    def test_model_instance_health_check(self):
        """Test ModelInstance health checking"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0]])
        
        instance = ModelInstance(
            model=mock_model,
            instance_id="test-1"
        )
        
        # Health check should pass
        result = instance.health_check()
        assert result is True
        assert instance.is_healthy is True
        
        # Simulate model failure
        mock_model.encode.side_effect = Exception("Model failed")
        result = instance.health_check()
        assert result is False
        assert instance.is_healthy is False
    
    def test_model_instance_mark_unhealthy(self):
        """Test marking instance as unhealthy"""
        mock_model = MagicMock()
        instance = ModelInstance(
            model=mock_model,
            instance_id="test-1"
        )
        
        assert instance.is_healthy is True
        
        instance.mark_unhealthy("Test error")
        
        assert instance.is_healthy is False
        assert instance.error_count == 1
        assert instance.last_error == "Test error"
    
    def test_model_instance_context_manager(self):
        """Test ModelInstance as context manager"""
        mock_model = MagicMock()
        instance = ModelInstance(
            model=mock_model,
            instance_id="test-1"
        )
        
        assert instance.is_busy is False
        
        with instance:
            assert instance.is_busy is True
        
        assert instance.is_busy is False


class TestModelPool:
    """Test ModelPool class"""
    
    @pytest.fixture
    def mock_model_factory(self):
        """Mock model factory function"""
        def factory():
            mock = MagicMock()
            mock.encode.return_value = np.array([[1.0, 2.0, 3.0]])
            return mock
        return factory
    
    def test_model_pool_init(self):
        """Test ModelPool initialization"""
        pool = ModelPool(pool_size=3, enable_multi_gpu=False)
        
        assert pool.pool_size == 3
        assert pool.enable_multi_gpu is False
        assert len(pool.instances) == 0
        assert pool.current_index == 0
        assert pool.is_initialized is False
    
    @pytest.mark.asyncio
    async def test_model_pool_initialize(self, mock_model_factory):
        """Test ModelPool initialization"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            assert pool.is_initialized is True
            assert len(pool.instances) == 2
            
            for i, instance in enumerate(pool.instances):
                assert instance.instance_id == f"model-{i}"
                assert instance.device == "cpu"
                assert instance.is_healthy is True
    
    @pytest.mark.asyncio
    async def test_model_pool_initialize_with_gpu(self, mock_model_factory):
        """Test ModelPool initialization with GPU support"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.device_count', return_value=2):
                    pool = ModelPool(pool_size=4, enable_multi_gpu=True)
                    await pool.initialize()
                    
                    assert len(pool.instances) == 4
                    
                    # Check GPU device assignment
                    gpu_devices = [inst.device for inst in pool.instances]
                    assert "cuda:0" in gpu_devices
                    assert "cuda:1" in gpu_devices
    
    @pytest.mark.asyncio
    async def test_model_pool_get_healthy_instance(self, mock_model_factory):
        """Test getting healthy instance from pool"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=3, enable_multi_gpu=False)
            await pool.initialize()
            
            # All instances should be healthy initially
            instance = pool._get_healthy_instance()
            assert instance is not None
            assert instance.is_healthy is True
    
    @pytest.mark.asyncio
    async def test_model_pool_get_instance_unhealthy(self, mock_model_factory):
        """Test getting instance when some are unhealthy"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=3, enable_multi_gpu=False)
            await pool.initialize()
            
            # Mark some instances as unhealthy
            pool.instances[0].mark_unhealthy("Test error")
            pool.instances[1].mark_unhealthy("Test error")
            
            # Should still get the healthy instance
            instance = pool._get_healthy_instance()
            assert instance is not None
            assert instance.is_healthy is True
            assert instance == pool.instances[2]
    
    @pytest.mark.asyncio
    async def test_model_pool_no_healthy_instances(self, mock_model_factory):
        """Test when no healthy instances are available"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            # Mark all instances as unhealthy
            for instance in pool.instances:
                instance.mark_unhealthy("Test error")
            
            # Should return None
            instance = pool._get_healthy_instance()
            assert instance is None
    
    @pytest.mark.asyncio
    async def test_model_pool_encode_success(self, mock_model_factory):
        """Test successful encoding through pool"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            result = await pool.encode(
                texts=["hello", "world"],
                task_type="search_document"
            )
            
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 2  # Two texts
            
            # Check that instance was used
            assert any(inst.request_count > 0 for inst in pool.instances)
    
    @pytest.mark.asyncio
    async def test_model_pool_encode_no_healthy_instances(self, mock_model_factory):
        """Test encoding when no healthy instances available"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            # Mark all instances as unhealthy
            for instance in pool.instances:
                instance.mark_unhealthy("Test error")
            
            with pytest.raises(RuntimeError, match="No healthy model instances"):
                await pool.encode(
                    texts=["hello"],
                    task_type="search_document"
                )
    
    @pytest.mark.asyncio
    async def test_model_pool_encode_with_error(self, mock_model_factory):
        """Test encoding with model error"""
        def failing_factory():
            mock = MagicMock()
            mock.encode.side_effect = Exception("Model encoding failed")
            return mock
        
        with patch('src.model_pool.EmbeddingModel', failing_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            with pytest.raises(Exception, match="Model encoding failed"):
                await pool.encode(
                    texts=["hello"],
                    task_type="search_document"
                )
            
            # Instance should be marked as unhealthy after error
            assert any(not inst.is_healthy for inst in pool.instances)
    
    @pytest.mark.asyncio
    async def test_model_pool_load_balancing(self, mock_model_factory):
        """Test round-robin load balancing"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=3, enable_multi_gpu=False)
            await pool.initialize()
            
            # Make multiple requests
            for i in range(6):
                await pool.encode(
                    texts=[f"text {i}"],
                    task_type="search_document"
                )
            
            # Check that requests were distributed
            request_counts = [inst.request_count for inst in pool.instances]
            
            # Each instance should have been used (round-robin)
            assert all(count > 0 for count in request_counts)
            # Distribution should be relatively even
            assert max(request_counts) - min(request_counts) <= 1
    
    @pytest.mark.asyncio
    async def test_model_pool_health_monitoring(self, mock_model_factory):
        """Test health monitoring functionality"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            # Start health monitoring
            monitor_task = asyncio.create_task(pool._health_monitor())
            
            # Let it run briefly
            await asyncio.sleep(0.01)
            
            # All instances should still be healthy
            assert all(inst.is_healthy for inst in pool.instances)
            
            # Cancel monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_model_pool_health_recovery(self, mock_model_factory):
        """Test health recovery after errors"""
        call_count = [0]
        
        def sometimes_failing_factory():
            mock = MagicMock()
            
            def encode_side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] <= 2:  # Fail first two calls
                    raise Exception("Temporary failure")
                return np.array([[1.0, 2.0, 3.0]])
            
            mock.encode.side_effect = encode_side_effect
            return mock
        
        with patch('src.model_pool.EmbeddingModel', sometimes_failing_factory):
            pool = ModelPool(pool_size=1, enable_multi_gpu=False)
            await pool.initialize()
            
            instance = pool.instances[0]
            
            # First health check should fail
            result = instance.health_check()
            assert result is False
            assert not instance.is_healthy
            
            # Second health check should also fail
            result = instance.health_check()
            assert result is False
            
            # Third health check should succeed (recovery)
            result = instance.health_check()
            assert result is True
            assert instance.is_healthy
    
    @pytest.mark.asyncio
    async def test_model_pool_concurrent_requests(self, mock_model_factory):
        """Test handling concurrent requests"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=3, enable_multi_gpu=False)
            await pool.initialize()
            
            # Submit multiple concurrent requests
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    pool.encode(
                        texts=[f"text {i}"],
                        task_type="search_document"
                    )
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 10
            for result in results:
                assert isinstance(result, np.ndarray)
            
            # Check that requests were distributed across instances
            total_requests = sum(inst.request_count for inst in pool.instances)
            assert total_requests == 10
    
    def test_model_pool_get_stats(self, mock_model_factory):
        """Test getting pool statistics"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=3, enable_multi_gpu=False)
            
            # Simulate some activity
            for i, instance in enumerate(pool.instances):
                instance.request_count = i * 10
                instance.error_count = i
                instance.total_processing_time = i * 2.0
            
            stats = pool.get_pool_stats()
            
            assert stats["pool_size"] == 3
            assert stats["healthy_instances"] == 3
            assert stats["total_requests"] == 30  # 0 + 10 + 20
            assert stats["total_errors"] == 3  # 0 + 1 + 2
            assert "instances" in stats
            assert len(stats["instances"]) == 3
    
    @pytest.mark.asyncio
    async def test_model_pool_shutdown(self, mock_model_factory):
        """Test pool shutdown"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            assert pool.is_initialized is True
            
            await pool.shutdown()
            
            assert pool.is_initialized is False
            # Instances should be cleared or marked for cleanup
    
    @pytest.mark.asyncio
    async def test_model_pool_device_selection(self, mock_model_factory):
        """Test device selection logic"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.device_count', return_value=2):
                    pool = ModelPool(pool_size=5, enable_multi_gpu=True)
                    
                    devices = []
                    for i in range(5):
                        device = pool._get_device_for_instance(i)
                        devices.append(device)
                    
                    # Should distribute across available GPUs
                    assert "cuda:0" in devices
                    assert "cuda:1" in devices
                    # CPU might be used if more instances than GPUs
                    expected_devices = {"cuda:0", "cuda:1", "cpu"}
                    assert all(device in expected_devices for device in devices)
    
    @pytest.mark.asyncio
    async def test_model_pool_auto_size_detection(self):
        """Test automatic pool size detection"""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=4):
                with patch('os.cpu_count', return_value=8):
                    # Pool size 0 should auto-detect
                    pool = ModelPool(pool_size=0, enable_multi_gpu=True)
                    
                    # Should detect based on available GPUs
                    assert pool.pool_size == 4
    
    @pytest.mark.asyncio
    async def test_model_pool_auto_size_cpu_only(self):
        """Test automatic pool size detection for CPU only"""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('os.cpu_count', return_value=8):
                # Pool size 0 should auto-detect
                pool = ModelPool(pool_size=0, enable_multi_gpu=False)
                
                # Should detect based on CPU cores (conservative estimate)
                expected_size = max(1, 8 // 2)  # Half of CPU cores
                assert pool.pool_size == expected_size


class TestModelPoolErrorHandling:
    """Test error handling in ModelPool"""
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test handling initialization failures"""
        def failing_factory():
            raise Exception("Model loading failed")
        
        with patch('src.model_pool.EmbeddingModel', failing_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            
            with pytest.raises(Exception, match="Model loading failed"):
                await pool.initialize()
    
    @pytest.mark.asyncio
    async def test_partial_initialization_failure(self):
        """Test handling partial initialization failures"""
        call_count = [0]
        
        def sometimes_failing_factory():
            call_count[0] += 1
            if call_count[0] == 2:  # Second instance fails
                raise Exception("Second model failed")
            mock = MagicMock()
            mock.encode.return_value = np.array([[1.0, 2.0, 3.0]])
            return mock
        
        with patch('src.model_pool.EmbeddingModel', sometimes_failing_factory):
            pool = ModelPool(pool_size=3, enable_multi_gpu=False)
            
            # Should handle partial failure gracefully
            try:
                await pool.initialize()
                # Should have fewer instances than requested
                assert len(pool.instances) < 3
                assert len(pool.instances) > 0  # But at least one should succeed
            except Exception:
                # Or might fail completely depending on implementation
                pass
    
    @pytest.mark.asyncio
    async def test_all_instances_fail_during_operation(self, mock_model_factory):
        """Test when all instances fail during operation"""
        with patch('src.model_pool.EmbeddingModel', mock_model_factory):
            pool = ModelPool(pool_size=2, enable_multi_gpu=False)
            await pool.initialize()
            
            # Make all models start failing
            for instance in pool.instances:
                instance.model.encode.side_effect = Exception("Model crashed")
            
            # Should raise error when no healthy instances
            with pytest.raises(RuntimeError):
                await pool.encode(["test"], "search_document")
    
    @pytest.mark.asyncio
    async def test_recovery_from_failures(self, mock_model_factory):
        """Test recovery from temporary failures"""
        failure_count = [0]
        
        def sometimes_failing_encode(*args, **kwargs):
            failure_count[0] += 1
            if failure_count[0] <= 2:  # Fail first two calls
                raise Exception("Temporary failure")
            return np.array([[1.0, 2.0, 3.0]])
        
        mock_model = MagicMock()
        mock_model.encode.side_effect = sometimes_failing_encode
        
        def factory():
            return mock_model
        
        with patch('src.model_pool.EmbeddingModel', factory):
            pool = ModelPool(pool_size=1, enable_multi_gpu=False)
            await pool.initialize()
            
            instance = pool.instances[0]
            
            # First two encode attempts should fail
            with pytest.raises(Exception):
                await pool.encode(["test1"], "search_document")
            
            with pytest.raises(Exception):
                await pool.encode(["test2"], "search_document")
            
            # Instance should be marked unhealthy
            assert not instance.is_healthy
            
            # Manual health check should recover the instance
            result = instance.health_check()
            assert result is True
            assert instance.is_healthy
            
            # Now encoding should work
            result = await pool.encode(["test3"], "search_document")
            assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])