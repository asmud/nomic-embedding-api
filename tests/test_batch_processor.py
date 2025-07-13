import pytest
import asyncio
import numpy as np
import time
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_processor import (
    RequestPriority, 
    EmbeddingRequest, 
    PriorityRequest, 
    BatchProcessor
)


class TestRequestPriority:
    """Test RequestPriority enum"""
    
    def test_priority_values(self):
        """Test priority enum values"""
        assert RequestPriority.URGENT.value == 0
        assert RequestPriority.HIGH.value == 1
        assert RequestPriority.NORMAL.value == 2
        assert RequestPriority.LOW.value == 3
    
    def test_priority_ordering(self):
        """Test priority ordering (lower value = higher priority)"""
        assert RequestPriority.URGENT.value < RequestPriority.HIGH.value
        assert RequestPriority.HIGH.value < RequestPriority.NORMAL.value
        assert RequestPriority.NORMAL.value < RequestPriority.LOW.value


class TestEmbeddingRequest:
    """Test EmbeddingRequest dataclass"""
    
    def test_embedding_request_creation(self):
        """Test EmbeddingRequest creation"""
        future = asyncio.Future()
        request = EmbeddingRequest(
            texts=["hello", "world"],
            model="test-model",
            task_type="search_document",
            request_id="test-123",
            timestamp=time.time(),
            future=future,
            priority=RequestPriority.HIGH,
            user_id="user-123"
        )
        
        assert request.texts == ["hello", "world"]
        assert request.model == "test-model"
        assert request.task_type == "search_document"
        assert request.request_id == "test-123"
        assert request.priority == RequestPriority.HIGH
        assert request.user_id == "user-123"
        assert request.future == future
    
    def test_embedding_request_defaults(self):
        """Test EmbeddingRequest with default values"""
        future = asyncio.Future()
        request = EmbeddingRequest(
            texts=["hello"],
            model="test-model",
            task_type="search_document",
            request_id="test-123",
            timestamp=time.time(),
            future=future
        )
        
        assert request.priority == RequestPriority.NORMAL
        assert request.user_id is None
        assert request.estimated_processing_time == 0.0


class TestPriorityRequest:
    """Test PriorityRequest wrapper"""
    
    def test_priority_request_creation(self):
        """Test PriorityRequest creation"""
        future = asyncio.Future()
        embedding_req = EmbeddingRequest(
            texts=["hello"],
            model="test-model",
            task_type="search_document",
            request_id="test-123",
            timestamp=time.time(),
            future=future,
            priority=RequestPriority.HIGH
        )
        
        priority_req = PriorityRequest(
            priority=RequestPriority.HIGH.value,
            timestamp=time.time(),
            request=embedding_req
        )
        
        assert priority_req.priority == RequestPriority.HIGH.value
        assert priority_req.request == embedding_req
    
    def test_priority_request_ordering(self):
        """Test PriorityRequest ordering logic"""
        future1 = asyncio.Future()
        future2 = asyncio.Future()
        
        early_time = 1000.0
        late_time = 2000.0
        
        # Higher priority (lower number) should come first
        high_priority_req = PriorityRequest(
            priority=RequestPriority.HIGH.value,
            timestamp=late_time,
            request=EmbeddingRequest(
                texts=["high"], model="model", task_type="task",
                request_id="high", timestamp=late_time, future=future1
            )
        )
        
        low_priority_req = PriorityRequest(
            priority=RequestPriority.LOW.value,
            timestamp=early_time,
            request=EmbeddingRequest(
                texts=["low"], model="model", task_type="task",
                request_id="low", timestamp=early_time, future=future2
            )
        )
        
        # High priority should be "less than" low priority
        assert high_priority_req < low_priority_req
        
        # Same priority: earlier timestamp should come first
        same_priority_early = PriorityRequest(
            priority=RequestPriority.NORMAL.value,
            timestamp=early_time,
            request=EmbeddingRequest(
                texts=["early"], model="model", task_type="task",
                request_id="early", timestamp=early_time, future=asyncio.Future()
            )
        )
        
        same_priority_late = PriorityRequest(
            priority=RequestPriority.NORMAL.value,
            timestamp=late_time,
            request=EmbeddingRequest(
                texts=["late"], model="model", task_type="task",
                request_id="late", timestamp=late_time, future=asyncio.Future()
            )
        )
        
        assert same_priority_early < same_priority_late


class TestBatchProcessor:
    """Test BatchProcessor class"""
    
    @pytest.fixture
    def mock_model(self):
        """Mock model fixture"""
        model = MagicMock()
        model.encode.return_value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return model
    
    @pytest.fixture
    def batch_processor(self):
        """Batch processor fixture"""
        return BatchProcessor()
    
    def test_batch_processor_init(self, batch_processor):
        """Test BatchProcessor initialization"""
        assert batch_processor.request_queue is not None
        assert batch_processor.stats["total_requests"] == 0
        assert batch_processor.stats["total_batches"] == 0
        assert batch_processor.stats["total_processing_time"] == 0.0
        assert batch_processor.is_processing is False
    
    @pytest.mark.asyncio
    async def test_add_request_single(self, batch_processor):
        """Test adding a single request"""
        with patch.object(batch_processor, '_ensure_processing'):
            result_future = await batch_processor.add_request(
                texts=["hello world"],
                model="test-model",
                task_type="search_document",
                request_id="test-123",
                priority=RequestPriority.NORMAL,
                user_id="user-123"
            )
            
            assert isinstance(result_future, asyncio.Future)
            assert batch_processor.stats["total_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_add_request_multiple_texts(self, batch_processor):
        """Test adding request with multiple texts"""
        with patch.object(batch_processor, '_ensure_processing'):
            result_future = await batch_processor.add_request(
                texts=["hello", "world", "test"],
                model="test-model",
                task_type="search_document",
                request_id="test-123"
            )
            
            assert isinstance(result_future, asyncio.Future)
            assert batch_processor.stats["total_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_add_request_priority_ordering(self, batch_processor):
        """Test that requests are ordered by priority"""
        with patch.object(batch_processor, '_ensure_processing'):
            # Add requests with different priorities
            low_future = await batch_processor.add_request(
                texts=["low"], model="model", task_type="task",
                request_id="low", priority=RequestPriority.LOW
            )
            
            urgent_future = await batch_processor.add_request(
                texts=["urgent"], model="model", task_type="task",
                request_id="urgent", priority=RequestPriority.URGENT
            )
            
            normal_future = await batch_processor.add_request(
                texts=["normal"], model="model", task_type="task",
                request_id="normal", priority=RequestPriority.NORMAL
            )
            
            # Extract requests from queue (without removing)
            queue_items = []
            temp_queue = []
            
            # Drain queue to check ordering
            while not batch_processor.request_queue.empty():
                item = await batch_processor.request_queue.get()
                queue_items.append(item)
                temp_queue.append(item)
            
            # Restore queue
            for item in temp_queue:
                await batch_processor.request_queue.put(item)
            
            # Check ordering: urgent -> normal -> low
            assert queue_items[0].request.request_id == "urgent"
            assert queue_items[1].request.request_id == "normal"
            assert queue_items[2].request.request_id == "low"
    
    @pytest.mark.asyncio
    async def test_collect_batch_single_request(self, batch_processor):
        """Test collecting a batch with single request"""
        # Add a request
        future = asyncio.Future()
        request = EmbeddingRequest(
            texts=["hello"],
            model="test-model",
            task_type="search_document",
            request_id="test-123",
            timestamp=time.time(),
            future=future
        )
        priority_request = PriorityRequest(
            priority=RequestPriority.NORMAL.value,
            timestamp=time.time(),
            request=request
        )
        
        await batch_processor.request_queue.put(priority_request)
        
        # Collect batch
        batch = await batch_processor._collect_batch()
        
        assert len(batch) == 1
        assert batch[0] == request
    
    @pytest.mark.asyncio
    async def test_collect_batch_multiple_requests(self, batch_processor):
        """Test collecting a batch with multiple requests"""
        # Add multiple requests
        requests = []
        for i in range(5):
            future = asyncio.Future()
            request = EmbeddingRequest(
                texts=[f"text {i}"],
                model="test-model",
                task_type="search_document",
                request_id=f"test-{i}",
                timestamp=time.time(),
                future=future
            )
            requests.append(request)
            
            priority_request = PriorityRequest(
                priority=RequestPriority.NORMAL.value,
                timestamp=time.time(),
                request=request
            )
            await batch_processor.request_queue.put(priority_request)
        
        # Collect batch
        batch = await batch_processor._collect_batch()
        
        assert len(batch) <= 5  # Should collect all or up to max batch size
        assert all(req in requests for req in batch)
    
    @pytest.mark.asyncio
    async def test_collect_batch_timeout(self, batch_processor):
        """Test batch collection timeout"""
        # Add one request
        future = asyncio.Future()
        request = EmbeddingRequest(
            texts=["hello"],
            model="test-model",
            task_type="search_document",
            request_id="test-123",
            timestamp=time.time(),
            future=future
        )
        priority_request = PriorityRequest(
            priority=RequestPriority.NORMAL.value,
            timestamp=time.time(),
            request=request
        )
        
        await batch_processor.request_queue.put(priority_request)
        
        # Mock short timeout
        with patch('src.batch_processor.BATCH_TIMEOUT_MS', 1):  # 1ms timeout
            start_time = time.time()
            batch = await batch_processor._collect_batch()
            end_time = time.time()
            
            assert len(batch) == 1
            # Should return quickly due to timeout
            assert (end_time - start_time) < 0.1  # Less than 100ms
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, batch_processor, mock_model):
        """Test successful batch processing"""
        # Create requests
        requests = []
        for i in range(2):
            future = asyncio.Future()
            request = EmbeddingRequest(
                texts=[f"text {i}"],
                model="test-model",
                task_type="search_document",
                request_id=f"test-{i}",
                timestamp=time.time(),
                future=future
            )
            requests.append(request)
        
        # Process batch
        await batch_processor._process_batch(requests, mock_model)
        
        # Check that futures are completed
        for request in requests:
            assert request.future.done()
            assert not request.future.exception()
            result = request.future.result()
            assert isinstance(result, np.ndarray)
    
    @pytest.mark.asyncio
    async def test_process_batch_model_error(self, batch_processor, mock_model):
        """Test batch processing with model error"""
        mock_model.encode.side_effect = Exception("Model failed")
        
        # Create request
        future = asyncio.Future()
        request = EmbeddingRequest(
            texts=["text"],
            model="test-model",
            task_type="search_document",
            request_id="test-123",
            timestamp=time.time(),
            future=future
        )
        
        # Process batch
        await batch_processor._process_batch([request], mock_model)
        
        # Check that future has exception
        assert request.future.done()
        assert request.future.exception() is not None
        assert "Model failed" in str(request.future.exception())
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, batch_processor, mock_model):
        """Test batch processor cache integration"""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache.put = AsyncMock()
        
        with patch('src.batch_processor.embedding_cache', mock_cache):
            # Create request
            future = asyncio.Future()
            request = EmbeddingRequest(
                texts=["cached text"],
                model="test-model",
                task_type="search_document",
                request_id="test-123",
                timestamp=time.time(),
                future=future
            )
            
            # Process batch
            await batch_processor._process_batch([request], mock_model)
            
            # Check cache interactions
            mock_cache.get.assert_called()
            mock_cache.put.assert_called()
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, batch_processor):
        """Test batch processing with cache hit"""
        cached_embedding = np.array([1.0, 2.0, 3.0])
        mock_cache = AsyncMock()
        mock_cache.get.return_value = cached_embedding
        
        with patch('src.batch_processor.embedding_cache', mock_cache):
            # Create request
            future = asyncio.Future()
            request = EmbeddingRequest(
                texts=["cached text"],
                model="test-model",
                task_type="search_document",
                request_id="test-123",
                timestamp=time.time(),
                future=future
            )
            
            # Process batch (model should not be called)
            mock_model = MagicMock()
            await batch_processor._process_batch([request], mock_model)
            
            # Check result
            assert request.future.done()
            result = request.future.result()
            assert np.array_equal(result[0], cached_embedding)
            
            # Model should not be called for cached items
            mock_model.encode.assert_not_called()
    
    def test_get_stats(self, batch_processor):
        """Test getting batch processor statistics"""
        # Simulate some activity
        batch_processor.stats["total_requests"] = 100
        batch_processor.stats["total_batches"] = 10
        batch_processor.stats["total_processing_time"] = 5.0
        
        stats = batch_processor.get_stats()
        
        assert stats["total_requests"] == 100
        assert stats["total_batches"] == 10
        assert stats["total_processing_time"] == 5.0
        assert stats["average_batch_size"] == 10.0
        assert stats["average_processing_time"] == 0.5
        assert "queue_size" in stats
    
    def test_get_stats_no_activity(self, batch_processor):
        """Test getting stats with no activity"""
        stats = batch_processor.get_stats()
        
        assert stats["total_requests"] == 0
        assert stats["total_batches"] == 0
        assert stats["total_processing_time"] == 0.0
        assert stats["average_batch_size"] == 0.0
        assert stats["average_processing_time"] == 0.0
    
    @pytest.mark.asyncio
    async def test_process_batches_lifecycle(self, batch_processor, mock_model):
        """Test the main process_batches lifecycle"""
        # Start processing
        process_task = asyncio.create_task(
            batch_processor.process_batches(mock_model)
        )
        
        # Give it time to start
        await asyncio.sleep(0.01)
        assert batch_processor.is_processing is True
        
        # Add a request
        result_future = await batch_processor.add_request(
            texts=["test"],
            model="test-model",
            task_type="search_document",
            request_id="test-123"
        )
        
        # Wait for processing
        try:
            result = await asyncio.wait_for(result_future, timeout=1.0)
            assert isinstance(result, list)
        except asyncio.TimeoutError:
            pytest.fail("Request processing timed out")
        
        # Cancel processing
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass
        
        assert batch_processor.is_processing is False
    
    @pytest.mark.asyncio
    async def test_memory_management_integration(self, batch_processor, mock_model):
        """Test integration with memory management"""
        with patch('src.batch_processor.ENABLE_DYNAMIC_MEMORY', True):
            with patch('src.batch_processor.get_memory_stats') as mock_memory_stats:
                mock_memory_stats.return_value = {
                    'memory_pressure': 'low',
                    'available_memory_mb': 1000
                }
                
                # Create request
                future = asyncio.Future()
                request = EmbeddingRequest(
                    texts=["test"],
                    model="test-model",
                    task_type="search_document",
                    request_id="test-123",
                    timestamp=time.time(),
                    future=future
                )
                
                # Process should work normally with low memory pressure
                await batch_processor._process_batch([request], mock_model)
                
                assert request.future.done()


class TestBatchProcessorConcurrency:
    """Test batch processor concurrency and thread safety"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        batch_processor = BatchProcessor()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0]] * 10)
        
        # Start processing
        process_task = asyncio.create_task(
            batch_processor.process_batches(mock_model)
        )
        
        await asyncio.sleep(0.01)  # Let processing start
        
        # Submit multiple concurrent requests
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                batch_processor.add_request(
                    texts=[f"text {i}"],
                    model="test-model",
                    task_type="search_document",
                    request_id=f"test-{i}"
                )
            )
            tasks.append(task)
        
        # Wait for all requests to be queued
        futures = await asyncio.gather(*tasks)
        
        # Wait for all processing to complete
        try:
            results = await asyncio.gather(*futures, timeout=2.0)
            assert len(results) == 10
            for result in results:
                assert isinstance(result, list)
        except asyncio.TimeoutError:
            pytest.fail("Concurrent request processing timed out")
        finally:
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_priority_under_load(self):
        """Test priority handling under high load"""
        batch_processor = BatchProcessor()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0, 3.0]] * 20)
        
        # Add many low priority requests first
        low_priority_tasks = []
        for i in range(15):
            task = asyncio.create_task(
                batch_processor.add_request(
                    texts=[f"low {i}"],
                    model="test-model",
                    task_type="search_document",
                    request_id=f"low-{i}",
                    priority=RequestPriority.LOW
                )
            )
            low_priority_tasks.append(task)
        
        # Add urgent request
        urgent_task = asyncio.create_task(
            batch_processor.add_request(
                texts=["urgent"],
                model="test-model",
                task_type="search_document",
                request_id="urgent",
                priority=RequestPriority.URGENT
            )
        )
        
        # Start processing
        process_task = asyncio.create_task(
            batch_processor.process_batches(mock_model)
        )
        
        await asyncio.sleep(0.01)  # Let processing start
        
        # The urgent request should complete faster than low priority ones
        try:
            urgent_result = await asyncio.wait_for(urgent_task, timeout=1.0)
            assert isinstance(urgent_result, list)
        except asyncio.TimeoutError:
            pytest.fail("Urgent request did not complete in time")
        finally:
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])