import pytest
import asyncio
import time
import threading
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_fast_model(self):
        """Mock model with fast, consistent response times"""
        model = MagicMock()
        model.encode.return_value = np.random.random((1, 768))
        model.get_embedding_dimension.return_value = 768
        model.model_name = "fast-test-model"
        return model
    
    def test_single_request_latency(self, client):
        """Test single request latency"""
        payload = {
            "input": "Performance test sentence",
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        latencies = []
        
        # Warm up
        for _ in range(3):
            client.post("/v1/embeddings", json=payload)
        
        # Measure latency
        for _ in range(10):
            start_time = time.time()
            response = client.post("/v1/embeddings", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            
            logger.info(f"Average latency: {avg_latency:.2f}ms")
            logger.info(f"95th percentile latency: {p95_latency:.2f}ms")
            
            # Reasonable latency expectations (adjust based on system)
            assert avg_latency < 5000  # Less than 5 seconds average
            assert p95_latency < 10000  # Less than 10 seconds P95
    
    def test_batch_processing_performance(self, client):
        """Test batch processing performance"""
        batch_sizes = [1, 5, 10, 20]
        results = {}
        
        for batch_size in batch_sizes:
            texts = [f"Batch test sentence {i}" for i in range(batch_size)]
            payload = {
                "input": texts,
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            start_time = time.time()
            response = client.post("/v1/embeddings", json=payload)
            end_time = time.time()
            
            if response.status_code == 200:
                total_time = end_time - start_time
                time_per_text = total_time / batch_size
                results[batch_size] = {
                    'total_time': total_time,
                    'time_per_text': time_per_text,
                    'throughput': batch_size / total_time
                }
        
        if results:
            logger.info("Batch Performance Results:")
            for batch_size, metrics in results.items():
                logger.info(f"Batch size {batch_size}: {metrics['throughput']:.2f} texts/sec")
            
            # Larger batches should generally be more efficient
            if len(results) >= 2:
                batch_sizes_tested = sorted(results.keys())
                small_batch_efficiency = results[batch_sizes_tested[0]]['time_per_text']
                large_batch_efficiency = results[batch_sizes_tested[-1]]['time_per_text']
                
                # Larger batches should be at least as efficient (or system handles well)
                efficiency_ratio = large_batch_efficiency / small_batch_efficiency
                assert efficiency_ratio <= 2.0  # Not more than 2x slower per item
    
    def test_concurrent_request_performance(self, client):
        """Test performance under concurrent load"""
        concurrent_users = [1, 5, 10]
        requests_per_user = 3
        results = {}
        
        def make_requests(user_id, num_requests):
            times = []
            successes = 0
            
            for i in range(num_requests):
                payload = {
                    "input": f"Concurrent test from user {user_id}, request {i}",
                    "model": "nomic-embed-text-v2-moe-distilled"
                }
                
                start_time = time.time()
                try:
                    response = client.post("/v1/embeddings", json=payload)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        times.append(end_time - start_time)
                        successes += 1
                except Exception:
                    pass
            
            return times, successes
        
        for num_users in concurrent_users:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [
                    executor.submit(make_requests, user_id, requests_per_user)
                    for user_id in range(num_users)
                ]
                
                all_times = []
                total_successes = 0
                
                for future in as_completed(futures):
                    times, successes = future.result()
                    all_times.extend(times)
                    total_successes += successes
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if all_times:
                avg_response_time = statistics.mean(all_times)
                total_requests = num_users * requests_per_user
                overall_throughput = total_successes / total_time
                
                results[num_users] = {
                    'avg_response_time': avg_response_time,
                    'success_rate': total_successes / total_requests,
                    'throughput': overall_throughput
                }
        
        if results:
            logger.info("Concurrent Performance Results:")
            for users, metrics in results.items():
                logger.info(f"{users} users: {metrics['throughput']:.2f} req/sec, "
                      f"{metrics['success_rate']:.2%} success rate")
            
            # System should maintain reasonable performance under load
            for metrics in results.values():
                assert metrics['success_rate'] >= 0.8  # At least 80% success rate
    
    def test_memory_usage_under_load(self, client):
        """Test memory usage patterns under load"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate load
        for batch in range(5):
            texts = [f"Memory test batch {batch} text {i}" for i in range(10)]
            payload = {
                "input": texts,
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            response = client.post("/v1/embeddings", json=payload)
            
            # Check memory after each batch
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory shouldn't grow excessively
            assert memory_increase < 1000  # Less than 1GB increase
        
        # Force garbage collection and check final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
              f"(+{total_increase:.1f}MB)")
        
        # Should not have excessive memory growth
        assert total_increase < 500  # Less than 500MB growth
    
    def test_cache_performance_impact(self, client):
        """Test performance impact of caching"""
        # Test same request multiple times
        payload = {
            "input": "Cache performance test - this exact text",
            "model": "nomic-embed-text-v2-moe-distilled"
        }
        
        # Clear cache first
        client.post("/clear-caches")
        
        # First request (cache miss)
        start_time = time.time()
        response1 = client.post("/v1/embeddings", json=payload)
        first_request_time = time.time() - start_time
        
        if response1.status_code == 200:
            # Second request (potential cache hit)
            start_time = time.time()
            response2 = client.post("/v1/embeddings", json=payload)
            second_request_time = time.time() - start_time
            
            if response2.status_code == 200:
                # Cache should improve performance or at least not hurt it significantly
                performance_ratio = second_request_time / first_request_time
                
                logger.info(f"First request: {first_request_time*1000:.2f}ms")
                logger.info(f"Second request: {second_request_time*1000:.2f}ms")
                logger.info(f"Performance ratio: {performance_ratio:.2f}")
                
                # Second request should not be significantly slower
                assert performance_ratio <= 1.5  # At most 50% slower


class TestScalabilityLimits:
    """Test system scalability and limits"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_large_text_handling(self, client):
        """Test handling of increasingly large texts"""
        text_sizes = [100, 1000, 5000]  # Number of words
        
        for size in text_sizes:
            large_text = " ".join([f"word{i}" for i in range(size)])
            payload = {
                "input": large_text,
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            start_time = time.time()
            response = client.post("/v1/embeddings", json=payload)
            processing_time = time.time() - start_time
            
            logger.info(f"Text size {size} words: {processing_time:.2f}s, "
                  f"status: {response.status_code}")
            
            # Should handle gracefully (either succeed or fail with appropriate error)
            assert response.status_code in [200, 413, 422, 503, 500]
            
            # Processing time should scale reasonably
            if response.status_code == 200:
                assert processing_time < 30  # Max 30 seconds
    
    def test_batch_size_scaling(self, client):
        """Test scaling with different batch sizes"""
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            texts = [f"Scaling test text {i}" for i in range(batch_size)]
            payload = {
                "input": texts,
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            start_time = time.time()
            response = client.post("/v1/embeddings", json=payload)
            processing_time = time.time() - start_time
            
            logger.info(f"Batch size {batch_size}: {processing_time:.2f}s, "
                  f"status: {response.status_code}")
            
            # Should handle or reject gracefully
            assert response.status_code in [200, 413, 422, 503, 500]
            
            if response.status_code == 200:
                data = response.json()
                assert len(data["data"]) == batch_size
    
    def test_sustained_load_performance(self, client):
        """Test performance under sustained load"""
        duration_seconds = 10
        requests_made = 0
        successful_requests = 0
        response_times = []
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            payload = {
                "input": f"Sustained load test {requests_made}",
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            request_start = time.time()
            try:
                response = client.post("/v1/embeddings", json=payload)
                request_end = time.time()
                
                requests_made += 1
                if response.status_code == 200:
                    successful_requests += 1
                    response_times.append(request_end - request_start)
                
            except Exception:
                requests_made += 1
            
            # Small delay to avoid overwhelming
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        
        if requests_made > 0:
            success_rate = successful_requests / requests_made
            throughput = successful_requests / total_time
            
            logger.info(f"Sustained load results:")
            logger.info(f"  Requests made: {requests_made}")
            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(f"  Throughput: {throughput:.2f} req/sec")
            
            if response_times:
                avg_response_time = statistics.mean(response_times)
                logger.info(f"  Avg response time: {avg_response_time:.2f}s")
                
                # Performance should remain reasonable
                assert avg_response_time < 5.0  # Less than 5 seconds average
            
            # Should maintain decent success rate
            assert success_rate >= 0.7  # At least 70% success rate


class TestResourceUtilization:
    """Test resource utilization patterns"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_cpu_utilization_patterns(self, client):
        """Test CPU utilization during various workloads"""
        import psutil
        
        # Baseline CPU usage
        cpu_before = psutil.cpu_percent(interval=1)
        
        # Generate CPU-intensive workload
        for i in range(5):
            payload = {
                "input": f"CPU test {i} " * 100,  # Longer texts
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            client.post("/v1/embeddings", json=payload)
        
        # Check CPU usage after workload
        cpu_after = psutil.cpu_percent(interval=1)
        
        logger.info(f"CPU usage: {cpu_before}% -> {cpu_after}%")
        
        # CPU usage should be reasonable (not pegged at 100%)
        assert cpu_after <= 95  # Should not completely saturate CPU
    
    def test_memory_leak_detection(self, client):
        """Test for memory leaks over extended usage"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_readings = [initial_memory]
        
        # Generate repeated workload
        for cycle in range(10):
            for i in range(5):
                payload = {
                    "input": f"Memory leak test cycle {cycle} request {i}",
                    "model": "nomic-embed-text-v2-moe-distilled"
                }
                client.post("/v1/embeddings", json=payload)
            
            # Force garbage collection
            gc.collect()
            
            # Record memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_readings.append(current_memory)
        
        # Analyze memory trend
        memory_growth = memory_readings[-1] - memory_readings[0]
        max_memory = max(memory_readings)
        
        logger.info(f"Memory readings: {memory_readings[0]:.1f} -> {memory_readings[-1]:.1f}MB")
        logger.info(f"Peak memory: {max_memory:.1f}MB")
        logger.info(f"Total growth: {memory_growth:.1f}MB")
        
        # Should not have excessive memory growth
        assert memory_growth < 200  # Less than 200MB growth
        assert max_memory < initial_memory + 500  # Peak should be reasonable
    
    def test_response_time_degradation(self, client):
        """Test for response time degradation over time"""
        response_times = []
        batch_count = 10
        requests_per_batch = 3
        
        for batch in range(batch_count):
            batch_times = []
            
            for req in range(requests_per_batch):
                payload = {
                    "input": f"Degradation test batch {batch} req {req}",
                    "model": "nomic-embed-text-v2-moe-distilled"
                }
                
                start_time = time.time()
                response = client.post("/v1/embeddings", json=payload)
                end_time = time.time()
                
                if response.status_code == 200:
                    batch_times.append(end_time - start_time)
            
            if batch_times:
                avg_batch_time = statistics.mean(batch_times)
                response_times.append(avg_batch_time)
        
        if len(response_times) >= 3:
            # Check for significant degradation
            early_avg = statistics.mean(response_times[:3])
            late_avg = statistics.mean(response_times[-3:])
            degradation_ratio = late_avg / early_avg
            
            logger.info(f"Early avg: {early_avg:.2f}s, Late avg: {late_avg:.2f}s")
            logger.info(f"Degradation ratio: {degradation_ratio:.2f}")
            
            # Response times should not degrade significantly
            assert degradation_ratio <= 2.0  # Not more than 2x slower


class TestStressTests:
    """Stress tests to find system limits"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_rapid_fire_requests(self, client):
        """Test rapid-fire request handling"""
        num_requests = 50
        successful = 0
        errors = 0
        timeouts = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            payload = {
                "input": f"Rapid fire test {i}",
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            try:
                response = client.post("/v1/embeddings", json=payload, timeout=10)
                if response.status_code == 200:
                    successful += 1
                else:
                    errors += 1
            except Exception:
                timeouts += 1
        
        total_time = time.time() - start_time
        
        logger.info(f"Rapid fire results:")
        logger.info(f"  Successful: {successful}/{num_requests}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Timeouts: {timeouts}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Rate: {num_requests/total_time:.2f} req/sec")
        
        # Should handle most requests successfully
        success_rate = successful / num_requests
        assert success_rate >= 0.6  # At least 60% success under stress
    
    def test_extreme_concurrency(self, client):
        """Test extreme concurrency levels"""
        max_workers = 20
        requests_per_worker = 2
        results = []
        
        def worker_function(worker_id):
            successes = 0
            for i in range(requests_per_worker):
                payload = {
                    "input": f"Extreme concurrency worker {worker_id} req {i}",
                    "model": "nomic-embed-text-v2-moe-distilled"
                }
                
                try:
                    response = client.post("/v1/embeddings", json=payload, timeout=15)
                    if response.status_code == 200:
                        successes += 1
                except Exception:
                    pass
            
            return successes
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(worker_function, worker_id)
                for worker_id in range(max_workers)
            ]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=20)
                    results.append(result)
                except Exception:
                    results.append(0)
        
        total_time = time.time() - start_time
        total_successful = sum(results)
        total_attempted = max_workers * requests_per_worker
        
        logger.info(f"Extreme concurrency results:")
        logger.info(f"  Workers: {max_workers}")
        logger.info(f"  Total attempted: {total_attempted}")
        logger.info(f"  Total successful: {total_successful}")
        logger.info(f"  Success rate: {total_successful/total_attempted:.2%}")
        logger.info(f"  Time: {total_time:.2f}s")
        
        # Should maintain some level of service under extreme load
        success_rate = total_successful / total_attempted
        assert success_rate >= 0.3  # At least 30% success under extreme stress


class TestPerformanceRegression:
    """Test for performance regressions"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    def test_baseline_performance_metrics(self, client):
        """Establish baseline performance metrics"""
        # This test establishes baseline metrics that can be compared
        # in future test runs to detect performance regressions
        
        test_cases = [
            {"name": "single_short", "input": "Short text", "expected_max_time": 2.0},
            {"name": "single_long", "input": "Long text " * 100, "expected_max_time": 5.0},
            {"name": "batch_small", "input": ["Text " + str(i) for i in range(5)], "expected_max_time": 3.0},
            {"name": "batch_large", "input": ["Text " + str(i) for i in range(20)], "expected_max_time": 10.0},
        ]
        
        results = {}
        
        for test_case in test_cases:
            payload = {
                "input": test_case["input"],
                "model": "nomic-embed-text-v2-moe-distilled"
            }
            
            # Warm up
            client.post("/v1/embeddings", json=payload)
            
            # Measure performance
            times = []
            for _ in range(3):
                start_time = time.time()
                response = client.post("/v1/embeddings", json=payload)
                end_time = time.time()
                
                if response.status_code == 200:
                    times.append(end_time - start_time)
            
            if times:
                avg_time = statistics.mean(times)
                results[test_case["name"]] = avg_time
                
                logger.info(f"{test_case['name']}: {avg_time:.3f}s")
                
                # Check against expected performance
                assert avg_time <= test_case["expected_max_time"], \
                    f"{test_case['name']} took {avg_time:.3f}s, expected <= {test_case['expected_max_time']}s"
        
        # Store results for comparison (in real scenarios, you'd persist these)
        logger.info("Baseline metrics established:", results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output