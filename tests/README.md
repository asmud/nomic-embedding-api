# Test Suite Documentation

This directory contains a comprehensive test suite for the Nomic Embedding v2 API with >80% code coverage.

## Test Structure

### Core Test Files

- **`test_api.py`** - Enhanced API endpoint tests covering all REST endpoints, error conditions, streaming, priorities, and edge cases
- **`test_models.py`** - Model initialization, encoding, optimization, and error handling tests for both EmbeddingModel and NoMiCMoEModel
- **`test_config.py`** - Configuration validation, environment variable parsing, model presets, and settings tests
- **`test_cache.py`** - Comprehensive caching tests including LRU cache, Redis integration, thread safety, and performance
- **`test_batch_processor.py`** - Batch processing, request prioritization, queue management, and concurrent processing tests
- **`test_model_pool.py`** - Model pool management, load balancing, health monitoring, and multi-GPU support tests
- **`test_memory_manager.py`** - Memory monitoring, pressure detection, cache resizing, and garbage collection tests
- **`test_hardware_optimizer.py`** - Hardware detection, optimization strategies, workload profiling, and device selection tests
- **`test_integration.py`** - End-to-end integration tests covering complete workflows, error recovery, and system limits
- **`test_performance.py`** - Performance benchmarks, load testing, scalability limits, and resource utilization tests

### Configuration Files

- **`pytest.ini`** - Pytest configuration with coverage settings, markers, and test execution options
- **`conftest.py`** - Shared fixtures, test environment setup, and pytest configuration

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-asyncio pytest-timeout pytest-xdist
```

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only
pytest -m "not slow"    # Skip slow tests

# Run specific test files
pytest tests/test_api.py
pytest tests/test_models.py
```

### Parallel Execution

```bash
# Run tests in parallel (faster execution)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Performance Testing

```bash
# Run performance tests with output
pytest tests/test_performance.py -v -s

# Run integration tests with timing
pytest tests/test_integration.py -v --durations=10
```

## Test Categories

### Unit Tests (Primary Focus)
- **Models**: Model loading, encoding, optimization
- **Cache**: LRU cache operations, Redis integration
- **Config**: Environment variable parsing, validation
- **Batch Processing**: Request queuing, prioritization
- **Memory Management**: Monitoring, pressure detection
- **Hardware Optimization**: Device selection, workload profiling

### Integration Tests
- **Full Workflows**: Complete request processing
- **Error Recovery**: System resilience testing
- **Component Integration**: Cross-module functionality
- **Configuration Impact**: Environment-based testing

### Performance Tests
- **Latency Benchmarks**: Single request timing
- **Throughput Testing**: Concurrent request handling
- **Resource Utilization**: Memory and CPU monitoring
- **Scalability Limits**: Large batch and load testing

## Test Environment

### Environment Variables
Tests automatically configure a test environment with:
- `LOG_LEVEL=ERROR` (reduced log noise)
- `MODEL_POOL_SIZE=0` (disabled for most tests)
- `REDIS_ENABLED=false` (uses local cache)
- `ENABLE_HARDWARE_OPTIMIZATION=false` (simplified testing)

### Mocking Strategy
- **Model Loading**: Mocked to avoid downloading actual models
- **Hardware Detection**: Mocked for consistent test environments
- **External Services**: Redis, network calls mocked
- **Time-sensitive Operations**: Controllable timing

## Coverage Goals

The test suite aims for >80% code coverage across:

### High Priority Coverage (>90%)
- Core API endpoints (`src/api.py`)
- Model operations (`src/models.py`)
- Caching logic (`src/cache.py`)
- Batch processing (`src/batch_processor.py`)

### Medium Priority Coverage (>80%)
- Configuration management (`src/config.py`)
- Model pool operations (`src/model_pool.py`)
- Memory management (`src/memory_manager.py`)
- Hardware optimization (`src/hardware_optimizer.py`)

### Integration Coverage (>70%)
- End-to-end workflows
- Error handling paths
- Resource cleanup
- System limits

## Test Data

### Sample Texts
Tests use varied text samples including:
- Short and long texts
- Unicode and special characters
- Edge cases (empty, very long)
- Batch scenarios

### Mock Responses
Consistent mock embeddings with:
- Configurable dimensions (768, 256)
- Realistic value ranges
- Error injection capabilities

## Performance Expectations

### Latency Targets
- Single request: <5 seconds average
- Batch processing: <10 seconds for 20 texts
- Cache hits: <100ms additional overhead

### Throughput Targets
- Concurrent requests: >5 req/sec under load
- Success rate: >80% under normal load, >70% under stress
- Memory growth: <500MB during extended testing

### Resource Limits
- CPU utilization: <95% sustained
- Memory usage: <1GB growth during testing
- Response time degradation: <2x over time

## Debugging Tests

### Verbose Output
```bash
# See detailed test output
pytest -v -s

# Show test durations
pytest --durations=10

# Debug specific failing tests
pytest tests/test_api.py::test_embeddings_endpoint -v -s
```

### Coverage Analysis
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Show missing lines
pytest --cov=src --cov-report=term-missing
```

### Memory Debugging
```bash
# Monitor memory usage
pytest tests/test_performance.py::test_memory_usage_under_load -v -s

# Profile memory leaks
pytest tests/test_performance.py::test_memory_leak_detection -v -s
```

## Continuous Integration

### GitHub Actions
Example CI configuration:
```yaml
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml --cov-fail-under=80
    
- name: Upload coverage
  uses: codecov/codecov-action@v1
  with:
    file: ./coverage.xml
```

### Test Parallelization
For CI environments:
```bash
# Fast test execution
pytest -n auto -m "not slow"

# Full test suite with performance tests
pytest -n 4 --cov=src --cov-report=xml
```

## Test Development Guidelines

### Writing New Tests
1. Use appropriate test markers (`@pytest.mark.unit`, etc.)
2. Follow naming convention: `test_<functionality>`
3. Include docstrings describing test purpose
4. Use fixtures for setup/teardown
5. Mock external dependencies

### Performance Tests
1. Use `benchmark_timer` fixture for timing
2. Include performance assertions with reasonable tolerances
3. Test both success and failure scenarios
4. Monitor resource usage

### Integration Tests
1. Test complete workflows end-to-end
2. Include error recovery scenarios
3. Validate system behavior under stress
4. Test configuration variations

## Maintenance

### Regular Tasks
- Update performance baselines as system improves
- Add tests for new features
- Review and update mocks for external dependencies
- Monitor test execution time and optimize slow tests

### Coverage Monitoring
- Review coverage reports weekly
- Identify and test uncovered code paths
- Add tests for edge cases and error conditions
- Maintain >80% overall coverage

This comprehensive test suite ensures the reliability, performance, and maintainability of the Nomic Embedding API.