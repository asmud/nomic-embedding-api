import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config


class TestModelPresets:
    """Test model preset configurations"""
    
    def test_model_presets_structure(self):
        """Test that model presets have correct structure"""
        assert "nomic-moe-768" in config.MODEL_PRESETS
        assert "nomic-moe-256" in config.MODEL_PRESETS
        
        for preset_name, preset_config in config.MODEL_PRESETS.items():
            assert "model_name" in preset_config
            assert "use_model2vec" in preset_config
            assert "dimensions" in preset_config
            assert "description" in preset_config
            
            assert isinstance(preset_config["model_name"], str)
            assert isinstance(preset_config["use_model2vec"], bool)
            assert isinstance(preset_config["dimensions"], int)
            assert isinstance(preset_config["description"], str)
            assert preset_config["dimensions"] > 0
    
    def test_nomic_768_preset(self):
        """Test nomic-moe-768 preset configuration"""
        preset = config.MODEL_PRESETS["nomic-moe-768"]
        assert preset["model_name"] == "nomic-ai/nomic-embed-text-v2-moe"
        assert preset["use_model2vec"] is False
        assert preset["dimensions"] == 768
        assert "Full Nomic MoE model" in preset["description"]
    
    def test_nomic_256_preset(self):
        """Test nomic-moe-256 preset configuration"""
        preset = config.MODEL_PRESETS["nomic-moe-256"]
        assert preset["model_name"] == "Abdelkareem/nomic-embed-text-v2-moe_distilled"
        assert preset["use_model2vec"] is True
        assert preset["dimensions"] == 256
        assert "Distilled Nomic model" in preset["description"]


class TestEnvironmentVariables:
    """Test environment variable parsing and defaults"""
    
    def test_default_embedding_model(self):
        """Test default embedding model selection"""
        with patch.dict(os.environ, {}, clear=True):
            # Reload config to test defaults
            import importlib
            importlib.reload(config)
            assert config.EMBEDDING_MODEL in ["nomic-moe-768", "nomic-moe-256"]
    
    def test_custom_embedding_model(self):
        """Test custom embedding model from environment"""
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "nomic-moe-256"}):
            import importlib
            importlib.reload(config)
            assert config.EMBEDDING_MODEL == "nomic-moe-256"
    
    def test_invalid_embedding_model_fallback(self):
        """Test fallback for invalid embedding model"""
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "invalid-model"}):
            with patch('builtins.print') as mock_print:
                import importlib
                importlib.reload(config)
                mock_print.assert_called()
                assert config.EMBEDDING_MODEL == "nomic-moe-768"
    
    def test_host_and_port_defaults(self):
        """Test HOST and PORT defaults"""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(config)
            assert config.HOST == "0.0.0.0"
            assert config.PORT == 8000
    
    def test_host_and_port_custom(self):
        """Test custom HOST and PORT from environment"""
        with patch.dict(os.environ, {"HOST": "127.0.0.1", "PORT": "9000"}):
            import importlib
            importlib.reload(config)
            assert config.HOST == "127.0.0.1"
            assert config.PORT == 9000
    
    def test_log_level_parsing(self):
        """Test LOG_LEVEL parsing"""
        test_cases = [
            ("debug", "DEBUG"),
            ("info", "INFO"),
            ("warning", "WARNING"),
            ("error", "ERROR"),
            ("DEBUG", "DEBUG"),
            ("", "INFO")  # Default
        ]
        
        for env_value, expected in test_cases:
            env_dict = {"LOG_LEVEL": env_value} if env_value else {}
            with patch.dict(os.environ, env_dict, clear=True):
                import importlib
                importlib.reload(config)
                assert config.LOG_LEVEL == expected
    
    def test_boolean_parsing(self):
        """Test boolean environment variable parsing"""
        boolean_vars = [
            ("TRUST_REMOTE_CODE", "trust_remote_code"),
            ("ENABLE_QUANTIZATION", "enable_quantization"),
            ("TORCH_COMPILE", "torch_compile"),
            ("ENABLE_CACHING", "enable_caching"),
            ("ENABLE_MULTI_GPU", "enable_multi_gpu"),
            ("REDIS_ENABLED", "redis_enabled"),
            ("ENABLE_DYNAMIC_MEMORY", "enable_dynamic_memory"),
            ("ENABLE_HARDWARE_OPTIMIZATION", "enable_hardware_optimization"),
        ]
        
        for env_var, config_attr in boolean_vars:
            # Test True values
            for true_value in ["true", "True", "TRUE", "1", "yes"]:
                with patch.dict(os.environ, {env_var: true_value}):
                    import importlib
                    importlib.reload(config)
                    assert getattr(config, env_var) is True
            
            # Test False values
            for false_value in ["false", "False", "FALSE", "0", "no", ""]:
                with patch.dict(os.environ, {env_var: false_value}):
                    import importlib
                    importlib.reload(config)
                    assert getattr(config, env_var) is False
    
    def test_integer_parsing(self):
        """Test integer environment variable parsing"""
        integer_vars = [
            ("MAX_BATCH_SIZE", 32),
            ("BATCH_TIMEOUT_MS", 50),
            ("CACHE_SIZE", 1000),
            ("MAX_CONCURRENT_REQUESTS", 100),
            ("MODEL_POOL_SIZE", 0),
            ("HEALTH_CHECK_INTERVAL", 30),
            ("MAX_MODEL_ERROR_COUNT", 5),
            ("REDIS_DB", 0),
            ("REDIS_MAX_CONNECTIONS", 20),
            ("REDIS_CACHE_TTL", 3600),
            ("REDIS_SESSION_TTL", 300),
        ]
        
        for env_var, default_value in integer_vars:
            # Test default value
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                importlib.reload(config)
                assert getattr(config, env_var) == default_value
            
            # Test custom value
            custom_value = default_value + 10
            with patch.dict(os.environ, {env_var: str(custom_value)}):
                import importlib
                importlib.reload(config)
                assert getattr(config, env_var) == custom_value
    
    def test_float_parsing(self):
        """Test float environment variable parsing"""
        float_vars = [
            ("MEMORY_MONITORING_INTERVAL", 30.0),
            ("MEMORY_PRESSURE_HIGH_PERCENT", 85.0),
            ("MEMORY_PRESSURE_CRITICAL_PERCENT", 95.0),
            ("MIN_AVAILABLE_MEMORY_MB", 512.0),
        ]
        
        for env_var, default_value in float_vars:
            # Test default value
            with patch.dict(os.environ, {}, clear=True):
                import importlib
                importlib.reload(config)
                assert getattr(config, env_var) == default_value
            
            # Test custom value
            custom_value = default_value + 5.5
            with patch.dict(os.environ, {env_var: str(custom_value)}):
                import importlib
                importlib.reload(config)
                assert getattr(config, env_var) == custom_value


class TestCurrentModelConfig:
    """Test current model configuration derivation"""
    
    def test_current_model_config_nomic_768(self):
        """Test current model config for nomic-moe-768"""
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "nomic-moe-768"}):
            import importlib
            importlib.reload(config)
            
            assert config.CURRENT_MODEL_CONFIG == config.MODEL_PRESETS["nomic-moe-768"]
            assert config.MODEL_NAME == "nomic-ai/nomic-embed-text-v2-moe"
            assert config.USE_MODEL2VEC is False
            assert config.EMBEDDING_DIMENSIONS == 768
    
    def test_current_model_config_nomic_256(self):
        """Test current model config for nomic-moe-256"""
        with patch.dict(os.environ, {"EMBEDDING_MODEL": "nomic-moe-256"}):
            import importlib
            importlib.reload(config)
            
            assert config.CURRENT_MODEL_CONFIG == config.MODEL_PRESETS["nomic-moe-256"]
            assert config.MODEL_NAME == "Abdelkareem/nomic-embed-text-v2-moe_distilled"
            assert config.USE_MODEL2VEC is True
            assert config.EMBEDDING_DIMENSIONS == 256


class TestCachePaths:
    """Test cache path generation and directory setup"""
    
    def test_project_root_path(self):
        """Test PROJECT_ROOT is correctly set"""
        assert config.PROJECT_ROOT.is_dir()
        assert config.PROJECT_ROOT.name == "nomic-embedding-v2-custom"
    
    def test_models_dir_creation(self):
        """Test MODELS_DIR is correctly set"""
        assert config.MODELS_DIR == config.PROJECT_ROOT / "models"
        # Should exist or be creatable
        assert config.MODELS_DIR.exists() or config.MODELS_DIR.parent.exists()
    
    def test_get_cache_path_function(self):
        """Test get_cache_path function"""
        test_cases = [
            ("simple-model", "simple-model"),
            ("user/model-name", "user--model-name"),
            ("registry:tag", "registry-tag"),
            ("complex/model:with:colons", "complex--model-with-colons"),
        ]
        
        for model_name, expected_dir in test_cases:
            cache_path = config.get_cache_path(model_name)
            assert cache_path.parent == config.MODELS_DIR
            assert cache_path.name == expected_dir
            assert isinstance(cache_path, Path)
    
    def test_model_cache_path_generation(self):
        """Test MODEL_CACHE_PATH is correctly generated"""
        # Should be based on current model name
        expected_path = config.get_cache_path(config.MODEL_NAME)
        assert config.MODEL_CACHE_PATH == expected_path


class TestRedisConfiguration:
    """Test Redis configuration options"""
    
    def test_redis_defaults(self):
        """Test Redis default configuration"""
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(config)
            
            assert config.REDIS_ENABLED is False
            assert config.REDIS_URL == "redis://localhost:6379"
            assert config.REDIS_DB == 0
            assert config.REDIS_KEY_PREFIX == "nomic_embedding"
            assert config.REDIS_MAX_CONNECTIONS == 20
            assert config.REDIS_RETRY_ON_TIMEOUT is True
            assert config.REDIS_CACHE_TTL == 3600
            assert config.REDIS_SESSION_TTL == 300
    
    def test_redis_custom_config(self):
        """Test Redis custom configuration"""
        custom_config = {
            "REDIS_ENABLED": "true",
            "REDIS_URL": "redis://custom-host:6380",
            "REDIS_DB": "5",
            "REDIS_KEY_PREFIX": "custom_prefix",
            "REDIS_MAX_CONNECTIONS": "50",
            "REDIS_RETRY_ON_TIMEOUT": "false",
            "REDIS_CACHE_TTL": "7200",
            "REDIS_SESSION_TTL": "600",
        }
        
        with patch.dict(os.environ, custom_config):
            import importlib
            importlib.reload(config)
            
            assert config.REDIS_ENABLED is True
            assert config.REDIS_URL == "redis://custom-host:6380"
            assert config.REDIS_DB == 5
            assert config.REDIS_KEY_PREFIX == "custom_prefix"
            assert config.REDIS_MAX_CONNECTIONS == 50
            assert config.REDIS_RETRY_ON_TIMEOUT is False
            assert config.REDIS_CACHE_TTL == 7200
            assert config.REDIS_SESSION_TTL == 600


class TestHuggingFaceConfiguration:
    """Test HuggingFace cache configuration"""
    
    def test_hf_cache_environment_variables(self):
        """Test HuggingFace cache environment variable setup"""
        # These should be set during config import
        assert "HF_HOME" in os.environ
        assert "TRANSFORMERS_CACHE" in os.environ
        assert "SENTENCE_TRANSFORMERS_HOME" in os.environ
        
        # Should point to models directory
        assert os.environ["SENTENCE_TRANSFORMERS_HOME"] == str(config.MODELS_DIR)
    
    def test_cache_dir_setting(self):
        """Test CACHE_DIR configuration"""
        # Should use HF_HOME if set, otherwise MODELS_DIR
        if "HF_HOME" in os.environ:
            expected_cache_dir = os.environ["HF_HOME"]
        else:
            expected_cache_dir = str(config.MODELS_DIR)
        
        assert config.CACHE_DIR == expected_cache_dir


class TestOptimizationConfiguration:
    """Test optimization-related configuration"""
    
    def test_optimization_level_validation(self):
        """Test optimization level validation"""
        valid_levels = ["conservative", "balanced", "aggressive"]
        
        for level in valid_levels:
            with patch.dict(os.environ, {"OPTIMIZATION_LEVEL": level}):
                import importlib
                importlib.reload(config)
                assert config.OPTIMIZATION_LEVEL == level
        
        # Test default
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            importlib.reload(config)
            assert config.OPTIMIZATION_LEVEL == "balanced"
    
    def test_hardware_optimization_flags(self):
        """Test hardware optimization boolean flags"""
        flags = [
            "ENABLE_CPU_AFFINITY",
            "ENABLE_NUMA_AWARENESS", 
            "AUTO_DEVICE_SELECTION",
            "ENABLE_TORCH_OPTIMIZATION",
        ]
        
        for flag in flags:
            # Test True
            with patch.dict(os.environ, {flag: "true"}):
                import importlib
                importlib.reload(config)
                assert getattr(config, flag) is True
            
            # Test False  
            with patch.dict(os.environ, {flag: "false"}):
                import importlib
                importlib.reload(config)
                assert getattr(config, flag) is False


class TestPerformanceConfiguration:
    """Test performance-related configuration"""
    
    def test_batch_configuration(self):
        """Test batch processing configuration"""
        assert config.MAX_BATCH_SIZE > 0
        assert config.BATCH_TIMEOUT_MS > 0
        assert config.MAX_CONCURRENT_REQUESTS > 0
    
    def test_cache_configuration(self):
        """Test cache configuration"""
        assert config.CACHE_SIZE > 0
        assert isinstance(config.ENABLE_CACHING, bool)
    
    def test_model_pool_configuration(self):
        """Test model pool configuration"""
        assert config.MODEL_POOL_SIZE >= 0
        assert isinstance(config.ENABLE_MULTI_GPU, bool)
        assert config.HEALTH_CHECK_INTERVAL > 0
        assert config.MAX_MODEL_ERROR_COUNT > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])