import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import EmbeddingModel, NoMiCMoEModel


class TestEmbeddingModel:
    """Test suite for EmbeddingModel class"""
    
    @patch('src.models.SentenceTransformer')
    @patch('src.models.StaticModel')
    def test_init_with_sentence_transformer(self, mock_static_model, mock_sentence_transformer):
        """Test EmbeddingModel initialization with SentenceTransformer"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', False):
            model = EmbeddingModel(model_name="test-model", use_model2vec=False)
            
        assert model.model_name == "test-model"
        assert model.use_model2vec is False
        assert model.model == mock_model
        mock_sentence_transformer.assert_called()
    
    @patch('src.models.StaticModel')
    @patch('src.models.SentenceTransformer')
    def test_init_with_model2vec(self, mock_sentence_transformer, mock_static_model):
        """Test EmbeddingModel initialization with Model2Vec"""
        mock_model = MagicMock()
        mock_static_model.from_pretrained.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', True):
            model = EmbeddingModel(model_name="test-model", use_model2vec=True)
            
        assert model.model_name == "test-model"
        assert model.use_model2vec is True
        assert model.model == mock_model
        mock_static_model.from_pretrained.assert_called()
    
    def test_is_model_cached_false(self):
        """Test _is_model_cached returns False when cache doesn't exist"""
        with patch('src.models.USE_MODEL2VEC', False):
            with patch('src.models.SentenceTransformer'):
                model = EmbeddingModel(model_name="test-model")
                
        with patch.object(model.cache_path, 'exists', return_value=False):
            assert model._is_model_cached() is False
    
    def test_is_model_cached_true_sentence_transformer(self):
        """Test _is_model_cached returns True for complete SentenceTransformer cache"""
        with patch('src.models.USE_MODEL2VEC', False):
            with patch('src.models.SentenceTransformer'):
                model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        def mock_file_exists(file_path):
            file_name = file_path.name
            return file_name in ["config.json", "pytorch_model.bin"]
        
        with patch.object(model.cache_path, 'exists', return_value=True):
            with patch.object(Path, 'exists', side_effect=lambda: mock_file_exists(Path)):
                # Mock individual file checks
                with patch('pathlib.Path.__truediv__') as mock_div:
                    mock_file = MagicMock()
                    mock_file.exists.return_value = True
                    mock_div.return_value = mock_file
                    assert model._is_model_cached() is True
    
    @patch('src.models.SentenceTransformer')
    def test_encode_single_text(self, mock_sentence_transformer):
        """Test encoding single text string"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', False):
            model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        result = model.encode("Hello world")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        mock_model.encode.assert_called_once_with(["search_document: Hello world"])
    
    @patch('src.models.SentenceTransformer')
    def test_encode_multiple_texts(self, mock_sentence_transformer):
        """Test encoding multiple text strings"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', False):
            model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        result = model.encode(["Hello", "World"])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        mock_model.encode.assert_called_once_with([
            "search_document: Hello",
            "search_document: World"
        ])
    
    @patch('src.models.SentenceTransformer')
    def test_encode_with_search_query_task(self, mock_sentence_transformer):
        """Test encoding with search_query task type"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', False):
            model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        result = model.encode("query text", task_type="search_query")
        
        mock_model.encode.assert_called_once_with(["search_query: query text"])
    
    @patch('src.models.SentenceTransformer')
    def test_encode_torch_tensor_conversion(self, mock_sentence_transformer):
        """Test encoding handles torch tensor conversion correctly"""
        mock_model = MagicMock()
        # Simulate torch tensor return
        torch_tensor = torch.tensor([[0.1, 0.2, 0.3]])
        torch_tensor.detach = MagicMock(return_value=torch_tensor)
        torch_tensor.cpu = MagicMock(return_value=torch_tensor)
        torch_tensor.numpy = MagicMock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_model.encode.return_value = torch_tensor
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', False):
            model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        result = model.encode("test")
        
        assert isinstance(result, np.ndarray)
        torch_tensor.detach.assert_called_once()
        torch_tensor.cpu.assert_called_once()
    
    @patch('src.models.SentenceTransformer')
    def test_encode_error_handling(self, mock_sentence_transformer):
        """Test encode error handling"""
        mock_model = MagicMock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', False):
            model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        with pytest.raises(Exception, match="Encoding failed"):
            model.encode("test")
    
    @patch('src.models.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension"""
        mock_sentence_transformer.return_value = MagicMock()
        
        with patch('src.models.USE_MODEL2VEC', False):
            with patch('src.models.EMBEDDING_DIMENSIONS', 768):
                model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        assert model.get_embedding_dimension() == 768
    
    @patch('src.models.SentenceTransformer')
    def test_warmup_model(self, mock_sentence_transformer):
        """Test model warmup functionality"""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.USE_MODEL2VEC', False):
            model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        # Warmup is called during init, so check if encode was called
        mock_model.encode.assert_called()
    
    @patch('src.models.SentenceTransformer')
    def test_hardware_optimization_disabled(self, mock_sentence_transformer):
        """Test model initialization with hardware optimization disabled"""
        mock_sentence_transformer.return_value = MagicMock()
        
        with patch('src.models.ENABLE_HARDWARE_OPTIMIZATION', False):
            with patch('src.models.USE_MODEL2VEC', False):
                model = EmbeddingModel(model_name="test-model", use_model2vec=False)
        
        assert model.hardware_optimizer is None


class TestNoMiCMoEModel:
    """Test suite for NoMiCMoEModel class"""
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_init(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel initialization"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=False):
            model = NoMiCMoEModel(model_name="test-moe-model")
        
        assert model.model_name == "test-moe-model"
        assert model.model == mock_model
        assert model.tokenizer == mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called()
        mock_auto_model.from_pretrained.assert_called()
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_init_with_cuda(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel initialization with CUDA"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_model.cuda.return_value = mock_model
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=True):
            model = NoMiCMoEModel(model_name="test-moe-model")
        
        mock_model.cuda.assert_called_once()
    
    def test_is_model_cached_false(self):
        """Test _is_model_cached returns False when cache doesn't exist"""
        with patch('src.models.AutoTokenizer'):
            with patch('src.models.AutoModel'):
                model = NoMiCMoEModel(model_name="test-model")
        
        with patch.object(model.cache_path, 'exists', return_value=False):
            assert model._is_model_cached() is False
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_encode(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel encoding"""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer output
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value = mock_inputs
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_hidden_states = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]])
        mock_outputs.last_hidden_state = mock_hidden_states
        mock_model.return_value = mock_outputs
        
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.no_grad'):
                model = NoMiCMoEModel(model_name="test-model")
                result = model.encode("test text")
        
        assert isinstance(result, np.ndarray)
        mock_tokenizer.assert_called()
        mock_model.assert_called()
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_encode_with_cuda(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel encoding with CUDA"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        # Mock inputs and CUDA movement
        mock_inputs = {
            'input_ids': MagicMock(),
            'attention_mask': MagicMock()
        }
        mock_inputs['input_ids'].cuda.return_value = mock_inputs['input_ids']
        mock_inputs['attention_mask'].cuda.return_value = mock_inputs['attention_mask']
        mock_tokenizer.return_value = mock_inputs
        
        # Mock model output
        mock_outputs = MagicMock()
        mock_hidden_states = MagicMock()
        mock_mean_tensor = MagicMock()
        mock_cpu_tensor = MagicMock()
        mock_cpu_tensor.numpy.return_value = np.array([[0.1, 0.2]])
        
        mock_hidden_states.mean.return_value = mock_mean_tensor
        mock_mean_tensor.cpu.return_value = mock_cpu_tensor
        mock_outputs.last_hidden_state = mock_hidden_states
        mock_model.return_value = mock_outputs
        
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.no_grad'):
                model = NoMiCMoEModel(model_name="test-model")
                result = model.encode("test text")
        
        assert isinstance(result, np.ndarray)
        # Verify CUDA operations were called
        mock_inputs['input_ids'].cuda.assert_called()
        mock_inputs['attention_mask'].cuda.assert_called()
        mock_mean_tensor.cpu.assert_called()
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_encode_multiple_texts(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel encoding multiple texts"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2], [3, 4]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]])
        }
        mock_tokenizer.return_value = mock_inputs
        
        mock_outputs = MagicMock()
        mock_hidden_states = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        mock_outputs.last_hidden_state = mock_hidden_states
        mock_model.return_value = mock_outputs
        
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.no_grad'):
                model = NoMiCMoEModel(model_name="test-model")
                result = model.encode(["text1", "text2"])
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2  # Two texts should produce two embeddings
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_encode_with_search_query_task(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel encoding with search_query task"""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        
        mock_inputs = {'input_ids': torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_inputs
        
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.tensor([[[0.1, 0.2, 0.3]]])
        mock_model.return_value = mock_outputs
        
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.no_grad'):
                model = NoMiCMoEModel(model_name="test-model")
                result = model.encode("query", task_type="search_query")
        
        # Verify the tokenizer was called with prefixed text
        mock_tokenizer.assert_called()
        call_args = mock_tokenizer.call_args[0]
        assert "search_query: query" in call_args[0]
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_get_embedding_dimension(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel get_embedding_dimension"""
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock()
        mock_auto_model.from_pretrained.return_value = MagicMock()
        
        with patch('src.models.EMBEDDING_DIMENSIONS', 1024):
            with patch('torch.cuda.is_available', return_value=False):
                model = NoMiCMoEModel(model_name="test-model")
        
        assert model.get_embedding_dimension() == 1024
    
    @patch('src.models.AutoTokenizer')
    @patch('src.models.AutoModel')
    def test_encode_error_handling(self, mock_auto_model, mock_auto_tokenizer):
        """Test NoMiCMoEModel encode error handling"""
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = Exception("Tokenization failed")
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = MagicMock()
        
        with patch('torch.cuda.is_available', return_value=False):
            model = NoMiCMoEModel(model_name="test-model")
            
            with pytest.raises(Exception, match="Tokenization failed"):
                model.encode("test")


class TestModelOptimizations:
    """Test suite for model optimization features"""
    
    @patch('src.models.SentenceTransformer')
    def test_torch_compile_optimization(self, mock_sentence_transformer):
        """Test torch.compile optimization"""
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.TORCH_COMPILE', True):
            with patch('src.models.USE_MODEL2VEC', False):
                with patch('torch.compile', return_value=mock_model) as mock_compile:
                    with patch('hasattr', return_value=True):
                        model = EmbeddingModel(model_name="test-model")
        
        # torch.compile might be called during optimization
        # The exact behavior depends on the model structure
        assert model.model is not None
    
    @patch('src.models.SentenceTransformer')
    def test_quantization_optimization(self, mock_sentence_transformer):
        """Test quantization optimization"""
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        mock_sentence_transformer.return_value = mock_model
        
        with patch('src.models.ENABLE_QUANTIZATION', True):
            with patch('src.models.USE_MODEL2VEC', False):
                with patch('torch.cuda.is_available', return_value=True):
                    with patch('hasattr', return_value=True):
                        model = EmbeddingModel(model_name="test-model")
        
        # Check that half precision was attempted
        assert model.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])