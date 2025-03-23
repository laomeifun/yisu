import pytest
from unittest.mock import patch, MagicMock
from chromadb.api.types import EmbeddingFunction
from chromadb.chroma_utils import (  # 修改这里：chromedb -> chromadb
    get_embedding_function,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL
)

# 测试默认参数
def test_get_embedding_function_default():
    with patch('chromadb.chroma_utils._get_sentence_transformer_embedding_function') as mock_st:  # 修改这里
        mock_st.return_value = MagicMock(spec=EmbeddingFunction)
        embedding_fn = get_embedding_function()
        
        mock_st.assert_called_once_with(DEFAULT_EMBEDDING_MODEL)
        assert isinstance(embedding_fn, EmbeddingFunction)

# 测试 sentence_transformer provider
def test_get_embedding_function_sentence_transformer():
    custom_model = "custom-model"
    with patch('chromadb.chroma_utils._get_sentence_transformer_embedding_function') as mock_st:  # 修改这里
        mock_st.return_value = MagicMock(spec=EmbeddingFunction)
        embedding_fn = get_embedding_function(
            provider="sentence_transformer",
            model_name=custom_model
        )
        
        mock_st.assert_called_once_with(custom_model)
        assert isinstance(embedding_fn, EmbeddingFunction)

# 测试 OpenAI provider
def test_get_embedding_function_openai():
    api_key = "test-api-key"
    api_base = "https://custom-api-base.com"
    with patch('chromadb.chroma_utils._get_openai_embedding_function') as mock_openai:  # 修改这里
        mock_openai.return_value = MagicMock(spec=EmbeddingFunction)
        embedding_fn = get_embedding_function(
            provider="openai",
            openai_api_key=api_key,
            openai_api_base=api_base
        )
        
        mock_openai.assert_called_once_with(
            DEFAULT_OPENAI_EMBEDDING_MODEL,
            api_key,
            api_base
        )
        assert isinstance(embedding_fn, EmbeddingFunction)

# 测试无效的 provider
def test_get_embedding_function_invalid_provider():
    with patch('chromadb.utils.embedding_functions.DefaultEmbeddingFunction') as mock_default:
        mock_default.return_value = MagicMock(spec=EmbeddingFunction)
        embedding_fn = get_embedding_function(provider="invalid")
        
        mock_default.assert_called_once()
        assert isinstance(embedding_fn, EmbeddingFunction)

# 测试 sentence_transformer 导入错误
def test_sentence_transformer_import_error():
    with patch('chromadb.chroma_utils._get_sentence_transformer_embedding_function') as mock_st:  # 修改这里
        mock_st.return_value = None
        embedding_fn = get_embedding_function(provider="sentence_transformer")
        assert embedding_fn is None

# 测试 OpenAI 缺少 API key
def test_openai_missing_api_key():
    embedding_fn = get_embedding_function(provider="openai")
    assert embedding_fn is None

# 测试 OpenAI 导入错误
def test_openai_import_error():
    with patch('chromadb.chroma_utils._get_openai_embedding_function') as mock_openai:  # 修改这里
        mock_openai.return_value = None
        embedding_fn = get_embedding_function(
            provider="openai",
            openai_api_key="test-key"
        )
        assert embedding_fn is None
