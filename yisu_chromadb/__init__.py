"""
ChromaDB工具包

提供ChromaDB相关的工具函数和类
"""
from .chroma_utils import (
    get_embedding_function,
    get_chroma_client,
    create_client_config,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_API_BASE
)

__all__ = [
    'get_embedding_function',
    'get_chroma_client',
    'create_client_config',
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_OPENAI_EMBEDDING_MODEL',
    'DEFAULT_OPENAI_API_BASE'
]
