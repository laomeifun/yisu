"""
ChromaDB工具包

提供ChromaDB相关的工具函数和类
"""
# 导入所有需要的内容
from .chroma_utils import (
    get_chroma_client,
    create_client_config,
    get_embedding_function,
    create_collection,
    get_or_create_collection,
    list_collections,
    peek_collection,
    query_documents,
    get_documents,
    retrieve_memory,
    add_documents,
    delete_collection,
    store_memory,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENAI_API_BASE,
    DEFAULT_OPENAI_EMBEDDING_MODEL
)

# 明确列出要导出的内容
__all__ = [
    # 客户端相关
    'get_chroma_client',
    'create_client_config',
    
    # 嵌入函数相关
    'get_embedding_function',
    
    # 集合操作相关
    'create_collection',
    'get_or_create_collection',
    'list_collections',
    'peek_collection',
    'query_documents',
    'get_documents',
    'retrieve_memory',
    'add_documents',
    'query_documents',  
    'delete_collection',
    'store_memory',
    'retrieve_memory',
    
    # 常量
    'DEFAULT_EMBEDDING_MODEL',
    'DEFAULT_OPENAI_EMBEDDING_MODEL',
    'DEFAULT_OPENAI_API_BASE'
]
