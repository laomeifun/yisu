from typing import Optional, Tuple, List
import chromadb
from chromadb.api.types import Document, EmbeddingFunction
from openai.types import embedding
from dbtools import (
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
)  # 使用相对导入
from memory import (  # 使用相对导入
    Memory,
    MemoryQueryResult,
    create_memory,
    create_memory_from_dict,
    convert_memory_to_dict
)


MEMORY_COLLECTION_NAME = "memory_collection"  # 集合名称

from mcp.server.fastmcp import FastMCP
def init_chromadb() -> Optional[Tuple[chromadb.Client, EmbeddingFunction]]:
    """初始化ChromaDB"""
    # 创建客户端配置
    config= create_client_config("persistent", data_dir="D:\data\.chromadb")
    
    # 获取ChromaDB客户端
    client = get_chroma_client(config)
    # 创建嵌入函数
    # embedding_fn = get_embedding_function("openai",model_name="BAAI/bge-m3",openai_api_base="https://api.siliconflow.cn/v1/chat/completions",openai_api_key="sk-mqyloehefcafwafszzlrusuebmirwlprxawhfzdcbatjsirm")
    embedding_fn = get_embedding_function()
    if client:
        print("ChromaDB工具初始化成功")
        # print(f"使用默认embedding模型: {DEFAULT_EMBEDDING_MODEL}")
        return client, embedding_fn
    else:
        print("ChromaDB工具初始化失败")
        return None

#初始化集合或者获取现有集合
def init_collection(client: chromadb.Client, collection_name: str = MEMORY_COLLECTION_NAME, embedding_fn: EmbeddingFunction = None) -> Optional[chromadb.Collection]:
    """初始化集合"""
    # 创建集合
    collection = get_or_create_collection(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_fn
    )
    
    if collection:
        print(f"集合 '{collection_name}' 初始化成功")
        return collection
    else:
        print(f"集合 '{collection_name}' 初始化失败")
        return None

mcp = FastMCP("yisu")

# 正确解包初始化返回的元组
chroma_result = init_chromadb()
if chroma_result:
    chroma_client, embedding_fn = chroma_result
    # 初始化集合
    collection = init_collection(chroma_client, MEMORY_COLLECTION_NAME, embedding_fn)
else:
    chroma_client = None
    collection = None
    print("ChromaDB初始化失败，无法继续操作")

@mcp.tool()
def add_memory(
    content: str,
    tags: List[str] = None,
    memory_type: str = None,
    metadata: dict = None
) -> str:
    """
    添加一条新的记忆
    
    参数:
        content: 记忆内容
        tags: 标签列表
        memory_type: 记忆类型
        metadata: 额外的元数据
    
    返回:
        str: 操作结果消息
    """
    if not chroma_client:
        return "ChromaDB客户端未初始化，无法存储记忆"
        
    # 创建记忆对象
    memory, error = create_memory(
        content=content,
        tags=tags or [],
        memory_type=memory_type,
        metadata=metadata or {}
    )
    
    if error:
        return f"创建记忆失败: {error}"
    
    # 转换记忆为字典格式并存储
    memory_dict = convert_memory_to_dict(memory)
    success = store_memory(
        content=memory.content,
        metadata=memory_dict,
        memory_id=memory.content_hash,
        client=chroma_client
    )
    
    if success:
        return f"成功存储记忆: {content[:50]}..."
    else:
        return "存储记忆失败"


@mcp.tool()
def query_memory(
    query: str,
    n_results: int = 5
) -> List[MemoryQueryResult]:
    """
    查询记忆
    
    参数:
        query: 查询内容
        n_results: 返回结果数量
    
    返回:
        List[MemoryQueryResult]: 查询结果列表
    """
    if not chroma_client:
        return "ChromaDB客户端未初始化，无法查询记忆"
        
    # 执行查询（修改参数名称：query -> query_text）
    results = retrieve_memory(
        client=chroma_client,
        query_text=query,  # 修改为正确的参数名
        n_results=n_results
    )
    
    if results:
        return results
    else:
        return "查询失败或无结果"


    
def main():
    mcp.run(transport="stdio")
if __name__ == "__main__":
    main()
