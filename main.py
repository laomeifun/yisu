"""
主程序入口

提供主要功能的访问入口
"""
from typing import Optional, Tuple
import chromadb
from chromadb.api.types import Document, EmbeddingFunction
from dbtools import *
from memory import (
    Memory,
    MemoryQueryResult,
    create_memory,
    create_memory_from_dict,
    convert_memory_to_dict
)

def init_chromadb() -> Tuple[Optional[chromadb.Client], Optional[EmbeddingFunction]]:
    """初始化ChromaDB"""
    # 创建客户端配置
    config = create_client_config("persistent", data_dir=".chromadb")
    
    # 获取embedding function
    embedding_fn = get_embedding_function()
    
    # 获取ChromaDB客户端
    client = get_chroma_client(config)
    
    if client and embedding_fn:
        print("ChromaDB工具初始化成功")
        print(f"使用默认embedding模型: {DEFAULT_EMBEDDING_MODEL}")
        return client, embedding_fn
    else:
        print("ChromaDB工具初始化失败")
        return None, None

def demo_collection_operations(client, embedding_fn) -> None:
    """演示集合操作"""
    Collection_name = "test_collection" 
    collection = get_or_create_collection(client=client, collection_name=Collection_name, embedding_function=embedding_fn)     
    print(f"集合 {Collection_name} 创建成功")

    # 插入数据
    add_documents(collection_name=Collection_name, documents=["test document 1", "test document 2"], ids=["id1", "id2"],client=client)    
    print("数据插入成功")
    
    # 查询数据
    results = query_documents(collection_name=Collection_name, include=["metadatas", "distances"], query_texts=["test query"],client=client)
    print("查询结果:", results)
    
    # 删除集合
    delete_collection(collection_name=Collection_name,client=client)
    print("集合删除成功")

def demo_memory_operations(client, embedding_fn) -> None:
    """演示记忆操作"""
    print("\n=== 开始记忆操作演示 ===")
    
    # 创建示例记忆
    memory1, error = create_memory(
        content="今天是个好天气，阳光明媚。",
        tags=["天气", "心情"],
        memory_type="日记",
        metadata={"location": "北京", "temperature": "25°C"}
    )
    if error:
        print(f"创建记忆失败: {error}")
        return
    
    memory2, error = create_memory(
        content="完成了项目的重要里程碑，团队表现出色。",
        tags=["工作", "项目"],
        memory_type="工作记录",
        metadata={"project": "AI助手", "milestone": "1.0版本"}
    )
    if error:
        print(f"创建记忆失败: {error}")
        return
    
    # 存储记忆
    print("\n正在存储记忆...")
    for memory in [memory1, memory2]:
        memory_dict = convert_memory_to_dict(memory)
        success = store_memory(
            content=memory.content,
            metadata=memory_dict,
            memory_id=memory.content_hash,
            client=client,
            embedding_function=embedding_fn
        )
        if success:
            print(f"成功存储记忆: {memory.content[:30]}...")
        else:
            print(f"存储记忆失败: {memory.content[:30]}...")
    
    # 检索记忆
    print("\n正在检索记忆...")
    query_text = "天气"
    results = retrieve_memory(
        query_text=query_text,
        n_results=5,
        client=client,
        embedding_function=embedding_fn
    )
    
    print(f"\n查询 '{query_text}' 的结果:")
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results.get("distances") else None
            relevance = 1 - distance if distance is not None else None
            relevance_str = f"(相关度: {relevance:.2f})" if relevance is not None else ""
            print(f"- {doc[:50]}... {relevance_str}")
    else:
        print("未找到相关记忆")
    
    print("\n=== 记忆操作演示结束 ===")

def main() -> None:
    client, embedding_fn = init_chromadb()
    if client and embedding_fn:
        demo_collection_operations(client, embedding_fn)
        demo_memory_operations(client, embedding_fn)
    else:
        print("ChromaDB工具初始化失败，无法进行演示操作")

if __name__ == "__main__":
    main()





