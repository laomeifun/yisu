from typing import Optional, Tuple, List
import chromadb
from chromadb.api.types import Document, EmbeddingFunction
from dbtools import *  # 使用相对导入
from memory import (  # 使用相对导入
    Memory,
    MemoryQueryResult,
    create_memory,
    create_memory_from_dict,
    convert_memory_to_dict
)


from mcp.server.fastmcp import FastMCP
def init_chromadb() -> Optional[chromadb.Client]:
    """初始化ChromaDB"""
    # 创建客户端配置
    config = create_client_config("persistent", data_dir=".chromadb")
    
    # 获取ChromaDB客户端
    client = get_chroma_client(config)
    
    if client:
        print("ChromaDB工具初始化成功")
        print(f"使用默认embedding模型: {DEFAULT_EMBEDDING_MODEL}")
        return client
    else:
        print("ChromaDB工具初始化失败")
        return None


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




mcp = FastMCP("yisu")


client = init_chromadb()

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
        client=client
    )
    
    if success:
        return f"成功存储记忆: {content[:50]}..."
    else:
        return "存储记忆失败"

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
def main():
    mcp.run()
if __name__ == "__main__":
    main()
