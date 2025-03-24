from typing import Optional, Tuple, List, Dict, Any
import argparse
import sys
import chromadb
from chromadb.api.types import Document, EmbeddingFunction
from openai.types import embedding
from dbtools.chroma_utils import (
    get_chroma_client,
    create_client_config,
    get_embedding_function,
    create_collection,
    get_or_create_collection,
    list_collections,
    peek_collection,
    query_collection,
    get_documents,
    add_document_to_collection
)  # 使用更新后的导入

from memory.memory_service import (
    save_memory, 
    search_memories, 
    get_memory_by_id,
    delete_memory,
    count_memories
)  # 使用高级内存服务API

from memory.memory import (
    Memory,
    MemoryQueryResult,
    create_memory,
    create_query_result
)


MEMORY_COLLECTION_NAME = "memory"  # 集合名称

from mcp.server.fastmcp import FastMCP

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Yisu MCP 服务器')
    
    # 数据库连接参数
    db_group = parser.add_argument_group('数据库配置')
    db_group.add_argument('--client-type', type=str, default='persistent', 
                      choices=['persistent', 'http', 'ephemeral'], 
                      help='ChromaDB 客户端类型')
    db_group.add_argument('--data-dir', type=str, default="./.chromadb", 
                      help='ChromaDB 数据目录 (persistent 模式)')
    db_group.add_argument('--host', type=str, help='ChromaDB 服务器地址 (http 模式)')
    db_group.add_argument('--db-port', type=str, help='ChromaDB 服务器端口 (http 模式)')
    
    # 嵌入模型参数
    emb_group = parser.add_argument_group('嵌入模型配置')
    emb_group.add_argument('--embedding-provider', type=str, default='sentence_transformer',
                       choices=['sentence_transformer', 'openai', 'default'],
                       help='嵌入模型提供商')
    emb_group.add_argument('--embedding-model', type=str, 
                       help='嵌入模型名称')
    emb_group.add_argument('--openai-api-key', type=str, help='OpenAI API 密钥')
    emb_group.add_argument('--openai-api-base', type=str, help='OpenAI API 基础 URL')
    
    # MCP 服务器参数
    server_group = parser.add_argument_group('服务器配置')
    server_group.add_argument('--transport', type=str, default='stdio',
                          choices=['stdio', 'http', 'websocket'],
                          help='MCP 服务器传输方式')
    server_group.add_argument('--port', type=int, default=3000,
                          help='MCP 服务器端口号 (http/websocket 模式)')
    
    return parser.parse_args()

def init_chromadb(args: argparse.Namespace) -> Optional[Tuple[chromadb.Client, EmbeddingFunction]]:
    """初始化 ChromaDB"""
    try:
        # 创建客户端配置
        config = create_client_config(
            client_type=args.client_type,
            data_dir=args.data_dir if args.client_type == 'persistent' else None,
            host=args.host if args.client_type == 'http' else None,
            port=args.db_port if args.client_type == 'http' else None
        )
        
        # 获取 ChromaDB 客户端
        client = get_chroma_client(config)
        
        # 创建嵌入函数
        embedding_fn = get_embedding_function(
            provider=args.embedding_provider,
            model_name=args.embedding_model,
            openai_api_key=args.openai_api_key,
            openai_api_base=args.openai_api_base
        )
        
        if client and embedding_fn:
            print(f"ChromaDB 工具初始化成功 (类型: {args.client_type})")
            print(f"嵌入模型: {args.embedding_provider}" + 
                  (f" - {args.embedding_model}" if args.embedding_model else ""))
            return client, embedding_fn
        else:
            print("ChromaDB 工具或嵌入函数初始化失败")
            return None
    except Exception as e:
        print(f"ChromaDB 初始化出错: {str(e)}")
        return None

# 初始化集合或者获取现有集合
def init_collection(client: chromadb.Client, collection_name: str = MEMORY_COLLECTION_NAME, embedding_fn: EmbeddingFunction = None) -> Optional[chromadb.Collection]:
    """初始化集合"""
    try:
        # 创建集合
        collection = get_or_create_collection(
            collection_name=collection_name,
            client=client,
            embedding_function=embedding_fn
        )
        
        if collection:
            count = collection.count()
            print(f"集合 '{collection_name}' 初始化成功，包含 {count} 条记录")
            return collection
        else:
            print(f"集合 '{collection_name}' 初始化失败")
            return None
    except Exception as e:
        print(f"初始化集合出错: {str(e)}")
        return None

# 全局变量，但仅在模块初始化时设置一次
args = parse_arguments()
mcp = FastMCP("yisu")
chroma_client = None
embedding_fn = None
collection = None

# 初始化数据库
chroma_result = init_chromadb(args)
if chroma_result:
    chroma_client, embedding_fn = chroma_result
    # 初始化集合
    collection = init_collection(chroma_client, MEMORY_COLLECTION_NAME, embedding_fn)
else:
    print("ChromaDB 初始化失败，无法继续操作")

@mcp.tool()
def add_memory(
    content: str,
    tags: List[str] = None,
    memory_type: str = None,
    metadata: Dict[str, Any] = None
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
    if not chroma_client or not embedding_fn:
        return "ChromaDB 客户端未初始化，无法存储记忆"
    
    # 使用高级记忆服务 API 保存记忆
    memory_id, error = save_memory(
        content=content,
        tags=tags or [],
        memory_type=memory_type,
        metadata=metadata or {},
        embedding_function=embedding_fn
    )
    
    if error:
        return f"保存记忆失败: {error}"
    
    return f"成功保存记忆 (ID: {memory_id}): {content[:50]}..."

@mcp.tool()
def search_memory(
    query: str,
    limit: int = 5,
    tags: List[str] = None,
    memory_type: str = None
) -> str:
    """
    搜索记忆
    
    参数:
        query: 查询内容
        limit: 返回结果数量上限
        tags: 按标签筛选
        memory_type: 按记忆类型筛选
    
    返回:
        str: 格式化的查询结果
    """
    if not chroma_client or not embedding_fn:
        return "ChromaDB 客户端未初始化，无法查询记忆"
    
    # 使用高级记忆服务 API 搜索记忆
    results = search_memories(
        query=query,
        limit=limit,
        tags=tags,
        memory_type=memory_type,
        embedding_function=embedding_fn
    )
    
    if not results:
        return "未找到相关记忆"
    
    # 格式化结果
    formatted_results = []
    for i, result in enumerate(results):
        mem = result.memory
        formatted_results.append(
            f"结果 #{i+1} (相关性: {result.relevance_score:.2f})\n"
            f"内容: {mem.content[:200]}...\n"
            f"类型: {mem.memory_type or '无'} | 标签: {', '.join(mem.tags) if mem.tags else '无'}\n"
        )
    
    return "\n".join(formatted_results)

@mcp.tool()
def get_memory(memory_id: str) -> str:
    """
    获取指定 ID 的记忆详情
    
    参数:
        memory_id: 记忆 ID
    
    返回:
        str: 记忆详情
    """
    if not chroma_client:
        return "ChromaDB 客户端未初始化"
    
    memory = get_memory_by_id(memory_id)
    
    if not memory:
        return f"未找到 ID 为 {memory_id} 的记忆"
    
    return (
        f"记忆 ID: {memory_id}\n"
        f"内容: {memory.content}\n"
        f"类型: {memory.memory_type or '无'}\n"
        f"标签: {', '.join(memory.tags) if memory.tags else '无'}\n"
        f"时间: {memory.timestamp}\n"
        f"元数据: {memory.metadata}"
    )

@mcp.tool()
def remove_memory(memory_id: str) -> str:
    """
    删除指定 ID 的记忆
    
    参数:
        memory_id: 记忆 ID
    
    返回:
        str: 操作结果
    """
    if not chroma_client:
        return "ChromaDB 客户端未初始化"
    
    success = delete_memory(memory_id)
    
    if success:
        return f"成功删除记忆 {memory_id}"
    else:
        return f"删除记忆 {memory_id} 失败"

@mcp.tool()
def list_memory_stats() -> str:
    """
    获取记忆统计信息
    
    返回:
        str: 记忆统计信息
    """
    if not chroma_client:
        return "ChromaDB 客户端未初始化"
    
    total = count_memories()
    
    if total < 0:
        return "获取记忆统计信息失败"
    
    result = f"记忆总数: {total}\n"
    
    # 获取记忆类型统计
    if collection:
        try:
            all_memories = collection.get(include=["metadatas"])
            if all_memories and "metadatas" in all_memories:
                # 统计记忆类型
                memory_types = {}
                for metadata in all_memories["metadatas"]:
                    memory_type = metadata.get("type")
                    if memory_type:
                        memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                
                if memory_types:
                    result += "记忆类型分布:\n"
                    for memory_type, count in memory_types.items():
                        result += f"- {memory_type}: {count} 条\n"
        except Exception as e:
            result += f"获取详细统计信息出错: {str(e)}"
    
    return result

def main():
    """主函数"""
    try:
        # 启动 MCP 服务器
        if args.transport == 'http':
            mcp.run(transport="http", port=args.port)
        elif args.transport == 'websocket':
            mcp.run(transport="websocket", port=args.port)
        else:  # 默认为 stdio
            mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器运行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
