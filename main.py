"""
主程序入口

提供主要功能的访问入口
"""
from typing import Optional
from yisu_chromadb import (
    get_embedding_function,
    get_chroma_client,
    create_client_config,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_API_BASE
)

def main() -> None:
    """主程序入口函数"""
    # 创建客户端配置
    config = create_client_config("persistent", data_dir=".chromadb")
    
    # 获取embedding function
    # embedding_fn = get_embedding_function()
    
    # 获取ChromaDB客户端
    client = get_chroma_client(config)
    
    if client:
        print("ChromaDB工具初始化成功")
        print(f"使用默认embedding模型: {DEFAULT_EMBEDDING_MODEL}")
    else:
        print("ChromaDB工具初始化失败")

if __name__ == "__main__":
    main()


