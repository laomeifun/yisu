"""
Chroma 数据库工具模块

该模块提供了对Chroma数据库的各种操作，包括连接管理、集合操作和文档管理。
主要功能包括：
1. 创建和管理Chroma客户端连接
2. 集合(Collection)的创建、查询、修改和删除
3. 文档的添加、查询和获取

作者: laomei
日期: 2025-03-23
"""

from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.utils import embedding_functions  # 修改这行
from typing import Optional, Union, Dict, Any
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import ssl
import os

# 定义常量
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


# 默认嵌入
# DEFAULT_EMBEDDING = embedding_functions.DefaultEmbeddingFunction()

def get_embedding_function(provider: str = "sentence_transformer",
                           model_name: Optional[str] = None,
                           openai_api_key: Optional[str] = None,
                           openai_api_base: Optional[str] = None,
                           ) -> Union[EmbeddingFunction[Documents], None]:
    """
    获取嵌入函数，用于将文本转换为向量表示。
    支持多种嵌入提供者，包括sentence-transformers和OpenAI。

    参数:
        provider: 嵌入提供者，可选值: "sentence_transformer", "openai"
        model_name: 要使用的模型名称，如未提供则使用各提供者的默认值
        openai_api_key: OpenAI API密钥 (仅OpenAI提供者需要)
        openai_api_base: OpenAI API基础URL (可选)

    返回:
        嵌入函数或None（如果无法创建嵌入函数）
    """
    if provider == "sentence_transformer":
        if model_name is None:
            model_name = DEFAULT_EMBEDDING_MODEL
        return _get_sentence_transformer_embedding_function(model_name)
    elif provider == "openai":
        if model_name is None:
            model_name = DEFAULT_OPENAI_EMBEDDING_MODEL
        return _get_openai_embedding_function(model_name, openai_api_key, openai_api_base)
    else:
        return embedding_functions.DefaultEmbeddingFunction()


"""获取Sentence Transformer嵌入函数（内部函数）"""


def _get_sentence_transformer_embedding_function(model_name: str) -> Union[EmbeddingFunction[Documents], None]:
    try:
        # 尝试导入sentence_transformers，如果不存在则捕获异常
        from sentence_transformers import SentenceTransformer

        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    except ImportError:
        print("警告: sentence-transformers库未安装，无法创建嵌入函数。")
        print("提示: 请运行 'pip install sentence-transformers' 安装所需依赖。")
        return None


"""获取OpenAI嵌入函数（内部函数）"""


def _get_openai_embedding_function(model_name: str, openai_api_key: Optional[str] = None, openai_api_base: Optional[str] = None) -> Union[EmbeddingFunction[Documents], None]:
    if not openai_api_key:
        print("错误: 使用OpenAI嵌入时必须提供API密钥。")
        print("请通过参数传入API密钥或设置OPENAI_API_KEY环境变量。")
        return None

    try:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=model_name,
            api_base=openai_api_base
        )
    except ImportError:
        print("警告: openai库未安装，无法创建OpenAI嵌入函数。")
        print("提示: 请运行 'pip install openai' 安装所需依赖。")
        return None
    except Exception as e:
        print(f"创建OpenAI嵌入函数时出错: {str(e)}")
        return None


def create_client_config(
    client_type: str = "ephemeral",
    data_dir: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[str] = None,
    custom_auth_credentials: Optional[str] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
    api_key: Optional[str] = None,
    ssl_enabled: bool = True,
    dotenv_path: str = ".chroma_env"
) -> Dict[str, Any]:
    """
    创建Chroma客户端配置。

    参数:
        client_type: 客户端类型，可选值: 'http', 'cloud', 'persistent', 'ephemeral'
        data_dir: 持久化客户端数据目录
        host: Chroma主机地址
        port: Chroma端口
        custom_auth_credentials: 自定义认证凭据
        tenant: Chroma租户
        database: Chroma数据库
        api_key: Chroma API密钥
        ssl_enabled: 是否启用SSL
        dotenv_path: .env文件路径

    返回:
        Dict[str, Any]: 包含客户端配置的字典
    """
    # 加载环境变量
    load_dotenv(dotenv_path=dotenv_path)

    return {
        "client_type": os.getenv("CHROMA_CLIENT_TYPE", client_type),
        "data_dir": os.getenv("CHROMA_DATA_DIR", data_dir),
        "host": os.getenv("CHROMA_HOST", host),
        "port": os.getenv("CHROMA_PORT", port),
        "custom_auth_credentials": os.getenv("CHROMA_CUSTOM_AUTH_CREDENTIALS", custom_auth_credentials),
        "tenant": os.getenv("CHROMA_TENANT", tenant),
        "database": os.getenv("CHROMA_DATABASE", database),
        "api_key": os.getenv("CHROMA_API_KEY", api_key),
        "ssl_enabled": ssl_enabled if os.getenv("CHROMA_SSL") is None else os.getenv("CHROMA_SSL").lower() in ["true", "yes", "1", "t", "y"]
    }


def _create_http_client(host: str, port: Optional[str], ssl_enabled: bool,
                        custom_auth_credentials: Optional[str] = None) -> chromadb.Client:
    """创建HTTP客户端连接（内部辅助函数）"""
    settings = Settings()
    if custom_auth_credentials:
        settings = Settings(
            chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
            chroma_client_auth_credentials=custom_auth_credentials
        )

    try:
        return chromadb.HttpClient(
            host=host,
            port=port if port else None,
            ssl=ssl_enabled,
            settings=settings
        )
    except ssl.SSLError as e:
        print(f"SSL连接失败: {str(e)}")
        raise
    except Exception as e:
        print(f"连接到HTTP客户端时出错: {str(e)}")
        raise


def _create_cloud_client(tenant: str, database: str, api_key: str) -> chromadb.Client:
    """创建云端客户端连接（内部辅助函数）"""
    try:
        return chromadb.HttpClient(
            host="api.trychroma.com",
            ssl=True,  # 云端始终使用SSL
            tenant=tenant,
            database=database,
            headers={'x-chroma-token': api_key}
        )
    except ssl.SSLError as e:
        print(f"SSL连接失败: {str(e)}")
        raise
    except Exception as e:
        print(f"连接到云客户端时出错: {str(e)}")
        raise


def _create_persistent_client(data_dir: str) -> chromadb.Client:
    """创建持久化客户端连接（内部辅助函数）"""
    return chromadb.PersistentClient(path=data_dir)


def _create_ephemeral_client() -> chromadb.Client:
    """创建临时内存客户端连接（内部辅助函数）"""
    return chromadb.EphemeralClient()


def get_chroma_client(config: Optional[Dict[str, Any]] = None) -> chromadb.Client:
    """
    获取或创建Chroma客户端实例。

    参数:
        config: 可选的配置字典，包含连接信息。如果为None，将使用默认配置。

    返回:
        chromadb.Client: Chroma客户端实例

    异常:
        ValueError: 当必要参数缺失时抛出
        其他异常: 连接过程中出现的各种异常
    """
    if config is None:
        config = create_client_config()

    client_factory_map = {
        'http': lambda: _create_http_client(
            config["host"],
            config["port"],
            config["ssl_enabled"],
            config["custom_auth_credentials"]
        ) if config["host"] else ValueError("使用HTTP客户端时，必须提供主机地址"),

        'cloud': lambda: _create_cloud_client(
            config["tenant"],
            config["database"],
            config["api_key"]
        ) if all([config["tenant"], config["database"], config["api_key"]]) else ValueError(
            "使用cloud客户端时，必须提供租户、数据库和API密钥信息"
        ),

        'persistent': lambda: _create_persistent_client(config["data_dir"]) if config["data_dir"] else ValueError(
            "使用持久化客户端时，必须提供数据目录"
        ),

        'ephemeral': _create_ephemeral_client
    }

    client_factory = client_factory_map.get(config["client_type"])
    if not client_factory:
        raise ValueError(f"不支持的客户端类型: {config['client_type']}")

    client_or_error = client_factory()
    if isinstance(client_or_error, ValueError):
        raise client_or_error

    return client_or_error
##### 集合操作函数 #####

def list_collections(client=None, limit: Optional[int] = None, 
                    offset: Optional[int] = None) -> List[str]:
    """
    列出Chroma数据库中的所有集合名称，支持分页。
    
    参数:
        client: Chroma客户端，如未提供则自动获取
        limit: 可选的，返回结果的最大数量
        offset: 可选的，在返回结果前跳过的集合数量
    
    返回:
        集合名称列表
    """
    client = client or get_chroma_client()
    return client.list_collections(limit=limit, offset=offset)

def create_collection(
    collection_name: str,
    hnsw_space: Optional[str] = None,
    hnsw_construction_ef: Optional[int] = None,
    hnsw_search_ef: Optional[int] = None,
    hnsw_M: Optional[int] = None,
    hnsw_num_threads: Optional[int] = None,
    hnsw_resize_factor: Optional[float] = None,
    hnsw_batch_size: Optional[int] = None,
    hnsw_sync_threshold: Optional[int] = None,
    client=None
) -> str:
    """
    创建新的Chroma集合，可配置HNSW参数。
    
    参数:
        collection_name: 要创建的集合名称
        hnsw_space: HNSW索引使用的距离函数。选项: 'l2', 'ip', 'cosine'
        hnsw_construction_ef: 构建HNSW图的动态候选列表大小
        hnsw_search_ef: 搜索HNSW图的动态候选列表大小
        hnsw_M: 为每个新元素创建的双向链接数量
        hnsw_num_threads: HNSW构建期间使用的线程数
        hnsw_resize_factor: 索引满时的调整因子
        hnsw_batch_size: 索引构建期间批处理的元素数量
        hnsw_sync_threshold: 将索引同步到磁盘前处理的元素数量
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        表示操作成功的消息字符串
    """
    client = client or get_chroma_client()
    
    # 在元数据中直接构建HNSW配置，仅包含非None值
    metadata = {
        k: v for k, v in {
            "hnsw:space": hnsw_space,
            "hnsw:construction_ef": hnsw_construction_ef,
            "hnsw:M": hnsw_M,
            "hnsw:search_ef": hnsw_search_ef,
            "hnsw:num_threads": hnsw_num_threads,
            "hnsw:resize_factor": hnsw_resize_factor,
            "hnsw:batch_size": hnsw_batch_size,
            "hnsw:sync_threshold": hnsw_sync_threshold
        }.items() if v is not None
    }
    
    client.create_collection(
        name=collection_name,
        metadata=metadata if metadata else None
    )
    
    config_msg = f" 使用HNSW配置: {metadata}" if metadata else ""
    return f"已成功创建集合 {collection_name}{config_msg}"

def get_or_create_collection(
    collection_name: str,
    metadata: Optional[Dict] = None,
    embedding_function: Optional[Callable] = None,
    client=None
) -> chromadb.Collection:
    """
    获取已有集合，如果不存在则创建新集合。
    
    参数:
        collection_name: 要获取或创建的集合名称
        metadata: 可选的集合元数据（仅在创建新集合时使用）
        embedding_function: 可选的嵌入函数
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        chromadb.Collection: Chroma集合对象
    """
    client = client or get_chroma_client()
    
    # 如果未提供嵌入函数，尝试创建一个
    if embedding_function is None:
        embedding_function = get_embedding_function()
    
    return client.get_or_create_collection(
        name=collection_name,
        metadata=metadata,
        embedding_function=embedding_function
    )

def peek_collection(
    collection_name: str,
    limit: int = 5,
    client=None
) -> Dict:
    """
    查看Chroma集合中的文档示例。
    
    参数:
        collection_name: 要查看的集合名称
        limit: 要查看的文档数量
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        包含样本文档的字典
    """
    client = client or get_chroma_client()
    collection = client.get_collection(collection_name)
    return collection.peek(limit=limit)

def get_collection_info(collection_name: str, client=None) -> Dict:
    """
    获取Chroma集合的信息。
    
    参数:
        collection_name: 要获取信息的集合名称
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        包含集合信息的字典
    """
    client = client or get_chroma_client()
    collection = client.get_collection(collection_name)
    
    # 获取集合数量
    count = collection.count()
    
    # 查看几个示例文档
    peek_results = collection.peek(limit=3)
    
    return {
        "name": collection_name,
        "count": count,
        "sample_documents": peek_results
    }
    
def get_collection_count(collection_name: str, client=None) -> int:
    """
    获取Chroma集合中的文档数量。
    
    参数:
        collection_name: 要计数的集合名称
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        集合中的文档数量
    """
    client = client or get_chroma_client()
    collection = client.get_collection(collection_name)
    return collection.count()

def modify_collection(
    collection_name: str,
    new_name: Optional[str] = None,
    new_metadata: Optional[Dict] = None,
    client=None
) -> str:
    """
    修改Chroma集合的名称或元数据。
    
    参数:
        collection_name: 要修改的集合名称
        new_name: 可选的集合新名称
        new_metadata: 可选的集合新元数据
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        表示操作成功的消息字符串
    """
    client = client or get_chroma_client()
    collection = client.get_collection(collection_name)
    
    if new_name:
        collection.modify(name=new_name)
    if new_metadata:
        collection.modify(metadata=new_metadata)
    
    modified_aspects = []
    if new_name:
        modified_aspects.append("名称")
    if new_metadata:
        modified_aspects.append("元数据")
    
    return f"已成功修改集合 {collection_name}: 更新了 {' 和 '.join(modified_aspects)}"

def delete_collection(collection_name: str, client=None) -> str:
    """
    删除Chroma集合。
    
    参数:
        collection_name: 要删除的集合名称
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        表示操作成功的消息字符串
    """
    client = client or get_chroma_client()
    client.delete_collection(collection_name)
    return f"已成功删除集合 {collection_name}"