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

from typing import Optional, Union, Dict, Any, List, Callable
import ssl
import os
import chromadb
from chromadb.api import Collection
from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from memory import Memory

# 定义常量
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


# 默认嵌入
# DEFAULT_EMBEDDING = embedding_functions.DefaultEmbeddingFunction()

def get_embedding_function(
    provider: str = "sentence_transformer",
    model_name: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openai_api_base: Optional[str] = None,
) -> Optional[EmbeddingFunction]:
    """获取嵌入函数"""
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


def _get_sentence_transformer_embedding_function(model_name: str=DEFAULT_EMBEDDING_MODEL) -> Union[EmbeddingFunction[Documents], None]:
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
    embedding_function: Optional[EmbeddingFunction] = None,
    client: Optional[chromadb.Client] = None
) -> Collection:
    """
    获取已有集合，如果不存在则创建新集合。
    
    参数:
        collection_name: 要获取或创建的集合名称
        metadata: 可选的集合元数据（仅在创建新集合时使用）
        embedding_function: 可选的嵌入函数
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        Collection: Chroma集合对象
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
    client: Optional[chromadb.Client] = None
) -> Dict:
    """
    查看Chroma集合中的文档示例。
    
    参数:
        collection_name: 要查看的集合名称
        limit: 要查看的文档数量
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        Dict: 包含样本文档的字典
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

##### 文档操作函数 #####    

def add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
    ids: Optional[List[str]] = None,
    client=None
) -> str:
    """
    向Chroma集合添加文档。
    
    参数:
        collection_name: 要添加文档的集合名称
        documents: 要添加的文本文档列表
        metadatas: 可选的每个文档的元数据字典列表
        ids: 可选的文档ID列表
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        表示操作成功的消息字符串
    """
    client = client or get_chroma_client()
    collection = client.get_or_create_collection(collection_name)
    
    # 如果未提供ID，则生成顺序ID
    if ids is None:
        ids = [str(i) for i in range(len(documents))]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return f"已成功向集合 {collection_name} 添加 {len(documents)} 个文档"

def query_documents(
    collection_name: str,
    query_texts: List[str],
    n_results: int = 5,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: Optional[List[str]] = None,
    client=None
) -> Dict:
    """
    使用高级过滤从Chroma集合查询文档。
    
    参数:
        collection_name: 要查询的集合名称
        query_texts: 要搜索的查询文本列表
        n_results: 每个查询返回的结果数量
        where: 可选的使用Chroma查询操作符的元数据过滤器
               示例:
               - 简单相等: {"metadata_field": "value"}
               - 比较: {"metadata_field": {"$gt": 5}}
               - 逻辑与: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
               - 逻辑或: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
        where_document: 可选的文档内容过滤器
        include: 可选的响应中包含什么的列表。可以包含以下任何内容:
                ["documents", "embeddings", "metadatas", "distances"]
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        包含查询结果的字典
    """
    client = client or get_chroma_client()
    collection = client.get_collection(collection_name)
    
    return collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
        where_document=where_document,
        include=include
    )

def get_documents(
    collection_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    client=None
) -> Dict:
    """
    从Chroma集合获取文档，可选过滤。
    
    参数:
        collection_name: 要获取文档的集合名称
        ids: 可选的要检索的文档ID列表
        where: 可选的使用Chroma查询操作符的元数据过滤器
        where_document: 可选的文档内容过滤器
        include: 可选的响应中包含什么的列表。可以包含以下任何内容:
                ["documents", "embeddings", "metadatas"]
        limit: 可选的返回文档的最大数量
        offset: 可选的在返回结果前跳过的文档数量
        client: Chroma客户端，如未提供则自动获取
    
    返回:
        包含匹配文档、其ID和请求包含的字典
    """
    client = client or get_chroma_client()
    collection = client.get_collection(collection_name)
    
    return collection.get(
        ids=ids,
        where=where,
        where_document=where_document,
        include=include,
        limit=limit,
        offset=offset
    )

##### 记忆操作专用函数 #####

def store_memory(
    content: str,
    metadata: Dict[str, Any],
    memory_id: str,
    client=None,
    embedding_function: Optional[Callable] = None
) -> bool:
    """
    将单个记忆存储到memory集合中。
    
    此函数是add_documents的简化版本，专为存储记忆设计。
    它自动创建memory集合（如果不存在），并处理单个记忆的存储。
    
    参数:
        content: 记忆的文本内容
        metadata: 记忆的元数据字典（应包含从Memory对象转换的数据）
        memory_id: 记忆的唯一标识符
        client: Chroma客户端，如未提供则自动获取
        embedding_function: 可选的嵌入函数，如未提供则尝试创建一个
        
    返回:
        bool: 操作是否成功
    """
    try:
        client = client or get_chroma_client()
        
        # 如果未提供嵌入函数，尝试创建一个
        if embedding_function is None:
            embedding_function = get_embedding_function()
            
        # 确保memory集合存在
        collection = get_or_create_collection(
            "memory", 
            client=client,
            embedding_function=embedding_function
        )
        
        # 预处理元数据，确保所有值都是基本类型
        processed_metadata = {}
        for key, value in metadata.items():
            # 如果值为None，则跳过这个键
            if value is None:
                continue
            # 将非基本类型转换为字符串
            if not isinstance(value, (str, int, float, bool)):
                processed_metadata[key] = str(value)
            else:
                processed_metadata[key] = value
        
        # 存储记忆
        collection.add(
            documents=[content],
            metadatas=[processed_metadata],
            ids=[memory_id]
        )
        return True
    except Exception as e:
        print(f"存储记忆时出错: {str(e)}")
        return False

def retrieve_memory(
    query_text: str,
    n_results: int = 5,
    metadata_filter: Optional[Dict] = None,
    include_content: bool = True,
    include_metadata: bool = True,
    include_distances: bool = True,
    client=None,
    embedding_function: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    从memory集合中检索与查询相关的记忆。
    
    此函数是query_documents的简化版本，专为检索记忆设计。
    它自动处理memory集合，并提供更简洁的参数接口。
    
    参数:
        query_text: 用于搜索的查询文本
        n_results: 要返回的最相关结果数量
        metadata_filter: 可选的元数据过滤条件，使用Chroma查询语法
        include_content: 是否在结果中包含记忆内容
        include_metadata: 是否在结果中包含记忆元数据
        include_distances: 是否在结果中包含相似度距离
        client: Chroma客户端，如未提供则自动获取
        embedding_function: 可选的嵌入函数，如未提供则尝试创建一个
        
    返回:
        Dict[str, Any]: 包含检索结果的字典，格式与query_documents类似但更加精简
        
    示例:
        >>> results = retrieve_memory("重要会议")
        >>> for i, content in enumerate(results["documents"][0]):
        ...     print(f"记忆 {i+1}: {content[:50]}... (相关性: {1-results['distances'][0][i]:.2f})")
    """
    try:
        client = client or get_chroma_client()
        
        # 如果未提供嵌入函数，尝试创建一个
        if embedding_function is None:
            embedding_function = get_embedding_function()
        
        # 确保集合存在
        try:
            collection = client.get_collection(
                "memory",
                embedding_function=embedding_function
            )
        except ValueError:
            # 如果集合不存在，返回空结果
            empty_result = {
                "ids": [[]], 
                "documents": [[]] if include_content else None,
                "metadatas": [[]] if include_metadata else None,
                "distances": [[]] if include_distances else None
            }
            return empty_result
            
        # 准备查询参数
        include_params = []
        if include_content:
            include_params.append("documents")
        if include_metadata:
            include_params.append("metadatas")
        if include_distances:
            include_params.append("distances")
            
        # 执行查询
        return collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=metadata_filter,
            include=include_params
        )
    except Exception as e:
        print(f"检索记忆时出错: {str(e)}")
        # 返回空结果
        empty_result = {
            "ids": [[]], 
            "documents": [[]] if include_content else None,
            "metadatas": [[]] if include_metadata else None,
            "distances": [[]] if include_distances else None
        }
        return empty_result