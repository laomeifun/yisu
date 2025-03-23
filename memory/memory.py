"""记忆相关的数据模型（函数式风格）。"""
from typing import List, Optional, Dict, Any, NamedTuple, Tuple  # 导入类型提示工具
from datetime import datetime  # 导入日期时间处理
import hashlib
import json
from functools import partial

# 常量定义
REQUIRED_FIELDS = ["content", "content_hash"]
METADATA_EXCLUDED_FIELDS = ["content", "content_hash", "tags_json", "type", "timestamp"]

# 定义记忆和查询结果的命名元组
class Memory(NamedTuple):
    """
    表示单个记忆条目。
    
    字段:
    - content: 记忆的主要内容
    - content_hash: 内容的唯一标识哈希值
    - tags: 标签列表，用于分类和检索
    - memory_type: 记忆类型
    - timestamp: 创建时间戳
    - metadata: 额外的元数据信息
    - embedding: 向量嵌入表示，用于语义搜索
    """
    content: str
    content_hash: str
    tags: List[str] = []
    memory_type: Optional[str] = None
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None


class MemoryQueryResult(NamedTuple):
    """
    表示记忆查询结果，包含相关性评分和调试信息。
    
    字段:
    - memory: 记忆对象
    - relevance_score: 相关性评分
    - debug_info: 调试相关的信息
    """
    memory: Memory
    relevance_score: float
    debug_info: Dict[str, Any] = {}


def compute_content_hash(content: str) -> str:
    """
    计算内容的哈希值。
    
    参数:
        content: 要计算哈希的内容字符串
        
    返回:
        str: 计算得到的哈希值
    """
    return hashlib.md5(content.encode()).hexdigest()


def validate_memory_data(content: str, content_hash: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """
    验证记忆数据是否有效。
    
    参数:
        content: 记忆内容
        content_hash: 可选的内容哈希值
        
    返回:
        Tuple[bool, Optional[str]]: (是否有效, 错误信息)
    """
    if not content or not content.strip():
        return False, "内容不能为空"
        
    if content_hash is not None and not content_hash:
        return False, "如果提供了内容哈希，则不能为空"
        
    return True, None


def create_memory(
    content: str,
    content_hash: Optional[str] = None,
    tags: List[str] = None,
    memory_type: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    metadata: Dict[str, Any] = None,
    embedding: Optional[List[float]] = None
) -> Tuple[Optional[Memory], Optional[str]]:
    """
    创建新的记忆对象。
    
    参数:
        content: 记忆的主要内容
        content_hash: 内容的唯一标识哈希值，如不提供则自动计算
        tags: 标签列表，用于分类和检索
        memory_type: 记忆类型
        timestamp: 创建时间戳
        metadata: 额外的元数据信息
        embedding: 向量嵌入表示，用于语义搜索
        
    返回:
        Tuple[Optional[Memory], Optional[str]]: (记忆对象, 错误信息)，如果成功则错误信息为None
    """
    # 验证数据
    is_valid, error_msg = validate_memory_data(content, content_hash)
    if not is_valid:
        return None, error_msg
    
    # 如果未提供哈希，自动计算
    actual_hash = content_hash or compute_content_hash(content)
    
    # 创建记忆对象
    return Memory(
        content=content,
        content_hash=actual_hash,
        tags=tags or [],
        memory_type=memory_type,
        timestamp=timestamp or datetime.now(),
        metadata=metadata or {},
        embedding=embedding
    ), None


def convert_memory_to_dict(memory: Memory) -> Dict[str, Any]:
    """
    将记忆对象转换为字典格式以便存储。
    
    处理流程:
    1. 转换基本字段
    2. 使用JSON序列化标签列表
    3. 转换时间戳为浮点数
    4. 合并元数据
    
    参数:
        memory: 要转换的记忆对象
        
    返回:
        Dict[str, Any]: 包含记忆数据的字典
    """
    return {
        "content": memory.content,  # 记忆内容
        "content_hash": memory.content_hash,  # 内容哈希
        "tags_json": json.dumps(memory.tags, ensure_ascii=False),  # 标签使用JSON序列化
        "type": memory.memory_type,  # 记忆类型
        "timestamp": memory.timestamp.timestamp(),  # 时间戳转换为浮点数
        **memory.metadata  # 展开其他元数据
    }


def create_memory_from_dict(data: Dict[str, Any], embedding: Optional[List[float]] = None) -> Tuple[Optional[Memory], Optional[str]]:
    """
    从字典数据创建记忆实例。
    
    处理流程:
    1. 验证必需字段
    2. 解析JSON格式的标签
    3. 转换时间戳
    4. 提取元数据
    5. 构建记忆对象
    
    参数:
        data: 包含记忆数据的字典
        embedding: 可选的向量嵌入
        
    返回:
        Tuple[Optional[Memory], Optional[str]]: (记忆对象, 错误信息)，如果成功则错误信息为None
    """
    # 验证必需字段
    for field in REQUIRED_FIELDS:
        if field not in data or not data[field]:
            return None, f"缺少必需字段: {field}"
    
    try:
        # 解析标签列表
        tags = []
        if "tags_json" in data:
            try:
                tags = json.loads(data["tags_json"])
            except json.JSONDecodeError:
                return None, "标签JSON格式无效"
        elif "tags_str" in data:  # 向后兼容旧格式
            tags_str = data.get("tags_str", "")
            tags = [tag for tag in tags_str.split(",") if tag] if tags_str else []
        
        # 转换时间戳，如果没有则使用当前时间
        try:
            timestamp = datetime.fromtimestamp(float(data["timestamp"])) if "timestamp" in data else datetime.now()
        except (ValueError, TypeError):
            return None, "时间戳格式无效"
        
        # 提取其他元数据(排除已处理的字段)
        metadata = {k: v for k, v in data.items() if k not in METADATA_EXCLUDED_FIELDS}
        
        return Memory(
            content=data["content"],
            content_hash=data["content_hash"],
            tags=tags,
            memory_type=data.get("type"),
            timestamp=timestamp,
            metadata=metadata,
            embedding=embedding
        ), None
    except Exception as e:
        return None, f"创建记忆时出错: {str(e)}"


def create_query_result(
    memory: Memory,
    relevance_score: float,
    debug_info: Dict[str, Any] = None
) -> MemoryQueryResult:
    """
    创建查询结果对象。
    
    参数:
        memory: 记忆对象
        relevance_score: 相关性评分
        debug_info: 调试相关的信息
        
    返回:
        MemoryQueryResult: 查询结果对象
    """
    return MemoryQueryResult(
        memory=memory,
        relevance_score=relevance_score,
        debug_info=debug_info or {}
    )


# 向后兼容的别名函数
memory_to_dict = convert_memory_to_dict
memory_from_dict = lambda data, embedding=None: create_memory_from_dict(data, embedding)[0]
