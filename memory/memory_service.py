"""
记忆服务模块 - 提供统一的记忆操作接口

该模块整合了 memory.py 中的数据结构和函数与 dbtools/chroma_utils.py 中的存储功能，
提供简洁一致的记忆操作接口。采用函数式风格设计，无状态操作。

作者: laomei
日期: 2025-03-24
"""
from typing import List, Optional, Dict, Any, Tuple, Callable
import uuid
from datetime import datetime

from .memory import (
    Memory, MemoryQueryResult, 
    create_memory, create_query_result,
    convert_memory_to_dict, create_memory_from_dict,
    compute_content_hash
)
from dbtools.chroma_utils import (
    get_chroma_client, get_embedding_function,
    add_document_to_collection, query_collection
)

# 类型别名定义，提高代码可读性
MemoryID = str
MemoryList = List[Memory]
MemoryQueryResults = List[MemoryQueryResult]
EmbeddingFunction = Callable[[List[str]], List[List[float]]]

# 常量定义
MEMORY_COLLECTION = "memory"


def save_memory(
    content: str,
    tags: List[str] = None,
    memory_type: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    metadata: Dict[str, Any] = None,
    memory_id: Optional[str] = None,
    embedding_function: Optional[EmbeddingFunction] = None
) -> Tuple[Optional[MemoryID], Optional[str]]:
    """
    创建并保存记忆到存储中。
    
    参数:
        content: 记忆内容
        tags: 可选的标签列表
        memory_type: 可选的记忆类型
        timestamp: 可选的时间戳，默认为当前时间
        metadata: 可选的额外元数据
        memory_id: 可选的记忆ID，如果不提供则自动生成
        embedding_function: 可选的自定义嵌入函数
        
    返回:
        Tuple[Optional[str], Optional[str]]: (记忆ID, 错误信息)
        如果成功，错误信息为None；如果失败，记忆ID为None
    """
    # 步骤1: 创建Memory对象
    memory_obj, error = create_memory(
        content=content,
        tags=tags,
        memory_type=memory_type,
        timestamp=timestamp,
        metadata=metadata or {}
    )
    
    if error:
        return None, f"创建记忆对象失败: {error}"
        
    # 步骤2: 生成唯一ID（如果未提供）
    actual_id = memory_id or f"mem_{uuid.uuid4().hex}"
    
    # 步骤3: 转换为存储格式
    memory_dict = convert_memory_to_dict(memory_obj)
    
    # 步骤4: 存储记忆 - 使用通用函数
    success = add_document_to_collection(
        collection_name=MEMORY_COLLECTION,
        document=content,
        metadata=memory_dict,  # 包含了所有元数据
        doc_id=actual_id,
        embedding_function=embedding_function
    )
    
    if not success:
        return None, "存储记忆失败"
        
    return actual_id, None


def search_memories(
    query: str,
    limit: int = 5,
    metadata_filter: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    embedding_function: Optional[EmbeddingFunction] = None
) -> MemoryQueryResults:
    """
    按语义相似度搜索记忆。
    
    参数:
        query: 搜索查询文本
        limit: 返回结果数量上限，默认5
        metadata_filter: 可选的原始元数据过滤条件（Chroma格式）
        tags: 可选的要匹配的标签列表
        memory_type: 可选的记忆类型过滤
        date_from: 可选的起始日期
        date_to: 可选的结束日期
        embedding_function: 可选的自定义嵌入函数
        
    返回:
        List[MemoryQueryResult]: 记忆查询结果列表，按相关性排序
    """
    # 构建复合过滤条件
    combined_filter = _build_metadata_filter(
        base_filter=metadata_filter,
        tags=tags,
        memory_type=memory_type,
        date_from=date_from,
        date_to=date_to
    )
    
    # 获取原始记忆数据 - 使用通用函数
    raw_results = query_collection(
        collection_name=MEMORY_COLLECTION,
        query_text=query,
        n_results=limit,
        metadata_filter=combined_filter,
        include_params=["documents", "metadatas", "distances"],
        embedding_function=embedding_function
    )
    
    # 如果没有结果，返回空列表
    if not raw_results:
        return []
    
    # 转换为MemoryQueryResult对象
    return _convert_raw_results_to_query_results(raw_results)


def get_memory_by_id(memory_id: str) -> Optional[Memory]:
    """
    通过ID获取单个记忆。
    
    参数:
        memory_id: 记忆ID
        
    返回:
        Optional[Memory]: 如果找到则返回记忆对象，否则返回None
    """
    client = get_chroma_client()
    try:
        # 直接通过ID查询
        result = client.get_collection(MEMORY_COLLECTION).get(
            ids=[memory_id],
            include=["documents", "metadatas"]
        )
        
        # 检查是否有结果
        if not result["ids"]:
            return None
            
        # 合并文档内容和元数据
        i = 0  # 只有一个结果
        data = result["metadatas"][i].copy() if "metadatas" in result else {}
        
        # 添加内容
        if "documents" in result:
            data["content"] = result["documents"][i]
            
        # 转换为Memory对象
        memory_obj, _ = create_memory_from_dict(data)
        return memory_obj
        
    except Exception as e:
        print(f"获取记忆时出错: {str(e)}")
        return None


def batch_save_memories(memories: List[Memory]) -> Tuple[List[str], List[str]]:
    """
    批量保存多个记忆对象。
    
    参数:
        memories: Memory对象列表
        
    返回:
        Tuple[List[str], List[str]]: (成功的ID列表, 失败的ID列表)
    """
    success_ids = []
    failed_ids = []
    
    for memory in memories:
        memory_id = f"mem_{uuid.uuid4().hex}"
        memory_dict = convert_memory_to_dict(memory)
        
        # 使用通用函数
        success = add_document_to_collection(
            collection_name=MEMORY_COLLECTION,
            document=memory.content,
            metadata=memory_dict,
            doc_id=memory_id
        )
        
        if success:
            success_ids.append(memory_id)
        else:
            failed_ids.append(memory_id)
            
    return success_ids, failed_ids


def delete_memory(memory_id: str) -> bool:
    """
    从存储中删除指定记忆。
    
    参数:
        memory_id: 要删除的记忆ID
        
    返回:
        bool: 操作是否成功
    """
    try:
        client = get_chroma_client()
        collection = client.get_collection(MEMORY_COLLECTION)
        collection.delete(ids=[memory_id])
        return True
    except Exception as e:
        print(f"删除记忆时出错: {str(e)}")
        return False


def delete_memories_by_filter(
    metadata_filter: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None
) -> int:
    """
    根据条件批量删除记忆。
    
    参数:
        metadata_filter: 可选的原始元数据过滤条件
        tags: 可选的要匹配的标签列表
        memory_type: 可选的记忆类型过滤
        date_from: 可选的起始日期
        date_to: 可选的结束日期
        
    返回:
        int: 删除的记忆数量，如遇错误则返回-1
    """
    try:
        client = get_chroma_client()
        collection = client.get_collection(MEMORY_COLLECTION)
        
        # 构建过滤条件
        combined_filter = _build_metadata_filter(
            base_filter=metadata_filter,
            tags=tags,
            memory_type=memory_type,
            date_from=date_from,
            date_to=date_to
        )
        
        if not combined_filter:
            return 0  # 未提供过滤条件，安全起见不执行全部删除
            
        # 先获取匹配的记忆ID
        matching_items = collection.get(
            where=combined_filter, 
            include=[]  # 只需要ID
        )
        
        # 如果没有匹配项，直接返回0
        if not matching_items["ids"]:
            return 0
            
        # 删除匹配的记忆
        collection.delete(ids=matching_items["ids"])
        
        return len(matching_items["ids"])
    except Exception as e:
        print(f"批量删除记忆时出错: {str(e)}")
        return -1


def count_memories(
    metadata_filter: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None
) -> int:
    """
    计算匹配条件的记忆数量。
    
    参数:
        metadata_filter: 可选的原始元数据过滤条件
        tags: 可选的要匹配的标签列表
        memory_type: 可选的记忆类型过滤
        date_from: 可选的起始日期
        date_to: 可选的结束日期
        
    返回:
        int: 匹配的记忆数量，出错时返回-1
    """
    try:
        client = get_chroma_client()
        collection = client.get_collection(MEMORY_COLLECTION)
        
        # 构建过滤条件
        combined_filter = _build_metadata_filter(
            base_filter=metadata_filter,
            tags=tags,
            memory_type=memory_type,
            date_from=date_from,
            date_to=date_to
        )
        
        # 获取计数
        if combined_filter:
            # 使用get+过滤器来计算匹配项
            matching_items = collection.get(where=combined_filter, include=[])
            return len(matching_items["ids"])
        else:
            # 无过滤条件，返回总数
            return collection.count()
            
    except ValueError:
        # 集合不存在
        return 0
    except Exception as e:
        print(f"计算记忆数量时出错: {str(e)}")
        return -1


# 内部辅助函数

def _build_metadata_filter(
    base_filter: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    memory_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None
) -> Optional[Dict]:
    """
    构建复合元数据过滤条件。
    
    参数:
        base_filter: 基础过滤条件
        tags: 标签过滤
        memory_type: 类型过滤
        date_from: 开始日期
        date_to: 结束日期
        
    返回:
        Optional[Dict]: 组合后的过滤条件字典，如无条件则返回None
    """
    conditions = []
    
    # 添加基础过滤条件
    if base_filter:
        conditions.append(base_filter)
    
    # 添加标签过滤
    if tags and len(tags) > 0:
        # 使用'tags_json'字段，这里我们需要一个近似的文本匹配
        # 因为tags存储为JSON字符串，所以使用$contains操作符进行子字符串匹配
        tag_conditions = []
        for tag in tags:
            # 确保标签被引号包围，以防止匹配到标签的子字符串
            tag_conditions.append({"tags_json": {"$contains": f'"{tag}"'}})
            
        # 如果有多个标签，使用$and组合
        if len(tag_conditions) > 1:
            conditions.append({"$and": tag_conditions})
        else:
            conditions.append(tag_conditions[0])
    
    # 添加类型过滤
    if memory_type:
        conditions.append({"type": memory_type})
    
    # 添加日期范围过滤
    if date_from or date_to:
        time_condition = {}
        if date_from:
            time_condition["$gte"] = date_from.timestamp()
        if date_to:
            time_condition["$lte"] = date_to.timestamp()
        
        conditions.append({"timestamp": time_condition})
    
    # 组合所有条件
    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def _convert_raw_results_to_query_results(raw_results: List[Dict]) -> MemoryQueryResults:
    """
    将原始检索结果转换为MemoryQueryResult对象列表。
    
    参数:
        raw_results: 从query_collection返回的原始结果
        
    返回:
        List[MemoryQueryResult]: 记忆查询结果列表
    """
    query_results = []
    
    for result in raw_results:
        # 提取ID和相关性分数
        memory_id = result.pop("id", None)
        relevance = result.pop("relevance", 0.5)  # 默认中等相关性
        
        # 构建用于创建Memory对象的数据字典
        memory_data = {}
        
        # 内容是必需的
        if "content" in result:
            memory_data["content"] = result.pop("content")
        else:
            continue  # 跳过没有内容的结果
            
        # 处理content_hash
        if "content_hash" in result:
            memory_data["content_hash"] = result.pop("content_hash")
        else:
            # 如果没有提供hash，则动态计算
            memory_data["content_hash"] = compute_content_hash(memory_data["content"])
        
        # 复制剩余所有字段到memory_data
        memory_data.update(result)
        
        # 创建Memory对象
        memory_obj, error = create_memory_from_dict(memory_data)
        if error or not memory_obj:
            print(f"转换记忆时出错: {error}")
            continue
            
        # 创建查询结果对象
        query_result = create_query_result(
            memory=memory_obj,
            relevance_score=relevance,
            debug_info={"id": memory_id}
        )
        
        query_results.append(query_result)
        
    return query_results