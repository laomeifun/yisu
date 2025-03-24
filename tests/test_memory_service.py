"""
记忆服务模块测试。

该测试文件验证memory_service.py提供的功能，使用pytest框架。
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from memory.memory import Memory, create_memory
from memory.memory_service import (
    save_memory, search_memories, get_memory_by_id,
    delete_memory, count_memories
)


# 模拟数据和辅助函数
@pytest.fixture
def mock_memory():
    """创建测试用的记忆对象"""
    memory, _ = create_memory(
        content="这是一个测试记忆内容",
        tags=["测试", "记忆"],
        memory_type="test",
        timestamp=datetime.now(),
        metadata={"source": "unit_test"}
    )
    return memory


@pytest.fixture
def mock_store_success():
    """模拟成功存储记忆"""
    with patch('memory.memory_service._store_memory') as mock_store:
        mock_store.return_value = True
        yield mock_store


@pytest.fixture
def mock_retrieve_results():
    """模拟检索记忆结果"""
    mock_results = [
        {
            "id": "mem_123",
            "content": "测试记忆1",
            "content_hash": "hash1",
            "tags_json": '["测试", "记忆"]',
            "type": "test",
            "timestamp": datetime.now().timestamp(),
            "relevance": 0.9,
            "source": "unit_test"
        },
        {
            "id": "mem_456",
            "content": "测试记忆2",
            "content_hash": "hash2",
            "tags_json": '["测试"]',
            "type": "test",
            "timestamp": datetime.now().timestamp(),
            "relevance": 0.7,
            "source": "unit_test"
        }
    ]
    
    with patch('memory.memory_service._retrieve_raw_memory') as mock_retrieve:
        mock_retrieve.return_value = mock_results
        yield mock_retrieve


@pytest.fixture
def mock_chroma_client():
    """模拟Chroma客户端"""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    
    with patch('memory.memory_service.get_chroma_client') as mock_get_client:
        mock_get_client.return_value = mock_client
        yield mock_client, mock_collection


# 测试用例
def test_save_memory(mock_store_success):
    """测试保存记忆功能"""
    memory_id, error = save_memory(
        content="测试保存记忆的内容",
        tags=["测试", "保存"],
        memory_type="test",
        metadata={"source": "test_save"}
    )
    
    assert error is None
    assert memory_id is not None
    assert memory_id.startswith("mem_")
    assert mock_store_success.called


def test_save_memory_failure():
    """测试保存记忆失败的情况"""
    with patch('memory.memory_service._store_memory', return_value=False):
        memory_id, error = save_memory(
            content="测试失败的内容"
        )
        
        assert memory_id is None
        assert error is not None
        assert "失败" in error


def test_save_memory_validation_error():
    """测试记忆内容验证失败的情况"""
    memory_id, error = save_memory(content="")
    
    assert memory_id is None
    assert error is not None
    assert "内容不能为空" in error or "创建记忆对象失败" in error


def test_search_memories(mock_retrieve_results):
    """测试搜索记忆功能"""
    results = search_memories(
        query="测试记忆",
        limit=5,
        tags=["测试"],
        memory_type="test"
    )
    
    assert len(results) == 2
    assert results[0].memory.content == "测试记忆1"
    assert results[0].relevance_score == 0.9
    assert results[1].memory.content == "测试记忆2"
    assert results[1].relevance_score == 0.7
    
    # 验证过滤参数是否正确传递
    mock_retrieve_results.assert_called_once()
    args, kwargs = mock_retrieve_results.call_args
    assert kwargs["query_text"] == "测试记忆"
    assert kwargs["n_results"] == 5
    assert kwargs["metadata_filter"] is not None


def test_search_memories_empty_results():
    """测试搜索记忆无结果的情况"""
    with patch('memory.memory_service._retrieve_raw_memory', return_value=[]):
        results = search_memories(query="不存在的记忆")
        assert len(results) == 0


def test_get_memory_by_id(mock_chroma_client):
    """测试通过ID获取记忆"""
    mock_client, mock_collection = mock_chroma_client
    
    # 模拟返回结果
    mock_collection.get.return_value = {
        "ids": ["mem_123"],
        "documents": ["测试记忆内容"],
        "metadatas": [{"content_hash": "hash123", "type": "test"}]
    }
    
    memory = get_memory_by_id("mem_123")
    
    assert memory is not None
    assert memory.content == "测试记忆内容"
    assert memory.memory_type == "test"
    assert mock_collection.get.called
    mock_collection.get.assert_called_with(
        ids=["mem_123"],
        include=["documents", "metadatas"]
    )


def test_get_memory_by_id_not_found(mock_chroma_client):
    """测试通过ID获取不存在的记忆"""
    mock_client, mock_collection = mock_chroma_client
    
    # 模拟空结果
    mock_collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": []
    }
    
    memory = get_memory_by_id("not_exist")
    assert memory is None


def test_delete_memory(mock_chroma_client):
    """测试删除记忆"""
    mock_client, mock_collection = mock_chroma_client
    
    result = delete_memory("mem_123")
    assert result is True
    mock_collection.delete.assert_called_with(ids=["mem_123"])


def test_count_memories(mock_chroma_client):
    """测试计算记忆数量"""
    mock_client, mock_collection = mock_chroma_client
    
    # 模拟普通计数
    mock_collection.count.return_value = 42
    count = count_memories()
    assert count == 42
    
    # 模拟带过滤条件的计数
    mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}
    count = count_memories(
        tags=["测试"],
        date_from=datetime.now() - timedelta(days=1)
    )
    assert count == 3
    assert mock_collection.get.called


def test_count_memories_collection_not_exists(mock_chroma_client):
    """测试当集合不存在时计算记忆数量"""
    mock_client, mock_collection = mock_chroma_client
    
    # 模拟集合不存在
    mock_client.get_collection.side_effect = ValueError("Collection not found")
    count = count_memories()
    assert count == 0