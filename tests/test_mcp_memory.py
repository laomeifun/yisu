"""
测试 MCP 记忆工具功能
"""
import sys
import pytest
from unittest.mock import patch, MagicMock
import argparse
from datetime import datetime
import chromadb

# 导入要测试的模块
import main
from memory.memory import Memory, create_memory
from memory.memory_service import save_memory, search_memories


@pytest.fixture
def mock_args():
    """模拟命令行参数"""
    args = argparse.Namespace()
    args.client_type = 'ephemeral'
    args.data_dir = './.test_chromadb'
    args.host = None
    args.port = None
    args.embedding_provider = 'default'
    args.embedding_model = None
    args.openai_api_key = None
    args.openai_api_base = None
    args.transport = 'stdio'
    return args


@pytest.fixture
def mock_chroma_client():
    """模拟 ChromaDB 客户端"""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42
    
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    mock_client.get_or_create_collection.return_value = mock_collection
    
    return mock_client, mock_collection


@pytest.fixture
def mock_embedding_fn():
    """模拟嵌入函数"""
    def mock_fn(texts):
        return [[0.1] * 384 for _ in texts]
    return mock_fn


@pytest.fixture
def setup_mcp_environment(mock_args, mock_chroma_client, mock_embedding_fn):
    """设置模拟的 MCP 环境"""
    mock_client, mock_collection = mock_chroma_client
    
    # 保存原始全局变量
    orig_args = main.args
    orig_chroma_client = main.chroma_client
    orig_embedding_fn = main.embedding_fn
    orig_collection = main.collection
    
    # 设置测试全局变量
    main.args = mock_args
    main.chroma_client = mock_client
    main.embedding_fn = mock_embedding_fn
    main.collection = mock_collection
    
    yield
    
    # 恢复原始全局变量
    main.args = orig_args
    main.chroma_client = orig_chroma_client
    main.embedding_fn = orig_embedding_fn
    main.collection = orig_collection


def test_parse_arguments():
    """测试命令行参数解析功能"""
    with patch('sys.argv', ['main.py', 
                           '--client-type=persistent',
                           '--data-dir=/test/path',
                           '--embedding-provider=openai',
                           '--embedding-model=text-embedding-3-small',
                           '--openai-api-key=test-key',
                           '--transport=http',
                           '--port=8000']):
        args = main.parse_arguments()
        
        assert args.client_type == 'persistent'
        assert args.data_dir == '/test/path'
        assert args.embedding_provider == 'openai'
        assert args.embedding_model == 'text-embedding-3-small'
        assert args.openai_api_key == 'test-key'
        assert args.transport == 'http'
        assert args.port == 8000


def test_init_chromadb(mock_args):
    """测试初始化 ChromaDB 功能"""
    with patch('main.get_chroma_client') as mock_get_client, \
         patch('main.get_embedding_function') as mock_get_embedding:
        
        # 设置模拟返回值
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_get_client.return_value = mock_client
        mock_get_embedding.return_value = mock_embedding
        
        # 调用测试函数
        result = main.init_chromadb(mock_args)
        
        # 验证结果
        assert result is not None
        assert result[0] == mock_client
        assert result[1] == mock_embedding
        
        # 验证调用
        mock_get_client.assert_called_once()
        mock_get_embedding.assert_called_once_with(
            provider=mock_args.embedding_provider,
            model_name=mock_args.embedding_model,
            openai_api_key=mock_args.openai_api_key,
            openai_api_base=mock_args.openai_api_base
        )


def test_init_collection(mock_chroma_client):
    """测试初始化集合功能"""
    mock_client, mock_collection = mock_chroma_client
    mock_embedding_fn = lambda x: [[0.1] * 384 for _ in x]
    
    with patch('main.get_or_create_collection') as mock_get_or_create:
        mock_get_or_create.return_value = mock_collection
        
        result = main.init_collection(
            mock_client, 
            'test_collection', 
            mock_embedding_fn
        )
        
        assert result == mock_collection
        mock_get_or_create.assert_called_once_with(
            collection_name='test_collection',
            client=mock_client,
            embedding_function=mock_embedding_fn
        )


def test_add_memory_success(setup_mcp_environment):
    """测试添加记忆功能(成功场景)"""
    with patch('main.save_memory') as mock_save:
        # 模拟保存成功
        mock_save.return_value = ('mem_test_id', None)
        
        # 调用测试函数
        result = main.add_memory(
            content="测试记忆内容",
            tags=["测试"],
            memory_type="测试类型",
            metadata={"test": "value"}
        )
        
        # 验证结果
        assert "成功保存记忆" in result
        assert "mem_test_id" in result
        assert "测试记忆内容" in result
        
        # 验证调用参数
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert kwargs["content"] == "测试记忆内容"
        assert kwargs["tags"] == ["测试"]
        assert kwargs["memory_type"] == "测试类型"
        assert kwargs["metadata"] == {"test": "value"}


def test_add_memory_failure(setup_mcp_environment):
    """测试添加记忆功能(失败场景)"""
    with patch('main.save_memory') as mock_save:
        # 模拟保存失败
        mock_save.return_value = (None, "测试错误")
        
        # 调用测试函数
        result = main.add_memory(content="")
        
        # 验证结果
        assert "保存记忆失败" in result
        assert "测试错误" in result


def test_search_memory_success(setup_mcp_environment):
    """测试搜索记忆功能(成功场景)"""
    with patch('main.search_memories') as mock_search:
        # 创建模拟记忆结果
        memory1, _ = create_memory(
            content="测试记忆1",
            tags=["测试"],
            memory_type="测试类型",
            timestamp=datetime.now()
        )
        memory2, _ = create_memory(
            content="测试记忆2",
            tags=["测试", "重要"],
            memory_type="测试类型",
            timestamp=datetime.now()
        )
        
        # 创建模拟查询结果
        mock_results = [
            main.create_query_result(memory1, relevance_score=0.95),
            main.create_query_result(memory2, relevance_score=0.8)
        ]
        mock_search.return_value = mock_results
        
        # 调用测试函数
        result = main.search_memory("测试查询", limit=2, tags=["测试"])
        
        # 验证结果
        assert "结果 #1 (相关性: 0.95)" in result
        assert "结果 #2 (相关性: 0.80)" in result
        assert "测试记忆1" in result
        assert "测试记忆2" in result
        
        # 验证调用参数
        mock_search.assert_called_once_with(
            query="测试查询",
            limit=2,
            tags=["测试"],
            memory_type=None,
            embedding_function=main.embedding_fn
        )


def test_search_memory_empty(setup_mcp_environment):
    """测试搜索记忆功能(无结果场景)"""
    with patch('main.search_memories') as mock_search:
        # 模拟无结果
        mock_search.return_value = []
        
        # 调用测试函数
        result = main.search_memory("不存在的查询")
        
        # 验证结果
        assert "未找到相关记忆" in result


def test_get_memory_success(setup_mcp_environment):
    """测试获取记忆功能(成功场景)"""
    with patch('main.get_memory_by_id') as mock_get:
        # 创建模拟记忆
        memory, _ = create_memory(
            content="测试记忆内容",
            tags=["测试"],
            memory_type="测试类型",
            timestamp=datetime.now(),
            metadata={"test": "value"}
        )
        mock_get.return_value = memory
        
        # 调用测试函数
        result = main.get_memory("mem_test_id")
        
        # 验证结果
        assert "记忆 ID: mem_test_id" in result
        assert "内容: 测试记忆内容" in result
        assert "类型: 测试类型" in result
        assert "标签: 测试" in result
        
        # 验证调用参数
        mock_get.assert_called_once_with("mem_test_id")


def test_get_memory_not_found(setup_mcp_environment):
    """测试获取记忆功能(未找到场景)"""
    with patch('main.get_memory_by_id') as mock_get:
        # 模拟未找到
        mock_get.return_value = None
        
        # 调用测试函数
        result = main.get_memory("not_exist_id")
        
        # 验证结果
        assert "未找到 ID 为 not_exist_id 的记忆" in result


def test_remove_memory_success(setup_mcp_environment):
    """测试删除记忆功能(成功场景)"""
    with patch('main.delete_memory') as mock_delete:
        # 模拟删除成功
        mock_delete.return_value = True
        
        # 调用测试函数
        result = main.remove_memory("mem_test_id")
        
        # 验证结果
        assert "成功删除记忆 mem_test_id" in result
        
        # 验证调用参数
        mock_delete.assert_called_once_with("mem_test_id")


def test_remove_memory_failure(setup_mcp_environment):
    """测试删除记忆功能(失败场景)"""
    with patch('main.delete_memory') as mock_delete:
        # 模拟删除失败
        mock_delete.return_value = False
        
        # 调用测试函数
        result = main.remove_memory("mem_test_id")
        
        # 验证结果
        assert "删除记忆 mem_test_id 失败" in result


def test_list_memory_stats(setup_mcp_environment):
    """测试记忆统计功能"""
    with patch('main.count_memories') as mock_count:
        # 模拟总数
        mock_count.return_value = 42
        
        # 模拟集合的get方法
        main.collection.get.return_value = {
            "ids": list(range(42)),
            "metadatas": [
                {"type": "会议记录"} for _ in range(20)
            ] + [
                {"type": "学习笔记"} for _ in range(15)
            ] + [
                {"type": "任务记录"} for _ in range(7)
            ]
        }
        
        # 调用测试函数
        result = main.list_memory_stats()
        
        # 验证结果
        assert "记忆总数: 42" in result
        assert "会议记录: 20 条" in result
        assert "学习笔记: 15 条" in result
        assert "任务记录: 7 条" in result