# Yisu MCP 记忆功能使用指南

## 1. 概述

Yisu 的记忆功能基于 ChromaDB 向量数据库实现，支持记忆的存储、检索、管理等功能。通过 MCP (Model Context Protocol) 接口，可以方便地与各种 AI 模型集成。

## 2. 安装和依赖

确保安装了所有依赖：

```bash
pip install -r requirements.txt
```

如果使用 sentence-transformers 作为嵌入模型：

```bash
pip install sentence-transformers
```

如果使用 OpenAI 的嵌入功能：

```bash
pip install openai
```

## 3. 命令行参数

Yisu 支持通过命令行参数配置数据库连接和嵌入模型：

### 数据库配置

```bash
# 使用持久化存储
python main.py --client-type=persistent --data-dir=./data/chromadb

# 连接到远程 ChromaDB 服务器
python main.py --client-type=http --host=localhost --port=8000

# 使用临时内存存储（测试用）
python main.py --client-type=ephemeral
```

### 嵌入模型配置

```bash
# 使用 Sentence Transformers
python main.py --embedding-provider=sentence_transformer --embedding-model=all-MiniLM-L6-v2

# 使用 OpenAI 嵌入
python main.py --embedding-provider=openai --embedding-model=text-embedding-3-small --openai-api-key=YOUR_API_KEY

# 使用默认嵌入
python main.py --embedding-provider=default
```

### MCP 服务器配置

```bash
# 使用标准输入/输出（默认）
python main.py --transport=stdio

# 使用 HTTP 服务器
python main.py --transport=http --port=3000

# 使用 WebSocket 服务器
python main.py --transport=websocket --port=3000
```

## 4. 记忆功能 API

Yisu 提供以下 MCP 工具函数来操作记忆：

### 添加记忆

```json
{
  "type": "function",
  "name": "add_memory",
  "arguments": {
    "content": "这是一条重要会议记录，关于项目进度的讨论",
    "tags": ["会议", "项目进度"],
    "memory_type": "会议记录",
    "metadata": {"importance": "high", "project": "AI系统"}
  }
}
```

### 搜索记忆

```json
{
  "type": "function",
  "name": "search_memory",
  "arguments": {
    "query": "项目进度讨论",
    "limit": 5,
    "tags": ["会议"],
    "memory_type": "会议记录"
  }
}
```

### 获取记忆详情

```json
{
  "type": "function",
  "name": "get_memory",
  "arguments": {
    "memory_id": "mem_abc123"
  }
}
```

### 删除记忆

```json
{
  "type": "function",
  "name": "remove_memory",
  "arguments": {
    "memory_id": "mem_abc123"
  }
}
```

### 获取记忆统计信息

```json
{
  "type": "function",
  "name": "list_memory_stats",
  "arguments": {}
}
```

## 5. 实际使用示例

### 存储和检索情境知识

```json
// 存储会议记录
{
  "type": "function",
  "name": "add_memory",
  "arguments": {
    "content": "在2025年3月24日的会议中，团队讨论了项目A的进度。王工表示前端开发已完成80%，李工表示后端API已全部实现，但需要进一步测试。下周一将进行第一轮完整测试。",
    "tags": ["项目A", "进度会议", "开发"],
    "memory_type": "会议记录"
  }
}

// 稍后检索相关信息
{
  "type": "function",
  "name": "search_memory",
  "arguments": {
    "query": "项目A后端开发进度",
    "tags": ["项目A"]
  }
}
```

### 记忆知识库构建

```json
// 添加多条知识
{
  "type": "function",
  "name": "add_memory",
  "arguments": {
    "content": "Python 的字典推导式语法: {key: value for item in iterable}",
    "tags": ["Python", "语法", "字典"],
    "memory_type": "知识点"
  }
}

// 语义搜索知识
{
  "type": "function",
  "name": "search_memory",
  "arguments": {
    "query": "如何在Python中快速创建字典",
    "memory_type": "知识点"
  }
}
```

## 6. 最佳实践

1. **结构化标签**：
   - 使用一致的标签命名约定
   - 创建标签层次结构（如 "项目/前端", "项目/后端"）

2. **记忆类型**：
   - 为不同类型的记忆使用明确的类型名称
   - 常用类型：会议记录、知识点、总结、观察、决策

3. **内容格式**：
   - 尽量使用完整句子，提高语义搜索效果
   - 关键信息放在内容前面，增加检索命中率

4. **元数据使用**：
   - 用于存储结构化信息
   - 例如：重要性级别、相关人员、日期等

## 7. 故障排除

### 常见问题

1. **连接失败**：
   - 检查 ChromaDB 服务器是否在运行
   - 验证主机名和端口是否正确

2. **嵌入模型错误**：
   - 确认相关依赖已正确安装
   - 验证 OpenAI API 密钥是否有效

3. **搜索结果不相关**：
   - 尝试更换嵌入模型
   - 重构查询，使用更多相关关键词
   - 确保内容和查询使用相同的语言