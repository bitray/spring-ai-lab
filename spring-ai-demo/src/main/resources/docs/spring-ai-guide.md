# Spring AI 技术文档

## 什么是 Spring AI
Spring AI 是 Spring 官方推出的 AI 应用开发框架，目标是将 AI 能力以 Spring 的方式（自动配置、依赖注入、统一抽象）引入 Java 生态。它的核心理念是"可移植的 AI 应用"——切换模型提供商（如从 Ollama 换到 OpenAI）只需修改配置，无需改代码。

## 核心组件
- **ChatClient**：对话客户端，统一的链式调用入口。支持同步调用（call）和流式调用（stream）。
- **ChatModel**：底层模型抽象，由 Spring AI 自动配置。不同提供商（Ollama、OpenAI、Azure）有各自的实现。
- **EmbeddingModel**：向量化模型，将文本转为高维向量。用于 RAG 中的文档索引和查询匹配。
- **VectorStore**：向量数据库抽象，屏蔽 Qdrant、Milvus、PGVector 等底层差异。
- **Advisor**：拦截器模式，可在请求前后插入逻辑（如 RAG 的检索增强）。

## RAG 在 Spring AI 中的实现
RAG（Retrieval-Augmented Generation，检索增强生成）的核心流程：
1. 文档摄入：将文档切片（Chunking）→ 向量化（Embedding）→ 存入向量数据库（VectorStore）。
2. 查询增强：用户提问 → 向量化 → 在向量数据库中检索相似片段 → 将片段拼接到 Prompt 中。
3. 生成回答：LLM 基于检索到的上下文生成回答，而非凭空编造。

Spring AI 提供了 QuestionAnswerAdvisor 来简化 RAG 流程，只需一行代码即可为 ChatClient 添加检索增强能力。

## Structured Output
Spring AI 支持将 LLM 的输出直接反序列化为 Java 对象。通过 .entity() 方法，框架会自动将目标类转为 JSON Schema 并注入 Prompt，强制模型按指定格式输出。
