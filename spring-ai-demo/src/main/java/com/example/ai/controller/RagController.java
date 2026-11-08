package com.example.ai.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.markdown.MarkdownDocumentReader;
import org.springframework.ai.reader.markdown.config.MarkdownDocumentReaderConfig;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 阶段二 - RAG 核心控制器
 *
 * 两个核心接口：
 * 1. /rag/ingest  — 文档摄入：读取 Markdown → 切片 → 向量化 → 存入 Qdrant
 * 2. /rag/ask     — RAG 问答：用户提问 → 向量检索 → 拼接上下文 → LLM 生成回答
 *
 * 数据流全景：
 * ┌─── 摄入阶段 ───────────────────────────────────────────────────┐
 * │ Markdown文件 → MarkdownDocumentReader → TokenTextSplitter(切片) │
 * │ → EmbeddingModel(向量化) → VectorStore(存入Qdrant)              │
 * └────────────────────────────────────────────────────────────────┘
 * ┌─── 查询阶段 ───────────────────────────────────────────────────┐
 * │ 用户问题 → EmbeddingModel(向量化) → VectorStore(相似度检索)      │
 * │ → 取回Top-K片段 → 拼接到Prompt → ChatModel生成回答              │
 * └────────────────────────────────────────────────────────────────┘
 */
@RestController
@RequestMapping("/rag")
public class RagController {

    private final ChatClient chatClient;
    private final VectorStore vectorStore;

    /**
     * 自定义中文 QA 模板（替换 QuestionAnswerAdvisor 默认的英文模板）
     *
     * 为什么要换？
     * 1. 默认模板是英文的，小模型理解不稳定
     * 2. 默认模板的"If the answer is not in the context...can't answer"让小模型倾向于拒答
     *    （问题里的"35天"在资料里没有字面出现，模型就认为"没有答案"）
     * 3. 新模板明确要求"把规则应用到用户的具体情况"，引导模型做推理而不是字面匹配
     *
     * 两个坑：
     * 1. 上下文占位符必须叫 {question_answer_context}（不是 {context}，Advisor 写死了这个 key）
     * 2. 模板文本里避免出现括号，StringTemplate 可能把括号当表达式解析
     */
    private static final PromptTemplate QA_TEMPLATE =
            new PromptTemplate("""
                    以下是参考资料：
                    ---------------------
                    {question_answer_context}
                    ---------------------
                    请根据参考资料回答用户问题，要求：
                    1. 如果资料中包含与问题相关的规则或条款，必须引用并应用到用户的具体情况中推理
                    2. 即使无法给出精确数字，例如缺少工资基数，也要说明适用的规则和计算方式
                    3. 只有资料中完全没有相关信息时，才告知用户无法回答
                    用户问题：{query}
                    """);

    public RagController(ChatClient.Builder builder, VectorStore vectorStore) {
        this.chatClient = builder
                .defaultSystem("你是一个企业知识库助手。只根据提供的参考资料回答问题，如果资料中没有相关信息，请明确告知用户。用中文回复。")
                // 日志 Advisor：把发给 LLM 的完整 Prompt 打到控制台，方便排查 RAG 拼接问题
                .defaultAdvisors(new SimpleLoggerAdvisor())
                .build();
        this.vectorStore = vectorStore;
    }

    /**
     * 文档摄入接口
     *
     * 调用方式: POST /rag/ingest
     *
     * 流程:
     * 1. 扫描 classpath:docs/ 下所有 .md 文件
     * 2. 用 MarkdownDocumentReader 解析每个文件
     * 3. 用 TokenTextSplitter 切片（默认约800 token/片，重叠400）
     * 4. 每个切片附带 metadata（来源文件名），用于后续过滤
     * 5. Spring AI 自动调用 EmbeddingModel 向量化，然后存入 Qdrant
     *
     * 返回: 摄入了多少文档、多少切片
     */
    @PostMapping("/ingest")
    public Map<String, Object> ingest() throws IOException {
        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources("classpath:docs/*.md");

        List<Document> allChunks = new ArrayList<>();

        for (Resource resource : resources) {
            // 1. 读取 Markdown 文件为 Document 对象
            MarkdownDocumentReader reader = new MarkdownDocumentReader(resource,
                    MarkdownDocumentReaderConfig.defaultConfig());
            List<Document> documents = reader.get();

            // 2. 为每个文档添加 metadata（来源文件），用于数据隔离和溯源
            String source = resource.getFilename();
            documents.forEach(doc -> doc.getMetadata().put("source", source));

            // 3. 切片：将长文档切成多个小片段
            // TokenTextSplitter 参数: 默认chunkSize=800, minChunkSizeChars=350, overlap
            TokenTextSplitter splitter = new TokenTextSplitter();
            List<Document> chunks = splitter.apply(documents);

            allChunks.addAll(chunks);
        }

        // 4. 存入向量数据库（Spring AI 自动调用 EmbeddingModel 向量化）
        vectorStore.add(allChunks);

        return Map.of(
                "message", "文档摄入完成",
                "files", resources.length,
                "chunks", allChunks.size()
        );
    }

    /**
     * RAG 问答接口（使用 QuestionAnswerAdvisor 简化版）
     *
     * 调用方式: GET /rag/ask?question=入职8年有几天年假？
     *
     * QuestionAnswerAdvisor 帮你做了：
     * 1. 将用户问题向量化
     * 2. 在 Qdrant 中检索最相似的 Top-K 个片段
     * 3. 将片段拼接到 Prompt 中作为"参考资料"
     * 4. 调用 LLM 基于参考资料生成回答
     *
     * 你只需要一行 .advisors(...) 就搞定了整个 RAG 链路
     */
    @GetMapping("/ask")
    public String ask(@RequestParam String question) {
        return chatClient.prompt()
                .advisors(QuestionAnswerAdvisor.builder(vectorStore)
                        .searchRequest(SearchRequest.builder()
                                .topK(5)                // 检索最相似的5个片段
                                .similarityThreshold(0.0) // 中文 Embedding 分数偏低，不能设太高，否则全被过滤
                                .build())
                        .promptTemplate(QA_TEMPLATE)    // 自定义中文模板，引导模型应用规则推理
                        .build())
                .user(question)
                .call()
                .content();
    }

    /**
     * RAG 问答接口（带来源过滤版）
     *
     * 调用方式: GET /rag/ask/filter?question=出差住宿标准是多少？&source=company-expense-policy.md
     *
     * 通过 metadata 的 source 字段过滤，只在指定文档中检索。
     * 这就是"数据隔离"的实现——类似 SQL 的 WHERE source = 'xxx'
     */
    @GetMapping("/ask/filter")
    public String askWithFilter(@RequestParam String question, @RequestParam String source) {
        return chatClient.prompt()
                .advisors(QuestionAnswerAdvisor.builder(vectorStore)
                        .searchRequest(SearchRequest.builder()
                                .topK(5)
                                .similarityThreshold(0.0)
                                .filterExpression("source == '" + source + "'")
                                .build())
                        .promptTemplate(QA_TEMPLATE)
                        .build())
                .user(question)
                .call()
                .content();
    }
}
