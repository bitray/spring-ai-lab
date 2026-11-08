package com.example.ai.controller;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.markdown.MarkdownDocumentReader;
import org.springframework.ai.reader.markdown.config.MarkdownDocumentReaderConfig;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClient;

import java.io.IOException;
import java.util.*;

/**
 * 阶段二 - RAG 诊断与调优控制器
 *
 * 这个控制器的目的：让你"看见" RAG 内部发生了什么。
 * 生产环境中，这类诊断能力是调优 RAG 的基础设施。
 */
@RestController
@RequestMapping("/rag")
public class RagDebugController {

    private final VectorStore vectorStore;
    private final EmbeddingModel embeddingModel;
    private final RestClient qdrantHttpClient;
    private final String collectionName;

    public RagDebugController(VectorStore vectorStore,
                              EmbeddingModel embeddingModel,
                              @Value("${spring.ai.vectorstore.qdrant.host:localhost}") String qdrantHost,
                              @Value("${spring.ai.vectorstore.qdrant.collection-name:knowledge-base}") String collectionName) {
        this.vectorStore = vectorStore;
        this.embeddingModel = embeddingModel;
        this.collectionName = collectionName;
        // 用 Qdrant 的 HTTP REST API (6333) 来做 collection 级别的删除操作
        this.qdrantHttpClient = RestClient.create("http://" + qdrantHost + ":6333");
    }

    /**
     * 检索诊断接口 —— 只看"召回了什么"，不调 LLM
     *
     * 调用方式: GET /rag/debug?question=生病35天工资&topK=5&threshold=0.3
     *
     * 返回:
     * - 每个召回片段的相似度分数（score）
     * - 片段的实际文本内容（content）
     * - 片段的来源文件（metadata.source）
     *
     * 用途:
     * - 判断"检索有没有找到正确的文档片段"
     * - 对比不同 topK / threshold 的效果
     * - 如果 score 都很低 → 说明 Embedding 模型或 Chunking 有问题
     * - 如果正确片段 score 高但排在后面 → 说明需要 Re-ranking
     */
    @GetMapping("/debug")
    public Map<String, Object> debugRetrieval(
            @RequestParam String question,
            @RequestParam(defaultValue = "5") int topK,
            @RequestParam(defaultValue = "0.0") double threshold,
            @RequestParam(defaultValue = "false") boolean full) {

        // 直接调用 VectorStore 的相似度检索，不经过 LLM
        List<Document> results = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(question)
                        .topK(topK)
                        .similarityThreshold(threshold)
                        .build());

        // 组装诊断信息
        List<Map<String, Object>> hits = new ArrayList<>();
        for (int i = 0; i < results.size(); i++) {
            Document doc = results.get(i);
            Map<String, Object> hit = new LinkedHashMap<>();
            hit.put("rank", i + 1);
            // score 在 metadata 中，key 为 "distance" 或通过 getScore()
            hit.put("score", doc.getMetadata().getOrDefault("distance", "N/A"));
            hit.put("source", doc.getMetadata().getOrDefault("source", "unknown"));
            // full=true 时返回完整内容，否则只展示前200字
            String content = doc.getText();
            hit.put("content", full || content.length() <= 200 ? content : content.substring(0, 200) + "...");
            hit.put("content_length", content.length());
            hits.add(hit);
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("question", question);
        response.put("params", Map.of("topK", topK, "threshold", threshold));
        response.put("total_hits", results.size());
        response.put("hits", hits);
        response.put("diagnosis", results.isEmpty()
                ? "⚠️ 没有召回任何片段！可能原因：1)阈值太高 2)Embedding模型对中文支持差 3)文档未摄入"
                : "✅ 召回了 " + results.size() + " 个片段，检查排名靠前的内容是否包含答案");
        return response;
    }

    /**
     * 切片预览接口 —— Chunking 研究神器（免向量化、免入库、秒级返回）
     *
     * 调用方式: GET /rag/chunks/preview?chunkSize=300
     *
     * 作用：在不摄入的情况下，预览文档按指定 chunkSize 会被切成什么样。
     * 对比不同 chunkSize 时不用反复 clear/ingest，直接调这个接口看切块效果：
     *   /rag/chunks/preview?chunkSize=200
     *   /rag/chunks/preview?chunkSize=800
     *   /rag/chunks/preview?chunkSize=2000
     * 观察点：答案会不会被切成两半？无关内容会不会混进同一片？
     */
    @GetMapping("/chunks/preview")
    public Map<String, Object> previewChunks(
            @RequestParam(defaultValue = "800") int chunkSize,
            @RequestParam(defaultValue = "350") int minChunkSizeChars) throws IOException {

        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources("classpath:docs/*.md");

        List<Map<String, Object>> fileResults = new ArrayList<>();
        int totalChunks = 0;

        for (Resource resource : resources) {
            MarkdownDocumentReader reader = new MarkdownDocumentReader(resource,
                    MarkdownDocumentReaderConfig.defaultConfig());
            List<Document> documents = reader.get();

            TokenTextSplitter splitter = new TokenTextSplitter(
                    chunkSize, minChunkSizeChars, 5, 100, true);
            List<Document> chunks = splitter.apply(documents);
            totalChunks += chunks.size();

            List<Map<String, Object>> chunkList = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                String text = chunks.get(i).getText();
                Map<String, Object> chunk = new LinkedHashMap<>();
                chunk.put("index", i + 1);
                chunk.put("char_count", text.length());
                chunk.put("text", text);
                chunkList.add(chunk);
            }

            Map<String, Object> fileResult = new LinkedHashMap<>();
            fileResult.put("file", resource.getFilename());
            fileResult.put("sections_after_md_reader", documents.size());
            fileResult.put("chunks_after_token_splitter", chunks.size());
            fileResult.put("chunks", chunkList);
            fileResults.add(fileResult);
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("params", Map.of("chunkSize", chunkSize, "minChunkSizeChars", minChunkSizeChars));
        response.put("total_chunks", totalChunks);
        response.put("files", fileResults);
        response.put("hint", "对比不同 chunkSize 的结果：答案被切断→减小 chunkSize 或改结构切；噪声混→减小 chunkSize");
        return response;
    }

    /**
     * 已入库切片统计接口 —— 查看 Qdrant 中当前存了多少切片
     *
     * 调用方式: GET /rag/chunks/stats
     *
     * 作用：通过 Qdrant REST API 直接查 collection，统计各来源文件的切片数量。
     * 用途：确认重复摄入（同一 source 数量翻倍 = 摄入了多次）、验证 clear 是否干净。
     */
    @GetMapping("/chunks/stats")
    public Map<String, Object> chunkStats() {
        // 调用 Qdrant REST API 滚动读取 collection 中所有点
        Map<String, Object> scrollRequest = Map.of(
                "limit", 1000,
                "with_payload", true,
                "with_vector", false);

        @SuppressWarnings("unchecked")
        Map<String, Object> result = qdrantHttpClient.post()
                .uri("/collections/{name}/points/scroll", collectionName)
                .contentType(MediaType.APPLICATION_JSON)
                .body(scrollRequest)
                .retrieve()
                .body(Map.class);

        // 统计各 source 的切片数量
        Map<String, Integer> sourceCount = new LinkedHashMap<>();
        int total = 0;
        try {
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> points = (List<Map<String, Object>>)
                    ((Map<String, Object>) result.get("result")).get("points");
            for (Map<String, Object> point : points) {
                total++;
                @SuppressWarnings("unchecked")
                Map<String, Object> payload = (Map<String, Object>) point.get("payload");
                String source = payload != null && payload.get("source") != null
                        ? payload.get("source").toString() : "unknown";
                sourceCount.merge(source, 1, Integer::sum);
            }
        } catch (Exception e) {
            return Map.of("error", "collection 可能不存在（先 ingest）: " + e.getMessage());
        }

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("collection", collectionName);
        response.put("total_points", total);
        response.put("by_source", sourceCount);
        response.put("hint", "同一 source 数量超过预期 = 重复摄入了，先 POST /rag/clear 再摄入");
        return response;
    }

    /**
     * 重新摄入接口（可调参数版）—— 用于 Chunking 实验
     *
     * 调用方式: POST /rag/reingest?chunkSize=400&overlap=100
     *
     * 参数说明:
     * - chunkSize: 每个切片的最大 token 数。越小→片段越碎→检索越精准但上下文越少
     * - overlap: 相邻切片的重叠 token 数。防止关键信息被切断在两个片段之间
     *
     * 实验建议:
     * - 默认(800/400): 适合长文档、段落完整
     * - 小切片(300/100): 适合FAQ、条目式文档（如公司制度）
     * - 大切片(1500/500): 适合需要长上下文推理的场景
     *
     * ⚠️ 注意: 这会追加写入！多次调用会有重复数据。
     * 如需清空重来，先调 POST /rag/clear
     */
    @PostMapping("/reingest")
    public Map<String, Object> reingest(
            @RequestParam(defaultValue = "800") int chunkSize,
            @RequestParam(defaultValue = "400") int overlap,
            @RequestParam(defaultValue = "350") int minChunkSizeChars) throws IOException {

        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources("classpath:docs/*.md");

        List<Document> allChunks = new ArrayList<>();

        for (Resource resource : resources) {
            MarkdownDocumentReader reader = new MarkdownDocumentReader(resource,
                    MarkdownDocumentReaderConfig.defaultConfig());
            List<Document> documents = reader.get();

            String source = resource.getFilename();
            documents.forEach(doc -> doc.getMetadata().put("source", source));

            // 使用自定义参数的切片器
            // 参数: defaultChunkSize, minChunkSizeChars, minChunkLengthToEmbed, maxNumChunks, keepSeparator
            // ⚠️ minChunkSizeChars 必须与 /rag/chunks/preview 保持一致（默认都是350），否则切分结果对不上
            // ⚠️ overlap 参数无效：Spring AI 1.0.0 的 TokenTextSplitter 根本不支持重叠（源码已证实），保留仅为占位
            TokenTextSplitter splitter = new TokenTextSplitter(
                    chunkSize,            // 每片 token 预算（窗口大小，实际切点会被标点回拉）
                    minChunkSizeChars,    // 智能标点切分的启用门槛：块超过此字符数才回拉到标点
                    5,            // 最小切片长度（低于此不单独成片）
                    100,          // 最大切片数量
                    true          // 保留分隔符
            );
            List<Document> chunks = splitter.apply(documents);
            allChunks.addAll(chunks);
        }

        vectorStore.add(allChunks);

        return Map.of(
                "message", "重新摄入完成",
                "params", Map.of("chunkSize", chunkSize, "overlap(无效占位)", overlap, "minChunkSizeChars", minChunkSizeChars),
                "files", resources.length,
                "chunks", allChunks.size(),
                "hint", "现在去 /rag/debug 试试同样的问题，对比召回效果"
        );
    }

    /**
     * 清空向量数据库（删除 collection 并立即重建空库）
     *
     * 调用方式: POST /rag/clear
     *
     * 用途:
     * - 做 Chunking 对比实验前，清空旧数据
     * - 换 Embedding 模型后，必须清空旧向量再重新摄入
     *
     * 原理:
     * 1. 调用 Qdrant REST API 删除 collection
     * 2. 立即用与 Spring AI 一致的配置（Cosine + embedding 维度）重建空 collection
     *    —— 否则 initialize-schema 只在启动时生效，clear 后必须重启应用才能 ingest（历史踩坑）
     * 重建后可直接调 /rag/reingest，无需重启
     */
    @PostMapping("/clear")
    public Map<String, Object> clear() {
        // 1. 删除 collection（不存在时容忍 404，不阻断流程）
        String deleteResult;
        try {
            deleteResult = qdrantHttpClient.delete()
                    .uri("/collections/{name}", collectionName)
                    .retrieve()
                    .body(String.class);
        } catch (Exception e) {
            deleteResult = "collection 不存在，跳过删除: " + e.getMessage();
        }

        // 2. 立即重建空 collection，配置与 QdrantVectorStore 自动建库一致：Cosine 距离 + embedding 维度
        int dimensions = embeddingModel.dimensions();
        Map<String, Object> createBody = Map.of(
                "vectors", Map.of("size", dimensions, "distance", "Cosine"));
        String createResult = qdrantHttpClient.put()
                .uri("/collections/{name}", collectionName)
                .contentType(MediaType.APPLICATION_JSON)
                .body(createBody)
                .retrieve()
                .body(String.class);

        return Map.of(
                "message", "Collection '" + collectionName + "' 已清空并重建，无需重启",
                "embedding_dimensions", dimensions,
                "delete_response", deleteResult != null ? deleteResult : "ok",
                "create_response", createResult != null ? createResult : "ok",
                "next_step", "现在直接调用 POST /rag/reingest?chunkSize=xxx 重新摄入"
        );
    }
}
