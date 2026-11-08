package com.example.ai.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 阶段二 - RAG 轻量评估控制器
 *
 * 背景：chunking 实验（I+ 节）暴露了单次回答的随机性（同问题一好一坏），
 * 没有量化指标，任何调优都是"掷骰子"。这个控制器就是那把"尺子"。
 *
 * 两个自动打分维度（都是机器可判，不依赖人读）：
 * - retrievalScore 召回分：debug 检索的 topK 结果里是否命中期望来源文件（任一命中 = 1.0）
 * - answerScore   答案分：回答文本中包含期望要素关键词的比例（如 3 个关键词命中 2 个 = 0.67）
 *
 * 调用方式:
 *   GET /rag/eval?topK=5&threshold=0.0&runs=1
 *   runs>1 时每个问题跑多轮取平均，用于对抗小模型回答的随机性
 *
 * 用例数据: classpath:eval-cases.json
 * 特殊用例: expectedSources 与 answerKeywords 均为空 = 知识库外问题，期望模型诚实拒答（不打分）
 */
@RestController
@RequestMapping("/rag")
public class RagEvalController {

    /** 与 RagController 保持完全一致的中文 QA 模板（评估必须和生产同一套 Prompt） */
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

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final List<EvalCase> cases;

    public record EvalCase(String id, String question, List<String> expectedSources, List<String> answerKeywords) {}

    public RagEvalController(ChatClient.Builder builder, VectorStore vectorStore) throws IOException {
        this.chatClient = builder.build();
        this.vectorStore = vectorStore;
        ObjectMapper mapper = new ObjectMapper()
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        this.cases = mapper.readValue(new ClassPathResource("eval-cases.json").getInputStream(),
                new TypeReference<>() {});
    }

    @GetMapping("/eval")
    public Map<String, Object> evaluate(
            @RequestParam(defaultValue = "5") int topK,
            @RequestParam(defaultValue = "0.0") double threshold,
            @RequestParam(defaultValue = "1") int runs) {

        List<Map<String, Object>> results = new ArrayList<>();
        double totalRetrieval = 0;
        double totalAnswer = 0;
        int scored = 0;

        for (EvalCase c : cases) {
            // ① 召回分：只看向量检索是否命中期望来源（不消耗 LLM）
            List<Document> hits = vectorStore.similaritySearch(SearchRequest.builder()
                    .query(c.question())
                    .topK(topK)
                    .similarityThreshold(threshold)
                    .build());
            List<String> hitSources = hits.stream()
                    .map(d -> String.valueOf(d.getMetadata().getOrDefault("source", "unknown")))
                    .distinct()
                    .toList();

            boolean outOfScope = c.expectedSources().isEmpty();
            double retrievalScore;
            if (outOfScope) {
                retrievalScore = -1; // 知识库外问题：无法定义"正确来源"，不打分
            } else {
                boolean hit = hitSources.stream().anyMatch(c.expectedSources()::contains);
                retrievalScore = hit ? 1.0 : 0.0;
            }

            // ② 答案分：跑 runs 轮 ask，统计期望关键词命中率的平均值
            double answerSum = 0;
            List<Map<String, Object>> runDetails = new ArrayList<>();
            for (int i = 0; i < runs; i++) {
                String answer = ask(c.question());
                double s = answerScore(c, answer);
                answerSum += s;
                if (runs > 1) {
                    runDetails.add(Map.of("run", i + 1, "answerScore", s));
                }
            }
            double avgAnswer = outOfScope ? -1 : answerSum / runs;

            Map<String, Object> row = new LinkedHashMap<>();
            row.put("id", c.id());
            row.put("question", c.question());
            row.put("outOfScope", outOfScope);
            row.put("retrievalScore", retrievalScore);
            row.put("hitSources", hitSources);
            row.put("answerScore", avgAnswer);
            if (!runDetails.isEmpty()) {
                row.put("runs", runDetails);
            }
            results.add(row);

            if (!outOfScope) {
                totalRetrieval += retrievalScore;
                totalAnswer += avgAnswer;
                scored++;
            }
        }

        Map<String, Object> summary = new LinkedHashMap<>();
        summary.put("totalCases", cases.size());
        summary.put("scoredCases", scored);
        summary.put("topK", topK);
        summary.put("threshold", threshold);
        summary.put("runs", runs);
        summary.put("avgRetrievalScore", scored == 0 ? -1 : totalRetrieval / scored);
        summary.put("avgAnswerScore", scored == 0 ? -1 : totalAnswer / scored);

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("summary", summary);
        response.put("results", results);
        response.put("hint", "avgRetrievalScore 衡量检索层（换 chunking/模型看它）；avgAnswerScore 衡量端到端（还受模板与模型推理影响）；outOfScope 用例观察模型是否诚实拒答");
        return response;
    }

    /** 与 RagController.ask 同一套 Advisor 配置，保证评估口径与生产一致 */
    private String ask(String question) {
        String content = chatClient.prompt()
                .advisors(QuestionAnswerAdvisor.builder(vectorStore)
                        .searchRequest(SearchRequest.builder()
                                .topK(5)
                                .similarityThreshold(0.0)
                                .build())
                        .promptTemplate(QA_TEMPLATE)
                        .build())
                .user(question)
                .call()
                .content();
        return content == null ? "" : content;
    }

    /** 答案分 = 命中的期望关键词数 / 期望关键词总数；无期望关键词（越界问题）返回 -1 表示不打分 */
    private double answerScore(EvalCase c, String answer) {
        if (c.answerKeywords().isEmpty()) {
            return -1;
        }
        long matched = c.answerKeywords().stream()
                .filter(kw -> answer.contains(kw))
                .count();
        return (double) matched / c.answerKeywords().size();
    }
}
