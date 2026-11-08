package com.example.ai.controller;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

/**
 * 阶段一 - Step 2 & 3：基础对话 + 流式输出
 *
 * 核心概念：
 * - ChatClient 是 Spring AI 的统一入口，类似 RestTemplate 之于 HTTP
 * - 它屏蔽了底层模型差异（Ollama/OpenAI/...），换模型只需改配置
 */
@RestController
@RequestMapping("/ai")
public class ChatController {

    private final ChatClient chatClient;

    /**
     * Spring AI 自动注入 ChatClient.Builder，我们 build 出 ChatClient 实例。
     * 这里可以设置默认的 System Prompt（角色设定），所有请求都会携带。
     */
    public ChatController(ChatClient.Builder builder) {
        this.chatClient = builder
                .defaultSystem("你是一个专业的 Java 技术顾问，回答简洁明了，用中文回复。")
                .build();
    }

    /**
     * Step 2: 最基础的同步对话
     *
     * 调用方式: GET /ai/chat?message=什么是Spring Boot的自动配置？
     *
     * 流程: 用户输入 → ChatClient 组装 Prompt → 发给 Ollama → 等待完整响应 → 返回
     */
    @GetMapping("/chat")
    public String chat(@RequestParam String message) {
        return chatClient.prompt()
                .user(message)
                .call()
                .content();  // 直接拿到 String 响应
    }

    /**
     * Step 3: 流式输出（SSE - Server-Sent Events）
     *
     * 调用方式: GET /ai/chat/stream?message=解释一下Java虚拟线程
     *
     * 流程: 用户输入 → ChatClient → Ollama 逐 token 生成 → 逐块推送给前端 → 打字机效果
     *
     * 为什么需要流式？
     * - LLM 生成一段 500 字的回答可能需要 10-30 秒
     * - 同步等待：用户盯着白屏 30 秒
     * - 流式输出：用户 0.5 秒后就开始看到文字逐字出现，体验完全不同
     */
    @GetMapping(value = "/chat/stream", produces = "text/event-stream;charset=UTF-8")
    public Flux<String> chatStream(@RequestParam String message) {
        return chatClient.prompt()
                .user(message)
                .stream()       // 切换为流式模式
                .content();     // 返回 Flux<String>，每个元素是一小段文本
    }
}
