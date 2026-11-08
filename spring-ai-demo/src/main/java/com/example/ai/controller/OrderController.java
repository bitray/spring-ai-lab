package com.example.ai.controller;

import com.example.ai.vo.UserOrderVo;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 阶段一 - Step 4 & 5：Prompt Engineering + Structured Output
 *
 * Step 4 核心认知：
 * - Prompt 的质量直接决定输出质量，同一个模型，Prompt 写得好坏差距巨大
 * - Few-shot（给示例）比 Zero-shot（不给示例）在格式化任务上准确率高 30%+
 * - System Prompt 定义"角色和规则"，User Prompt 定义"具体任务"
 *
 * Step 5 核心认知：
 * - Structured Output = 强制 LLM 按你定义的 JSON Schema 输出
 * - Spring AI 的 entity() 方法会自动：1) 生成 JSON Schema 注入 Prompt  2) 解析响应为目标类型
 * - 底层原理：把 Java 类转成 JSON Schema → 塞进 Prompt 告诉模型"你必须按这个格式输出"
 */
@RestController
@RequestMapping("/ai")
public class OrderController {

    private final ChatClient chatClient;

    public OrderController(ChatClient.Builder builder) {
        this.chatClient = builder.build();
    }

    /**
     * Step 4: Prompt Engineering 实践 - Few-shot 示例驱动
     *
     * 调用方式: GET /ai/order/analyze?text=我昨天买了两杯星巴克拿铁，花了76块，已经送到了
     *
     * 这里我们手动在 Prompt 中给了一个"示例"（Few-shot），
     * 让模型知道我们期望的输出格式和推理方式。
     */
    @GetMapping("/order/analyze")
    public UserOrderVo analyzeOrder(@RequestParam String text) {
        String systemPrompt = """
                你是一个订单信息提取专家。用户会给你一段自然语言描述，你需要从中提取订单信息。
                
                规则：
                1. 如果某个字段无法从文本中确定，使用合理的默认值
                2. orderId 如果未提及，生成一个 "ORD-" 开头的随机编号
                3. status 只能是: PENDING / SHIPPED / DELIVERED / CANCELLED 之一
                """;

        String userPrompt = """
                请从以下文本中提取订单信息：
                
                "{text}"
                
                示例参考：
                输入: "上周三在京东买了一台ThinkPad X1，12999元，还在路上"
                输出: orderId=ORD-20240101, productName=ThinkPad X1, quantity=1, totalPrice=12999.0, status=SHIPPED
                """;

        return chatClient.prompt()
                .system(systemPrompt)
                .user(u -> u.text(userPrompt).param("text", text))
                .call()
                .entity(UserOrderVo.class);  // 关键！自动反序列化为 Java 对象
    }

    /**
     * Step 5: Structured Output - 批量提取
     *
     * 调用方式: GET /ai/order/batch?text=张三买了一台iPhone 16 Pro花了9999已签收，李四买了两双Nike跑鞋共1598还在发货中
     *
     * 注意 .entity() 的泛型用法：直接返回 List<UserOrderVo>
     * Spring AI 会自动处理 JSON 数组的反序列化
     */
    @GetMapping("/order/batch")
    public List<UserOrderVo> batchAnalyze(@RequestParam String text) {
        return chatClient.prompt()
                .system("""
                        你是一个订单信息批量提取专家。
                        从用户描述中提取所有订单，返回订单列表。
                        规则：orderId 未提及则生成 ORD- 开头编号；
                        status 限定: PENDING / SHIPPED / DELIVERED / CANCELLED
                        """)
                .user("请提取以下文本中的所有订单信息：\n\n" + text)
                .call()
                .entity(new ParameterizedTypeReference<List<UserOrderVo>>() {
                });
    }
}
