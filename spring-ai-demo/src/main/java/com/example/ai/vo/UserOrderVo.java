package com.example.ai.vo;

/**
 * 阶段一 - Step 5: Structured Output 的目标实体
 *
 * 我们要让 LLM 直接返回这个 Java 对象，而不是自由文本。
 * 这就是 Structured Output 的价值：LLM 的输出直接变成强类型对象，省去手动解析。
 */
public record UserOrderVo(
        String orderId,
        String productName,
        int quantity,
        double totalPrice,
        String status
) {
}
