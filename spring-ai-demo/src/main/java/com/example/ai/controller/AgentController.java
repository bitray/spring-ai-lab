package com.example.ai.controller;

import com.example.ai.tools.AgentTools;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * Enterprise AI Agent Controller featuring:
 * 1. Autonomous Tool Calling (queryWeather, calculateTravelAllowance)
 * 2. Multi-turn State Management via ChatMemory and MessageChatMemoryAdvisor
 * 3. Session isolation via conversationId
 * 4. DDD rule encapsulation: deterministic logic stays in Java, intent extraction stays in LLM.
 */
@RestController
@RequestMapping("/ai/agent")
public class AgentController {

    private final ChatClient chatClient;
    private final AgentTools agentTools;
    private final ChatMemory chatMemory;

    public AgentController(ChatClient.Builder builder, AgentTools agentTools, ChatMemory chatMemory) {
        this.agentTools = agentTools;
        this.chatMemory = chatMemory;
        this.chatClient = builder
                .defaultSystem("""
                        你是一个专业、严谨的企业行政与出行助手。
                        你可以根据用户的需求，自主判断并使用手头的工具来查询信息或执行计算。
                        要求：
                        1. 如果用户的问题涉及天气或穿衣出行，调用对应天气工具查询；
                        2. 如果用户的问题涉及出差补贴金额，调用对应补贴计算工具；
                        3. 如果工具返回了数据，必须据实引用工具给出的数字和建议回答，不要捏造；
                        4. 如果用户只是进行日常问候或技术咨询，不需要调用任何工具，直接礼貌回答；
                        5. 结合历史会话记忆理解用户的追问（例如：用户追问'如果再多呆2天'，结合前文目的地和天数进行计算）。
                        """)
                .defaultAdvisors(
                        new SimpleLoggerAdvisor(),
                        MessageChatMemoryAdvisor.builder(chatMemory).build()
                )
                .build();
    }

    /**
     * Agent execution endpoint with multi-turn conversation memory.
     *
     * Example Multi-turn Test:
     * Round 1:
     * GET /ai/agent/ask?prompt=我下周要到北京出差4天，算一下出差补贴&conversationId=session-001
     * -> Result: Beijing Tier-1 (150/day * 4 = 600 RMB)
     *
     * Round 2 (Context Inheritance):
     * GET /ai/agent/ask?prompt=那如果临时多呆2天呢？总共多少补贴？&conversationId=session-001
     * -> Result: Automatically inherits Beijing, calculates 4 + 2 = 6 days (150/day * 6 = 900 RMB)!
     */
    @GetMapping("/ask")
    public String askAgent(
            @RequestParam String prompt,
            @RequestParam(defaultValue = "session-001") String conversationId
    ) {
        System.out.println("\n==================== [Agent Request Start] ====================");
        System.out.println("Session ID:  " + conversationId);
        System.out.println("User Prompt: " + prompt);

        String response = chatClient.prompt()
                .user(prompt)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, conversationId))
                .tools(agentTools)
                .call()
                .content();

        System.out.println("Agent Response: " + response);
        System.out.println("==================== [Agent Request End] ====================\n");
        return response;
    }

    /**
     * Clear memory for a specific conversation session.
     */
    @GetMapping("/clear")
    public String clearSession(@RequestParam(defaultValue = "session-001") String conversationId) {
        chatMemory.clear(conversationId);
        System.out.println(">>> [ChatMemory] Cleared session: " + conversationId);
        return "Session [" + conversationId + "] memory cleared successfully.";
    }
}
