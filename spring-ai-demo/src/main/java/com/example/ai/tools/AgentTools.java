package com.example.ai.tools;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

/**
 * Enterprise Agent Tool definition for Spring AI 1.0.0.
 * In Spring AI 1.0.0, methods annotated with @Tool are automatically discovered
 * and exposed as capabilities for the LLM to call autonomously.
 */
@Component
public class AgentTools {

    public record WeatherResponse(
            String cityName,
            String weatherCondition,
            int temperatureCelsius,
            String dressingAdvice
    ) {}

    public record TravelAllowanceResponse(
            int days,
            String cityTier,
            double dailyRateRmb,
            double totalAllowanceRmb,
            String policyNote
    ) {}

    /**
     * Tool 1: Query weather information for a specific city.
     * In real enterprise scenarios, this calls a 3rd party REST API (e.g. QWeather / OpenWeather).
     */
    @Tool(description = "根据城市名称查询实时天气、当前温度及穿衣出行建议。当用户询问某个城市的天气、气温或出行穿衣时调用。")
    public WeatherResponse queryWeather(
            @ToolParam(description = "目标城市名称，例如北京、上海、广州、深圳") String cityName
    ) {
        System.out.printf(">>> [Raw LLM Call] queryWeather 收到大模型传递的原始实参: cityName = [%s]%n", cityName);
        String city = cityName == null ? "未知城市" : cityName.trim();

        if (city.contains("北京")) {
            return new WeatherResponse("北京", "晴朗微风", 6, "气温偏低，建议携带羽绒服或厚大衣");
        } else if (city.contains("上海")) {
            return new WeatherResponse("上海", "小雨绵绵", 14, "体感较凉并有降雨，出门请携带雨伞");
        } else if (city.contains("深圳") || city.contains("广州")) {
            return new WeatherResponse(city, "多云转晴", 24, "气候温暖舒适，穿单衣或薄外套即可");
        } else {
            return new WeatherResponse(city, "晴", 18, "天气良好，适宜出行");
        }
    }

    /**
     * Tool 2: Calculate business travel allowance based on destination tier and duration.
     * Demonstrates an internal rule/finance calculation tool.
     */
    @Tool(description = "计算员工国内差旅的每日出差伙食与公杂补贴总额。当用户咨询出差补贴标准、计算差旅补助金额时调用。")
    public TravelAllowanceResponse calculateTravelAllowance(
            @ToolParam(description = "城市级别，如'一线城市'或'二线及其他'") String cityTier,
            @ToolParam(description = "出差天数") int days
    ) {
        System.out.printf(">>> [Raw LLM Call] calculateTravelAllowance 收到大模型传递的原始实参: cityTier = [%s], days = [%d]%n", cityTier, days);
        int actualDays = Math.max(1, days);
        boolean isTierOne = cityTier != null &&
                (cityTier.contains("一线") || cityTier.contains("北上广深") ||
                 cityTier.contains("北京") || cityTier.contains("上海") ||
                 cityTier.contains("广州") || cityTier.contains("深圳"));

        double dailyRate = isTierOne ? 150.0 : 100.0;
        double total = dailyRate * actualDays;
        String tierLabel = isTierOne ? "一线城市标准(150元/天)" : "二线及普通城市标准(100元/天)";

        System.out.printf(">>> [Tool Execution] calculateTravelAllowance triggered: %s, days=%d, total=%.1f%n",
                tierLabel, actualDays, total);

        return new TravelAllowanceResponse(
                actualDays,
                tierLabel,
                dailyRate,
                total,
                "标准依据公司2026年出差管理条例：一线城市按150元/天包干，二线城市及其他按100元/天包干。"
        );
    }
}
