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

    public record EmployeeProfileResponse(
            String employeeName,
            String department,
            String hometownCity,
            String baseLocation,
            String jobTitle
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

        // 模拟企业生产级外部微服务/第三方气象接口故障（用于验证大模型容错与自愈能力）
        // 使用真实城市（如拉萨），避免大模型凭借预训练常识认为"火星不是地球城市"而直接拒绝调用工具
        if (city.contains("拉萨") || city.contains("火星") || city.contains("故障") || city.contains("超时")) {
            System.err.printf(">>> [Tool Exception Simulation] 模拟第三方气象接口异常: 城市=[%s]%n", city);
            throw new RuntimeException("拉萨第三方气象数据服务连接超时 [HTTP 504 Gateway Timeout]，当前无法获取该地区气象数据");
        }

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
     * Tool 2: Calculate business travel allowance based on destination city and duration.
     * Demonstrates an internal rule/finance calculation tool with DDD rule encapsulation.
     */
    @Tool(description = "根据出差目的地城市名称与出差天数，计算员工差旅伙食与公杂补贴总额。当用户咨询出差补贴标准、计算差旅补助金额时调用。")
    public TravelAllowanceResponse calculateTravelAllowance(
            @ToolParam(description = "出差目的地城市名称，例如北京、上海、广州、深圳、杭州等") String cityName,
            @ToolParam(description = "出差天数") int days
    ) {
        System.out.printf(">>> [Raw LLM Call] calculateTravelAllowance 收到大模型传递的原始实参: cityName = [%s], days = [%d]%n", cityName, days);

        int actualDays = Math.max(1, days);
        String city = cityName == null ? "未知城市" : cityName.trim();

        // 确定性业务规则下沉到 Java：由系统判定几线城市，不让小模型猜
        boolean isTierOne = city.contains("北京") || city.contains("上海") ||
                            city.contains("广州") || city.contains("深圳");

        double dailyRate = isTierOne ? 150.0 : 100.0;
        double total = dailyRate * actualDays;
        String tierLabel = isTierOne ? "一线城市标准(150元/天)" : "二线及普通城市标准(100元/天)";

        System.out.printf(">>> [Tool Execution] calculateTravelAllowance triggered: 目的地=%s, 判定级别=%s, 天数=%d天, 总补贴=%.1f元%n",
                city, tierLabel, actualDays, total);

        return new TravelAllowanceResponse(
                actualDays,
                tierLabel,
                dailyRate,
                total,
                "标准依据公司出差管理条例：一线城市(北上广深)按150元/天包干，二线城市及其他按100元/天包干。"
        );
    }

    /**
     * Tool 3: Query employee profile by name.
     * Demonstrates an enterprise HRMS / directory lookup capability.
     * Used in multi-hop ReAct chains (e.g. resolve employee's hometown before calculating travel allowance).
     *
     * @param employeeName Employee's name
     * @return Employee profile including department, hometown, base location, and job title
     */
    @Tool(description = "根据员工姓名查询员工基本档案信息，包括所属部门、籍贯或老家城市、常驻办公地与职务。当需要获取员工籍贯、老家或所在部门时调用。")
    public EmployeeProfileResponse queryEmployeeProfile(
            @ToolParam(description = "员工姓名，例如张三、李四、王五") String employeeName
    ) {
        System.out.printf(">>> [Raw LLM Call] queryEmployeeProfile 收到大模型传递的原始实参: employeeName = [%s]%n", employeeName);
        String name = employeeName == null ? "" : employeeName.trim();

        if (name.contains("张三")) {
            return new EmployeeProfileResponse("张三", "智能研发部", "深圳", "北京", "资深架构师");
        } else if (name.contains("李四")) {
            return new EmployeeProfileResponse("李四", "行政财务部", "成都", "上海", "财务主管");
        } else if (name.contains("王五")) {
            return new EmployeeProfileResponse("王五", "市场营销部", "武汉", "北京", "业务经理");
        } else {
            return new EmployeeProfileResponse(name, "综合部", "未知城市", "北京", "普通员工");
        }
    }
}
