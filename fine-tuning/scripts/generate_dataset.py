# -*- coding: utf-8 -*-
"""
Enterprise Tool-Calling & ReAct Multi-Hop Dataset Generator for Qwen2.5.

Generates 300 OpenAI/Qwen-compatible tool-use training trajectories (`train.json`)
aligned 1:1 with `AgentTools.java` in `spring-ai-demo`.

Covers 5 capabilities without relying on verbose system-prompt rules:
1. Multi-hop hometown travel allowance (`queryEmployeeProfile` -> `calculateTravelAllowance`)
2. Direct city travel allowance & combined weather lookup
3. Multi-turn incremental days recalculation via deterministic tool invocation
4. Graceful handling of missing profile fields or external tool timeout errors
5. Compliance rejection for non-business / non-terrestrial travel requests
"""

import json
import random
from pathlib import Path

# Seed randomness for reproducible dataset generation across runs
random.seed(42)

# Exact OpenAI tool definitions matching Spring AI's serialized /api/chat payload
SPRING_AI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculateTravelAllowance",
            "description": "根据出差目的地城市名称与出差天数，计算员工差旅伙食与公杂补贴总额。当用户咨询出差补贴标准、计算差旅补助金额时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cityName": {
                        "type": "string",
                        "description": "出差目的地城市名称，例如北京、上海、广州、深圳、杭州等",
                    },
                    "days": {
                        "type": "integer",
                        "description": "出差天数",
                    },
                },
                "required": ["cityName", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queryEmployeeProfile",
            "description": "根据员工姓名查询员工基本档案信息，包括所属部门、籍贯或老家城市、常驻办公地与职务。当需要获取员工籍贯、老家或所在部门时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "employeeName": {
                        "type": "string",
                        "description": "员工姓名，例如张三、李四、王五",
                    }
                },
                "required": ["employeeName"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queryWeather",
            "description": "根据城市名称查询实时天气、当前温度及穿衣出行建议。当用户询问某个城市的天气、气温或出行穿衣时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "cityName": {
                        "type": "string",
                        "description": "目标城市名称，例如北京、上海、广州、深圳",
                    }
                },
                "required": ["cityName"],
            },
        },
    },
]

SYSTEM_PROMPTS = [
    "你是一个专业、严谨的企业行政与出行助手。",
    "你是企业智能行政助手。",
]

EMPLOYEE_NAMES = [
    "张三", "李四", "王五", "赵六", "孙七", "周八", "吴九", "郑十",
    "陈十一", "刘十二", "钱十三", "孙十四", "林峰", "苏敏", "杨洋", "黄伟",
    "周婷", "徐磊", "朱芳", "高明", "马超", "何静", "郭强", "罗兰",
]

DEPARTMENTS = [
    "智能研发部", "行政财务部", "市场营销部", "基础架构部",
    "人力资源部", "云原生平台部", "大数据中台部", "前端体验部",
    "移动研发部", "质量保障部", "信息安全部", "运维SRE部",
]

JOB_TITLES = [
    "资深架构师", "后端开发工程师", "财务主管", "业务经理",
    "高级产品经理", "测试专家", "运维工程师", "安全顾问",
]

TIER_ONE_CITIES = ["北京", "上海", "广州", "深圳"]
TIER_TWO_CITIES = [
    "成都", "杭州", "武汉", "西安", "南京", "长沙",
    "重庆", "苏州", "青岛", "天津", "厦门", "福州",
]
ALL_CITIES = TIER_ONE_CITIES + TIER_TWO_CITIES

WEATHER_BY_CITY = {
    "北京": ("晴朗微风", 6, "气温偏低，建议携带羽绒服或厚大衣"),
    "上海": ("小雨绵绵", 14, "体感较凉并有降雨，出门请携带雨伞"),
    "广州": ("多云转晴", 24, "气候温暖舒适，穿单衣或薄外套即可"),
    "深圳": ("多云转晴", 24, "气候温暖舒适，穿单衣或薄外套即可"),
}


def build_assistant_tool_call(tool_name: str, arguments: dict) -> dict:
    """Create an assistant message requesting a single tool invocation."""
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
        ],
    }


def build_tool_response(tool_name: str, payload: dict | str) -> dict:
    """Create a tool response message matching Ollama/Qwen2.5 chat format."""
    content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {
        "role": "tool",
        "name": tool_name,
        "content": content,
    }


def build_travel_allowance_payload(city: str, days: int) -> dict:
    """Mirror AgentTools.calculateTravelAllowance response fields verbatim."""
    is_tier_one = city in TIER_ONE_CITIES
    daily_rate = 150.0 if is_tier_one else 100.0
    tier_label = "一线城市标准(150元/天)" if is_tier_one else "二线及普通城市标准(100元/天)"
    return {
        "days": days,
        "cityTier": tier_label,
        "dailyRateRmb": daily_rate,
        "totalAllowanceRmb": daily_rate * days,
        "policyNote": "标准依据公司出差管理条例：一线城市(北上广深)按150元/天包干，二线城市及其他按100元/天包干。",
    }


def build_employee_profile_payload(name: str, hometown: str) -> dict:
    """Mirror AgentTools.queryEmployeeProfile response fields verbatim."""
    return {
        "employeeName": name,
        "department": random.choice(DEPARTMENTS),
        "hometownCity": hometown,
        "baseLocation": random.choice(TIER_ONE_CITIES),
        "jobTitle": random.choice(JOB_TITLES),
    }


def build_weather_payload(city: str) -> dict:
    """Mirror AgentTools.queryWeather response fields verbatim."""
    condition, temp, advice = WEATHER_BY_CITY.get(
        city, ("晴", 18, "天气良好，适宜出行")
    )
    return {
        "cityName": city,
        "weatherCondition": condition,
        "temperatureCelsius": temp,
        "dressingAdvice": advice,
    }


def generate_hometown_multihop_samples(count: int = 120) -> list[dict]:
    """
    Generate 2-hop ReAct trajectories where user asks for hometown allowance.
    Randomized employee->hometown pairings prevent weight memorization and force
    the model to call queryEmployeeProfile first, then calculateTravelAllowance.
    """
    prompts = [
        "帮{name}查一下他回老家出差{days}天能领多少出差补贴？",
        "请问{name}申请回原籍老家出差{days}天，差旅补贴怎么核算？",
        "{name}下周回老家探亲兼出差{days}天，能报销多少差旅公杂补贴？",
        "帮{name}核算一下回老家出差{days}天的差旅补贴总额。",
        "{name}要回自己老家出差{days}天，按公司政策能领多少补助？",
        "查询{name}回籍贯所在地出差{days}天的差旅伙食包干总金额。",
        "{name}下周回老家出差{days}天，算一下补贴。",
    ]

    samples = []
    for _ in range(count):
        name = random.choice(EMPLOYEE_NAMES)
        hometown = random.choice(ALL_CITIES)
        days = random.randint(1, 10)
        profile = build_employee_profile_payload(name, hometown)
        allowance = build_travel_allowance_payload(hometown, days)

        final_answer = (
            f"已为您完成核算：\n"
            f"- 员工：{name}（{profile['department']}，老家籍贯：{hometown}）\n"
            f"- 补贴标准：{allowance['cityTier']}\n"
            f"- 出差天数：{days}天\n"
            f"- 补贴总额：{allowance['totalAllowanceRmb']:.1f}元"
        )

        samples.append({
            "tools": SPRING_AI_TOOLS,
            "messages": [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": random.choice(prompts).format(name=name, days=days)},
                build_assistant_tool_call("queryEmployeeProfile", {"employeeName": name}),
                build_tool_response("queryEmployeeProfile", profile),
                build_assistant_tool_call("calculateTravelAllowance", {"cityName": hometown, "days": days}),
                build_tool_response("calculateTravelAllowance", allowance),
                {"role": "assistant", "content": final_answer},
            ],
        })
    return samples


def generate_direct_and_combined_samples(count: int = 70) -> list[dict]:
    """Generate single-hop allowance, weather, and 3-hop combined trajectories."""
    direct_prompts = [
        "{name}去{city}出差{days}天，补贴总额是多少？",
        "请帮{name}核算一下到{city}出差{days}天的差旅补贴。",
        "算一下去{city}出差{days}天的差旅包干补贴总额。",
        "{city}出差{days}天，按公司标准能领多少补助？",
    ]
    combined_prompts = [
        "帮{name}查一下他回老家出差{days}天的补贴，顺便看下他老家天气穿什么衣服？",
        "{name}要去老家出差{days}天，查下补贴金额和当地天气情况。",
    ]

    samples = []
    for index in range(count):
        name = random.choice(EMPLOYEE_NAMES)
        days = random.randint(1, 10)

        if index % 2 == 0:
            # Direct single-hop destination calculation
            city = random.choice(ALL_CITIES)
            allowance = build_travel_allowance_payload(city, days)
            final_answer = (
                f"{city}出差{days}天执行{allowance['cityTier']}，"
                f"补贴总额为 {allowance['totalAllowanceRmb']:.1f} 元。"
            )
            messages = [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": random.choice(direct_prompts).format(name=name, city=city, days=days)},
                build_assistant_tool_call("calculateTravelAllowance", {"cityName": city, "days": days}),
                build_tool_response("calculateTravelAllowance", allowance),
                {"role": "assistant", "content": final_answer},
            ]
        else:
            # Combined hometown profile -> allowance + weather
            hometown = random.choice(ALL_CITIES)
            profile = build_employee_profile_payload(name, hometown)
            allowance = build_travel_allowance_payload(hometown, days)
            weather = build_weather_payload(hometown)
            final_answer = (
                f"{name}（{profile['department']}）老家为{hometown}：\n"
                f"1. 差旅补贴：{days}天按{allowance['cityTier']}，共 {allowance['totalAllowanceRmb']:.1f} 元。\n"
                f"2. 当地天气：{weather['weatherCondition']}，{weather['temperatureCelsius']}°C，{weather['dressingAdvice']}。"
            )
            messages = [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": random.choice(combined_prompts).format(name=name, days=days)},
                build_assistant_tool_call("queryEmployeeProfile", {"employeeName": name}),
                build_tool_response("queryEmployeeProfile", profile),
                build_assistant_tool_call("calculateTravelAllowance", {"cityName": hometown, "days": days}),
                build_tool_response("calculateTravelAllowance", allowance),
                build_assistant_tool_call("queryWeather", {"cityName": hometown}),
                build_tool_response("queryWeather", weather),
                {"role": "assistant", "content": final_answer},
            ]

        samples.append({"tools": SPRING_AI_TOOLS, "messages": messages})
    return samples


def generate_incremental_days_samples(count: int = 40) -> list[dict]:
    """Train the model to invoke calculateTravelAllowance again when days are added."""
    samples = []
    followup_templates = [
        "项目临时延期，需要再多呆{add_days}天，总共能领多少补贴？",
        "行程有变，追加出差{add_days}天，帮我按最新总天数重算一下补贴。",
    ]

    for _ in range(count):
        city = random.choice(ALL_CITIES)
        orig_days = random.randint(2, 6)
        add_days = random.randint(1, 4)
        total_days = orig_days + add_days

        orig_allowance = build_travel_allowance_payload(city, orig_days)
        new_allowance = build_travel_allowance_payload(city, total_days)
        delta_rmb = new_allowance["totalAllowanceRmb"] - orig_allowance["totalAllowanceRmb"]

        messages = [
            {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
            {"role": "user", "content": f"去{city}出差{orig_days}天能领多少补贴？"},
            build_assistant_tool_call("calculateTravelAllowance", {"cityName": city, "days": orig_days}),
            build_tool_response("calculateTravelAllowance", orig_allowance),
            {
                "role": "assistant",
                "content": f"{city}出差{orig_days}天按{orig_allowance['cityTier']}，补贴总额为 {orig_allowance['totalAllowanceRmb']:.1f} 元。",
            },
            {"role": "user", "content": random.choice(followup_templates).format(add_days=add_days)},
            build_assistant_tool_call("calculateTravelAllowance", {"cityName": city, "days": total_days}),
            build_tool_response("calculateTravelAllowance", new_allowance),
            {
                "role": "assistant",
                "content": (
                    f"行程变更后累计出差 {total_days} 天（原 {orig_days} 天 + 追加 {add_days} 天），"
                    f"按{new_allowance['cityTier']}最新补贴总额为 {new_allowance['totalAllowanceRmb']:.1f} 元"
                    f"（较原金额增加 {delta_rmb:.1f} 元）。"
                ),
            },
        ]
        samples.append({"tools": SPRING_AI_TOOLS, "messages": messages})
    return samples


def generate_tool_failure_guardrail_samples(count: int = 35) -> list[dict]:
    """Train honest degradation when employee hometown is unknown or external API fails."""
    samples = []
    for index in range(count):
        name = random.choice(EMPLOYEE_NAMES)
        days = random.randint(2, 5)
        if index % 2 == 0:
            # Missing hometown in HRMS directory -> must NOT guess a random city
            unknown_profile = build_employee_profile_payload(name, "未知城市")
            messages = [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": f"帮{name}查一下他回老家出差{days}天的补贴是多少？"},
                build_assistant_tool_call("queryEmployeeProfile", {"employeeName": name}),
                build_tool_response("queryEmployeeProfile", unknown_profile),
                {
                    "role": "assistant",
                    "content": f"经查询员工档案，{name}的籍贯/老家城市登记为“未知城市”，无法确定具体出差目的地标准。请补充确认目标城市后再为您核算补贴。",
                },
            ]
        else:
            # External weather API timeout -> report error honestly without fabricating temperature
            messages = [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": "明天去拉萨出差，帮我查一下拉萨实时天气和穿衣建议。"},
                build_assistant_tool_call("queryWeather", {"cityName": "拉萨"}),
                build_tool_response(
                    "queryWeather",
                    "Error: 拉萨第三方气象数据服务连接超时 [HTTP 504 Gateway Timeout]，当前无法获取该地区气象数据",
                ),
                {
                    "role": "assistant",
                    "content": "抱歉，拉萨第三方气象数据服务当前连接超时（HTTP 504），暂时无法获取实时天气与穿衣建议，请稍后再试或查阅官方气象预报。",
                },
            ]
        samples.append({"tools": SPRING_AI_TOOLS, "messages": messages})
    return samples


def generate_compliance_rejection_samples(count: int = 35) -> list[dict]:
    """Train direct refusal for out-of-policy / non-terrestrial requests without tool abuse."""
    scenarios = [
        (
            "帮我算一下去火星出差10天的补贴能领多少钱？",
            "无法受理该请求。根据公司出差管理条例，差旅补贴仅适用于境内/境外合法地面因公出差，不支持地外天体或虚构目的地。",
        ),
        (
            "我去上海出差顺便带家人旅游5天，家人的差旅补贴能一起算进去吗？",
            "不符合报销政策。公司差旅伙食与公杂补贴仅针对员工本人因公出差行程核算，随行家属或私人旅游天数不计入补贴。",
        ),
        (
            "帮我把二线城市的出差补贴私下改按一线城市150元/天计算行吗？",
            "抱歉，无法修改补贴标准。所有城市级别与日包干金额均严格执行公司出差管理条例（一线北上广深150元/天，其他100元/天），无法人为调整。",
        ),
    ]

    samples = []
    for index in range(count):
        user_q, rejection = scenarios[index % len(scenarios)]
        samples.append({
            "tools": SPRING_AI_TOOLS,
            "messages": [
                {"role": "system", "content": random.choice(SYSTEM_PROMPTS)},
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": rejection},
            ],
        })
    return samples


def main() -> None:
    dataset = (
        generate_hometown_multihop_samples(120)
        + generate_direct_and_combined_samples(70)
        + generate_incremental_days_samples(40)
        + generate_tool_failure_guardrail_samples(35)
        + generate_compliance_rejection_samples(35)
    )
    random.shuffle(dataset)

    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully generated {len(dataset)} Tool-Calling samples -> {output_file}")


if __name__ == "__main__":
    main()
