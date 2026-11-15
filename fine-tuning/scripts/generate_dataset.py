# -*- coding: utf-8 -*-
"""
Enterprise Travel & Administrative Specialist Dataset Generator
Generates Alpaca-format fine-tuning dataset (train.json) for Qwen2.5.

Covers 4 core capabilities:
1. Multi-hop Reasoning: Resolve hometown city from employee profile, then calculate allowance.
2. Direct Policy Calculation: Deterministic Tier-1 (150/day) vs Tier-2 (100/day) rules.
3. Multi-turn Incremental Days: Implicit addition (orig + added days) and delta calculation.
4. Compliance & Boundary Guardrails: Rejection of non-terrestrial or illicit travel requests.
"""

import json
import random
from pathlib import Path

# Seed Employee Profiles (Deterministic HRMS Directory)
EMPLOYEES = [
    {"name": "张三", "dept": "智能研发部", "hometown": "深圳", "tier": "一线城市标准（150.0元/天包干）", "rate": 150.0},
    {"name": "李四", "dept": "行政财务部", "hometown": "成都", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
    {"name": "王五", "dept": "市场营销部", "hometown": "武汉", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
    {"name": "赵六", "dept": "基础架构部", "hometown": "广州", "tier": "一线城市标准（150.0元/天包干）", "rate": 150.0},
    {"name": "孙七", "dept": "人力资源部", "hometown": "杭州", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
    {"name": "周八", "dept": "云原生平台部", "hometown": "北京", "tier": "一线城市标准（150.0元/天包干）", "rate": 150.0},
    {"name": "吴九", "dept": "大数据中台部", "hometown": "西安", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
    {"name": "郑十", "dept": "前端体验部", "hometown": "上海", "tier": "一线城市标准（150.0元/天包干）", "rate": 150.0},
    {"name": "陈十一", "dept": "移动研发部", "hometown": "南京", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
    {"name": "刘十二", "dept": "质量保障部", "hometown": "长沙", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
    {"name": "钱十三", "dept": "信息安全部", "hometown": "重庆", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
    {"name": "孙十四", "dept": "运维SRE部", "hometown": "苏州", "tier": "二线及普通城市标准（100.0元/天包干）", "rate": 100.0},
]

TIER_ONE_CITIES = {"北京", "上海", "广州", "深圳"}
TIER_TWO_CITIES = {"成都", "杭州", "武汉", "西安", "南京", "长沙", "重庆", "苏州", "青岛", "天津", "厦门", "福州", "济南", "大连"}

COMMON_INSTRUCTION = "你是企业智能行政助手。根据员工出差诉求，自主结合员工档案（籍贯、部门）与差旅标准，输出规范的核算结果，去除一切客套废话。"

def get_city_tier_and_rate(city: str):
    if city in TIER_ONE_CITIES:
        return "一线城市标准（150.0元/天包干）", 150.0
    return "二线及普通城市标准（100.0元/天包干）", 100.0

def generate_hometown_multihop_samples(count=100):
    samples = []
    templates = [
        "帮{name}查一下他回老家出差{days}天能领多少出差补贴？",
        "请问{name}申请回原籍老家出差{days}天，差旅补贴怎么核算？",
        "{name}下周回老家探亲兼出差{days}天，能报销多少差旅公杂补贴？",
        "帮{name}核算一下回老家出差{days}天的差旅补贴总额。",
        "{name}要回自己老家出差{days}天，按公司政策能领多少补助？",
        "查询{name}回老家出差{days}天的差旅公杂伙食包干总金额。",
        "{name}下周回老家出差{days}天，系统里算一下补贴。",
    ]
    
    for _ in range(count):
        emp = random.choice(EMPLOYEES)
        days = random.randint(2, 10)
        tmpl = random.choice(templates)
        user_input = tmpl.format(name=emp["name"], days=days)
        total = emp["rate"] * days
        
        output = (
            f"【差旅补贴核算单】\n"
            f"- 员工姓名：{emp['name']}\n"
            f"- 所属部门：{emp['dept']}\n"
            f"- 出差目的地：{emp['hometown']}（员工籍贯地）\n"
            f"- 城市级别：{emp['tier']}\n"
            f"- 出差天数：{days}天\n"
            f"- 补贴总额：¥{total:.1f}元\n"
            f"- 状态：核算完毕，符合公司出差管理条例。"
        )
        samples.append({
            "instruction": COMMON_INSTRUCTION,
            "input": user_input,
            "output": output
        })
    return samples

def generate_direct_destination_samples(count=80):
    samples = []
    all_cities = list(TIER_ONE_CITIES) + list(TIER_TWO_CITIES)
    templates = [
        "{name}去{city}出差{days}天，补贴总额是多少？",
        "请帮{name}核算一下到{city}出差{days}天的差旅补贴。",
        "{name}计划下周前往{city}出差{days}天，能领多少出差公杂补贴？",
        "帮{name}算一下{city}出差{days}天的差旅包干补贴总额。",
        "{name}因公去{city}出差{days}天，核算下补贴金额。",
    ]
    
    for _ in range(count):
        emp = random.choice(EMPLOYEES)
        city = random.choice(all_cities)
        days = random.randint(1, 12)
        tier_label, rate = get_city_tier_and_rate(city)
        total = rate * days
        tmpl = random.choice(templates)
        user_input = tmpl.format(name=emp["name"], city=city, days=days)
        
        output = (
            f"【差旅补贴核算单】\n"
            f"- 员工姓名：{emp['name']}\n"
            f"- 所属部门：{emp['dept']}\n"
            f"- 出差目的地：{city}\n"
            f"- 城市级别：{tier_label}\n"
            f"- 出差天数：{days}天\n"
            f"- 补贴总额：¥{total:.1f}元\n"
            f"- 状态：核算完毕，符合公司出差管理条例。"
        )
        samples.append({
            "instruction": COMMON_INSTRUCTION,
            "input": user_input,
            "output": output
        })
    return samples

def generate_incremental_days_samples(count=40):
    samples = []
    all_cities = list(TIER_ONE_CITIES) + list(TIER_TWO_CITIES)
    templates = [
        "{name}在{city}原定出差{orig_days}天（{orig_total:.1f}元），临时因公多呆{add_days}天，总共多少补贴？",
        "{name}在{city}出差{orig_days}天（已核算¥{orig_total:.1f}元），项目延期{add_days}天，最新总补贴是多少？",
        "请更新{name}在{city}的差旅：原定{orig_days}天（{orig_total:.1f}元），追加{add_days}天，重新核算总额。",
    ]
    
    for _ in range(count):
        emp = random.choice(EMPLOYEES)
        city = random.choice(all_cities)
        orig_days = random.randint(2, 6)
        add_days = random.randint(1, 4)
        total_days = orig_days + add_days
        tier_label, rate = get_city_tier_and_rate(city)
        orig_total = rate * orig_days
        delta_total = rate * add_days
        new_total = rate * total_days
        
        tmpl = random.choice(templates)
        user_input = tmpl.format(
            name=emp["name"], city=city,
            orig_days=orig_days, orig_total=orig_total,
            add_days=add_days
        )
        
        output = (
            f"【差旅变更追加核算单】\n"
            f"- 员工姓名：{emp['name']}\n"
            f"- 出差目的地：{city}（{tier_label}）\n"
            f"- 原定行程：{orig_days}天（¥{orig_total:.1f}元）\n"
            f"- 追加天数：{add_days}天（累计 {total_days}天）\n"
            f"- 重新核算总额：¥{new_total:.1f}元（差额追加 ¥{delta_total:.1f}元）\n"
            f"- 状态：已按最新出差周期更新。"
        )
        samples.append({
            "instruction": "你是企业智能行政助手。结合前序对话上下文，处理员工追加出差天数的差额或总额计算，去除一切客套废话。",
            "input": user_input,
            "output": output
        })
    return samples

def generate_compliance_rejection_samples(count=30):
    samples = []
    absurd_destinations = [
        ("火星", "非地球已知行政区域"),
        ("月球", "地外天体"),
        ("空间站", "非民用行政区域"),
        ("百慕大群岛深水区", "极度危险非营商区域"),
        ("侏罗纪公园", "虚构地理区域"),
        ("亚特兰蒂斯", "虚构传说区域"),
        ("澳门新葡京赌场娱乐区", "违规涉赌娱乐场所，严禁列支差旅费用"),
        ("公海游轮娱乐会所", "违规非营商消费场所"),
    ]
    
    templates = [
        "我下周要去{dest}出差{days}天，算一下出差补贴是多少？",
        "帮{name}申请去{dest}出差{days}天，差旅补贴怎么核算？",
        "{name}下周因公去{dest}考察{days}天，请发放出差补助。",
        "我要报销去{dest}的差旅费用，一共呆了{days}天。",
    ]
    
    for _ in range(count):
        emp = random.choice(EMPLOYEES)
        dest, reason = random.choice(absurd_destinations)
        days = random.randint(2, 7)
        tmpl = random.choice(templates)
        user_input = tmpl.format(name=emp["name"], dest=dest, days=days)
        
        output = (
            f"【合规驳回提醒】\n"
            f"- 申请事项：前往「{dest}」差旅核算\n"
            f"- 驳回原因：{reason}\n"
            f"- 处理结果：该地点不属于公司差旅报销条例认可的境内外合法营商行政区划，行政系统无法核算补贴，申请已被合规拦截驳回。"
        )
        samples.append({
            "instruction": "你是企业智能行政助手。遇到非合法营商区域、虚构地点或违规消费时，严正指出合规边界，拒绝编造补贴数据。",
            "input": user_input,
            "output": output
        })
    return samples

def main():
    random.seed(42)  # For reproducible dataset generation
    
    all_samples = []
    all_samples.extend(generate_hometown_multihop_samples(100))
    all_samples.extend(generate_direct_destination_samples(80))
    all_samples.extend(generate_incremental_days_samples(40))
    all_samples.extend(generate_compliance_rejection_samples(30))
    
    random.shuffle(all_samples)
    
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Successfully generated {len(all_samples)} fine-tuning samples!")
    print(f"📁 Dataset written to: {output_file}")
    
    # Quick statistics
    print("\nDataset Category Breakdown:")
    print(" - Hometown Multi-hop Reasoning: 100 samples")
    print(" - Direct Destination Policy:     80 samples")
    print(" - Incremental Days Calculation:  40 samples")
    print(" - Compliance Boundary Rejection: 30 samples")
    print(f" - Total Samples:                {len(all_samples)} samples")

if __name__ == "__main__":
    main()
