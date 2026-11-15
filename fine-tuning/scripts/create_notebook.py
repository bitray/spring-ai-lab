# -*- coding: utf-8 -*-
"""
Generates the ready-to-run Jupyter Notebook for Kaggle / Colab fine-tuning.
"""
import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 Qwen2.5 企业差旅特种兵模型：QLoRA 极速微调与 GGUF 导出\n",
    "\n",
    "**目标**：基于 `Qwen/Qwen2.5-3B-Instruct`（或 1.5B），使用 QLoRA（4-bit 量化）微调出专属的企业行政模型。\n",
    "**硬件建议**：Kaggle / Colab 免费 T4 GPU（16GB 显存，完全免费，耗时约 5~8 分钟）。\n",
    "**产出交付**：一键导出 `Q4_K_M` 量化 GGUF 文件，直接导入本地 Ollama 供 Java Spring AI 调用！"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 1：安装微调极速利器 Unsloth 与相关依赖\n",
    "> Unsloth 相比原生 HuggingFace 显存减少 70%，训练速度提升 2~3 倍，并支持原生一键导出 GGUF。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%%capture\n",
    "!pip install --no-deps \"xformers<0.0.29\" trl peft accelerate bitsandbytes\n",
    "!pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 2：加载 Qwen2.5 底座模型（4-bit QLoRA 预量化）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from unsloth import FastLanguageModel\n",
    "import torch\n",
    "\n",
    "max_seq_length = 2048\n",
    "dtype = None # 自动检测（T4 支持 float16，A100 支持 bfloat16）\n",
    "load_in_4bit = True # 开启 4-bit 量化，极省显存\n",
    "\n",
    "# 推荐使用 unsloth 官方维护的 4-bit 高速 Qwen2.5 镜像\n",
    "model_name = \"unsloth/Qwen2.5-3B-Instruct\" # 若追求极速也可选 \"unsloth/Qwen2.5-1.5B-Instruct\"\n",
    "\n",
    "model, tokenizer = FastLanguageModel.from_pretrained(\n",
    "    model_name = model_name,\n",
    "    max_seq_length = max_seq_length,\n",
    "    dtype = dtype,\n",
    "    load_in_4bit = load_in_4bit,\n",
    ")\n",
    "print(\"✅ Base Model Loaded Successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 3：注入 LoRA 适配层（低秩旁路矩阵）\n",
    "> 仅训练 1%~2% 的 LoRA 参数，绝大部分底座权重冻结，防止灾难性遗忘。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = FastLanguageModel.get_peft_model(\n",
    "    model,\n",
    "    r = 16, # LoRA 秩（Rank），推荐 16 或 32\n",
    "    target_modules = [\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",\n",
    "                      \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
    "    lora_alpha = 16,\n",
    "    lora_dropout = 0, # 设为 0 时 Unsloth 会自动采用极致加速内核\n",
    "    bias = \"none\",\n",
    "    use_gradient_checkpointing = \"unsloth\", # 节省 30% 显存\n",
    "    random_state = 3407,\n",
    ")\n",
    "print(\"✅ LoRA Adapter Configured!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 4：加载与格式化 250 条企业微调数据集"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import Dataset\n",
    "import json\n",
    "\n",
    "# 如果你将 train.json 上传到了当前运行目录，直接读取：\n",
    "# with open(\"train.json\", \"r\", encoding=\"utf-8\") as f:\n",
    "#     raw_data = json.load(f)\n",
    "\n",
    "# 这里内置读取本地上传的 train.json\n",
    "dataset_path = \"train.json\"\n",
    "with open(dataset_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    raw_data = json.load(f)\n",
    "\n",
    "alpaca_prompt = \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
    "\n",
    "### Instruction:\n",
    "{}\n",
    "\n",
    "### Input:\n",
    "{}\n",
    "\n",
    "### Response:\n",
    "{}\"\"\"\n",
    "\n",
    "def formatting_prompts_func(examples):\n",
    "    instructions = examples[\"instruction\"]\n",
    "    inputs       = examples[\"input\"]\n",
    "    outputs      = examples[\"output\"]\n",
    "    texts = []\n",
    "    for instruction, input_text, output in zip(instructions, inputs, outputs):\n",
    "        # 必须以 EOS Token 结尾，教会大模型何时停止吐字！\n",
    "        text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token\n",
    "        texts.append(text)\n",
    "    return { \"text\" : texts, }\n",
    "\n",
    "dataset = Dataset.from_list(raw_data)\n",
    "dataset = dataset.map(formatting_prompts_func, batched = True)\n",
    "print(f\"✅ Successfully formatted {len(dataset)} training samples!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 5：启动 SFT 训练（大约耗时 5~8 分钟）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from trl import SFTTrainer\n",
    "from transformers import TrainingArguments\n",
    "from unsloth import is_bfloat16_supported\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model = model,\n",
    "    tokenizer = tokenizer,\n",
    "    train_dataset = dataset,\n",
    "    dataset_text_field = \"text\",\n",
    "    max_seq_length = max_seq_length,\n",
    "    dataset_num_proc = 2,\n",
    "    packing = False,\n",
    "    args = TrainingArguments(\n",
    "        per_device_train_batch_size = 2,\n",
    "        gradient_accumulation_steps = 4,\n",
    "        warmup_steps = 5,\n",
    "        max_steps = 60, # 250条数据跑 60 步约等于 2 个 Epoch，足以让小模型固化记忆\n",
    "        learning_rate = 2e-4,\n",
    "        fp16 = not is_bfloat16_supported(),\n",
    "        bf16 = is_bfloat16_supported(),\n",
    "        logging_steps = 5,\n",
    "        optim = \"adamw_8bit\",\n",
    "        weight_decay = 0.01,\n",
    "        lr_scheduler_type = \"linear\",\n",
    "        seed = 3407,\n",
    "        output_dir = \"outputs\",\n",
    "    ),\n",
    ")\n",
    "\n",
    "print(\"🚀 Starting QLoRA Fine-Tuning...\")\n",
    "trainer_stats = trainer.train()\n",
    "print(\"🎉 Training Complete!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 6：微调效果现场测试（验证是否还胡编'杭州'？）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "FastLanguageModel.for_inference(model) # 切换至极速推理模式\n",
    "\n",
    "test_prompt = \"\"\"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
    "\n",
    "### Instruction:\n",
    "你是企业智能行政助手。根据员工出差诉求，自主结合员工档案（籍贯、部门）与差旅标准，输出规范的核算结果，去除一切客套废话。\n",
    "\n",
    "### Input:\n",
    "帮张三查一下他回老家出差3天能领多少出差补贴？\n",
    "\n",
    "### Response:\n",
    "\"\"\"\n",
    "\n",
    "inputs = tokenizer([test_prompt], return_tensors = \"pt\").to(\"cuda\")\n",
    "outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)\n",
    "print(tokenizer.batch_decode(outputs)[0].split(\"### Response:\")[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 7：一键合并并导出为 Ollama 原生支持的 GGUF 量化格式！"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 一键将 LoRA 权重与 Base 模型合并，并量化为标准的 Q4_K_M GGUF 文件\n",
    "model.save_pretrained_gguf(\"my_qwen_enterprise\", tokenizer, quantization_method = \"q4_k_m\")\n",
    "print(\"✅ Model successfully exported to GGUF format! You can now download my_qwen_enterprise-q4_k_m.gguf to your local machine!\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = Path(__file__).resolve().parent.parent / "notebooks" / "qwen_lora_train.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ Jupyter Notebook created at: {output_path}")
