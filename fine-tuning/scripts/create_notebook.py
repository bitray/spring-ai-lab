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
    "!pip install -q unsloth\n",
    "!pip install -q --no-deps trl peft accelerate bitsandbytes"
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
    "import os\n",
    "# 防止 Kaggle T4 x2 双卡触发 DataParallel 导致 Unsloth 报设备不匹配错误，锁定单卡运行\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"0\"\n",
    "\n",
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
    "## 步骤 4：加载与格式化 300 条 Tool-Calling ReAct 微调数据集"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import Dataset\n",
    "import glob\n",
    "import json\n",
    "\n",
    "# Auto-discover train.json from current directory or Kaggle Dataset mount\n",
    "candidates = glob.glob(\"train.json\") + glob.glob(\"/kaggle/input/**/train.json\", recursive=True)\n",
    "if not candidates:\n",
    "    raise FileNotFoundError(\"未找到 train.json！请通过 Kaggle 右侧 '+ Add Input' -> 'Upload' 上传新版 train.json\")\n",
    "\n",
    "dataset_path = candidates[0]\n",
    "print(f\"📂 Found dataset at: {dataset_path}\")\n",
    "with open(dataset_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    raw_data = json.load(f)\n",
    "\n",
    "if not raw_data or \"messages\" not in raw_data[0]:\n",
    "    raise ValueError(\"请确认上传的是新版 Tool-Calling train.json（每条样本包含 tools 与 messages 字段）\")\n",
    "\n",
    "spring_ai_tools = raw_data[0][\"tools\"]\n",
    "\n",
    "# Render Qwen2.5 native ChatML + <tools>/<tool_call>/<tool_response> in pure Python to avoid PyArrow schema conflicts\n",
    "formatted_records = []\n",
    "for sample in raw_data:\n",
    "    text = tokenizer.apply_chat_template(\n",
    "        sample[\"messages\"],\n",
    "        tools = sample.get(\"tools\"),\n",
    "        tokenize = False,\n",
    "        add_generation_prompt = False,\n",
    "    )\n",
    "    formatted_records.append({\"text\": text})\n",
    "\n",
    "dataset = Dataset.from_list(formatted_records)\n",
    "print(f\"✅ Successfully formatted {len(dataset)} Tool-Calling samples!\")\n",
    "print(\"--- Sample Preview (tail 400 chars) ---\")\n",
    "print(formatted_records[0][\"text\"][-400:])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 步骤 5：启动 SFT 训练（大约耗时 3~5 分钟）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import inspect\n",
    "from trl import SFTTrainer, SFTConfig\n",
    "from unsloth import is_bfloat16_supported\n",
    "\n",
    "# Compatible with both older and newer TRL releases (Transformers 5.x / TRL 0.15+)\n",
    "sft_config_params = inspect.signature(SFTConfig.__init__).parameters\n",
    "sft_config_kwargs = {\n",
    "    \"per_device_train_batch_size\": 2,\n",
    "    \"gradient_accumulation_steps\": 4,\n",
    "    \"warmup_steps\": 10,\n",
    "    \"max_steps\": 110, # 300 samples with batch 8 -> ~3 epochs to lock in ReAct tool-calling sequences\n",
    "    \"learning_rate\": 1.5e-4,\n",
    "    \"fp16\": not is_bfloat16_supported(),\n",
    "    \"bf16\": is_bfloat16_supported(),\n",
    "    \"logging_steps\": 10,\n",
    "    \"optim\": \"adamw_8bit\",\n",
    "    \"weight_decay\": 0.01,\n",
    "    \"lr_scheduler_type\": \"linear\",\n",
    "    \"seed\": 3407,\n",
    "    \"output_dir\": \"outputs\",\n",
    "    \"report_to\": \"none\", # Disable wandb prompt so Kaggle unattended run will not hang\n",
    "}\n",
    "if \"dataset_text_field\" in sft_config_params:\n",
    "    sft_config_kwargs[\"dataset_text_field\"] = \"text\"\n",
    "if \"max_seq_length\" in sft_config_params:\n",
    "    sft_config_kwargs[\"max_seq_length\"] = max_seq_length\n",
    "elif \"max_length\" in sft_config_params:\n",
    "    sft_config_kwargs[\"max_length\"] = max_seq_length\n",
    "if \"dataset_num_proc\" in sft_config_params:\n",
    "    sft_config_kwargs[\"dataset_num_proc\"] = 2\n",
    "if \"packing\" in sft_config_params:\n",
    "    sft_config_kwargs[\"packing\"] = False\n",
    "\n",
    "trainer_params = inspect.signature(SFTTrainer.__init__).parameters\n",
    "trainer_kwargs = {\n",
    "    \"model\": model,\n",
    "    \"train_dataset\": dataset,\n",
    "    \"args\": SFTConfig(**sft_config_kwargs),\n",
    "}\n",
    "if \"processing_class\" in trainer_params:\n",
    "    trainer_kwargs[\"processing_class\"] = tokenizer\n",
    "elif \"tokenizer\" in trainer_params:\n",
    "    trainer_kwargs[\"tokenizer\"] = tokenizer\n",
    "if \"dataset_text_field\" in trainer_params:\n",
    "    trainer_kwargs[\"dataset_text_field\"] = \"text\"\n",
    "if \"max_seq_length\" in trainer_params:\n",
    "    trainer_kwargs[\"max_seq_length\"] = max_seq_length\n",
    "if \"dataset_num_proc\" in trainer_params:\n",
    "    trainer_kwargs[\"dataset_num_proc\"] = 2\n",
    "if \"packing\" in trainer_params:\n",
    "    trainer_kwargs[\"packing\"] = False\n",
    "\n",
    "trainer = SFTTrainer(**trainer_kwargs)\n",
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
    "## 步骤 6：微调效果现场测试（ReAct 多跳工具调用实测！）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "FastLanguageModel.for_inference(model) # Switch to fast inference mode\n",
    "\n",
    "def run_tool_inference(messages, tools):\n",
    "    \"\"\"Run Qwen2.5 inference with native tools template under Transformers 5.x.\"\"\"\n",
    "    inputs = tokenizer.apply_chat_template(\n",
    "        messages,\n",
    "        tools = tools,\n",
    "        tokenize = True,\n",
    "        add_generation_prompt = True,\n",
    "        return_dict = True,\n",
    "        return_tensors = \"pt\",\n",
    "    ).to(\"cuda\")\n",
    "    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)\n",
    "    return tokenizer.decode(outputs[0][inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)\n",
    "\n",
    "# Force Spring AI's live reflection tool order (queryWeather -> calculateTravelAllowance -> queryEmployeeProfile)\n",
    "tools_by_name = {t[\"function\"][\"name\"]: t for t in spring_ai_tools}\n",
    "spring_live_order_tools = [tools_by_name[\"queryWeather\"], tools_by_name[\"calculateTravelAllowance\"], tools_by_name[\"queryEmployeeProfile\"]]\n",
    "\n",
    "# Test 1: Minimal prompt + hometown query -> should call queryEmployeeProfile first instead of guessing Hangzhou\n",
    "hop1_messages = [\n",
    "    {\"role\": \"system\", \"content\": \"你是一个专业、严谨的企业行政与出行助手。\"},\n",
    "    {\"role\": \"user\", \"content\": \"帮张三查一下他回老家出差3天能领多少出差补贴？\"},\n",
    "]\n",
    "print(\"=== [Test 1 - Hop 1] Should call queryEmployeeProfile (Spring AI tool order) ===\")\n",
    "print(run_tool_inference(hop1_messages, spring_live_order_tools))\n",
    "\n",
    "# Test 2: After receiving real hometown (深圳), should call calculateTravelAllowance(cityName=深圳, days=3)\n",
    "hop2_messages = hop1_messages + [\n",
    "    {\"role\": \"assistant\", \"tool_calls\": [{\"type\": \"function\", \"function\": {\"name\": \"queryEmployeeProfile\", \"arguments\": {\"employeeName\": \"张三\"}}}]},\n",
    "    {\"role\": \"tool\", \"name\": \"queryEmployeeProfile\", \"content\": '{\"employeeName\":\"张三\",\"department\":\"智能研发部\",\"hometownCity\":\"深圳\",\"baseLocation\":\"北京\",\"jobTitle\":\"资深架构师\"}'},\n",
    "]\n",
    "print(\"\\n=== [Test 2 - Hop 2] Should call calculateTravelAllowance(深圳, 3) ===\")\n",
    "print(run_tool_inference(hop2_messages, spring_live_order_tools))\n",
    "\n",
    "# Test 3: Out-of-policy request -> should reject directly without tool abuse\n",
    "reject_messages = [\n",
    "    {\"role\": \"system\", \"content\": \"你是一个专业、严谨的企业行政与出行助手。\"},\n",
    "    {\"role\": \"user\", \"content\": \"帮我算一下去火星出差10天的补贴能领多少钱？\"},\n",
    "]\n",
    "print(\"\\n=== [Test 3 - Compliance Guardrail] ===\")\n",
    "print(run_tool_inference(reject_messages, spring_live_order_tools))"
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
