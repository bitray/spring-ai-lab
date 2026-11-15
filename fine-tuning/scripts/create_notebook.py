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
    "import glob\n",
    "import json\n",
    "\n",
    "# 自动查找工作区或 Kaggle Input 挂载目录 (/kaggle/input/<你的数据集名>/train.json)\n",
    "candidates = glob.glob(\"train.json\") + glob.glob(\"/kaggle/input/**/train.json\", recursive=True)\n",
    "if not candidates:\n",
    "    raise FileNotFoundError(\"未找到 train.json！请通过 Kaggle 右侧 '+ Add Input' -> 'Upload' 上传 train.json\")\n",
    "\n",
    "dataset_path = candidates[0]\n",
    "print(f\"📂 Found dataset at: {dataset_path}\")\n",
    "with open(dataset_path, \"r\", encoding=\"utf-8\") as f:\n",
    "    raw_data = json.load(f)\n",
    "\n",
    "def formatting_prompts_func(examples):\n",
    "    instructions = examples[\"instruction\"]\n",
    "    inputs       = examples[\"input\"]\n",
    "    outputs      = examples[\"output\"]\n",
    "    texts = []\n",
    "    for instruction, input_text, output in zip(instructions, inputs, outputs):\n",
    "        messages = [\n",
    "            {\"role\": \"system\", \"content\": instruction},\n",
    "            {\"role\": \"user\", \"content\": input_text},\n",
    "            {\"role\": \"assistant\", \"content\": output},\n",
    "        ]\n",
    "        # 使用 Qwen2.5 原生 ChatML 模板格式化（自动带 <|im_start|> 和 <|im_end|>），与本地 Ollama Modelfile 100% 对齐！\n",
    "        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)\n",
    "        texts.append(text)\n",
    "    return { \"text\" : texts, }\n",
    "\n",
    "dataset = Dataset.from_list(raw_data)\n",
    "dataset = dataset.map(formatting_prompts_func, batched = True, remove_columns = dataset.column_names)\n",
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
    "import inspect\n",
    "from trl import SFTTrainer, SFTConfig\n",
    "from unsloth import is_bfloat16_supported\n",
    "\n",
    "# Compatible with both older and newer TRL releases (Transformers 5.x / TRL 0.15+)\n",
    "sft_config_params = inspect.signature(SFTConfig.__init__).parameters\n",
    "sft_config_kwargs = {\n",
    "    \"per_device_train_batch_size\": 2,\n",
    "    \"gradient_accumulation_steps\": 4,\n",
    "    \"warmup_steps\": 5,\n",
    "    \"max_steps\": 60, # 250 samples * 60 steps (~2 epochs), enough to lock in multi-hop tool calling\n",
    "    \"learning_rate\": 2e-4,\n",
    "    \"fp16\": not is_bfloat16_supported(),\n",
    "    \"bf16\": is_bfloat16_supported(),\n",
    "    \"logging_steps\": 5,\n",
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
    "## 步骤 6：微调效果现场测试（验证是否还胡编'杭州'？）"
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
    "# Minimal system prompt without chain-of-thought hints to verify internalized multi-hop behavior\n",
    "messages = [\n",
    "    {\"role\": \"system\", \"content\": \"你是企业智能行政助手。\"},\n",
    "    {\"role\": \"user\", \"content\": \"帮张三查一下他回老家出差3天能领多少出差补贴？\"},\n",
    "]\n",
    "\n",
    "inputs = tokenizer.apply_chat_template(\n",
    "    messages,\n",
    "    tokenize = True,\n",
    "    add_generation_prompt = True,\n",
    "    return_dict = True,\n",
    "    return_tensors = \"pt\",\n",
    ").to(\"cuda\")\n",
    "\n",
    "outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)\n",
    "decoded = tokenizer.decode(outputs[0][inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)\n",
    "print(decoded)"
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
