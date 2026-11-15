# 🚀 企业级大模型微调实战（Phase 4：Fine-Tuning & SFT）

本项目包含了从**数据工程、QLoRA 轻量微调、GGUF 量化导出，到导入本地 Ollama 供 Java Spring AI 调用**的全生命周期工程资产。

---

## 一、 目录结构说明

```
fine-tuning/
├── data/
│   └── train.json                 # 250 条标准 Alpaca 格式的企业微调训练集
├── notebooks/
│   └── qwen_lora_train.ipynb      # 一键开箱即用的 Kaggle / Colab 微调 Notebook
├── scripts/
│   ├── generate_dataset.py        # 训练数据合成与批量扩充脚本
│   └── create_notebook.py         # Notebook 生成维护脚本
├── Modelfile                      # 本地 Ollama 导入量化模型的标准配置文件
└── README.md                      # 操作指南与架构说明
```

---

## 二、 训练数据核心能力分布（250 条 Alpaca 样本）

1. **多步依赖链式推理（100 条）**：
   - 看到“X回老家出差”，自动调取员工档案中的真实籍贯城市（深圳、成都、武汉等），杜绝未经训练时的“凭空捏造杭州”。
2. **确定性规章制度严格执行（80 条）**：
   - 一线城市（北上广深）150元/天，二线及其他城市 100元/天，100% 精准核算。
3. **多天数隐式心算与差额变更（40 条）**：
   - 处理原定天数 + 追加天数的累计与差额核算。
4. **合规边界拦截（30 条）**：
   - 面对火星、月球、公海会所等非合法营商区域，100% 触发严肃的 `【合规驳回提醒】`。

---

## 三、 云端极速训练指南（推荐 Kaggle / 免费 T4 GPU）

> **为什么推荐 Kaggle？**  
> 每周免费赠送 30 小时 NVIDIA T4（16G 显存）算力，无需本地配置复杂的 Python/CUDA 驱动，约 5~8 分钟即可训完。

### 操作步骤：
1. 打开并登录 [Kaggle](https://www.kaggle.com/)，点击左侧菜单栏 **Create -> New Notebook**；
2. 在 Notebook 页面顶部菜单：`File -> Import Notebook`，选择本项目中的 [qwen_lora_train.ipynb](file:///C:/workspace/ai_rag_test/fine-tuning/notebooks/qwen_lora_train.ipynb) 上传；
3. **关键配置（页面右侧 Notebook Settings 面板）**：
   - **Accelerator**：选择 **GPU T4**（或 GPU T4 x2）；
   - **Internet**：设置为 **Internet on**（确保可以联网拉取 Unsloth 和模型权重）；
4. **上传数据**：在右侧面板点击 **Input -> + Add Input -> Upload（上传）**，选择本项目的 [train.json](file:///C:/workspace/ai_rag_test/fine-tuning/data/train.json) 创建一个数据集（例如起名 `enterprise-travel-train`）；Notebook 已内置自动扫描 `/kaggle/input/**/train.json`，无需手动改路径；
5. 点击顶部菜单 **Run All（全部运行）**；
6. 训练完成后，在右侧输出目录中找到导出的 `my_qwen_enterprise-q4_k_m.gguf`（约 1.8GB~2GB），下载回本地电脑。

---

## 四、 本地 Ollama 投产与 Java Spring AI 闭环

### 1. 导入本地 Ollama
将下载的 `my_qwen_enterprise-q4_k_m.gguf` 放到与 `Modelfile` 同一目录下，在终端执行：
```bash
ollama create my-qwen-enterprise -f Modelfile
```
验证模型已就绪：
```bash
ollama run my-qwen-enterprise "帮张三查一下他回老家出差3天能领多少出差补贴？"
```

### 2. Spring AI Java 工程无缝切换
在 `spring-ai-demo/src/main/resources/application.yml` 中：
```yaml
spring:
  ai:
    ollama:
      chat:
        options:
          model: my-qwen-enterprise # 切换为你自己炼制的新模型！
```
现在，即便 `AgentController.java` 中的 System Prompt 彻底留空，您的本地专属小模型也能 100% 精准走完老家出差核算，零客套废话，毫秒级极速响应！
