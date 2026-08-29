# TinyLlama QLoRA Conversation Summarization

这是一个从零学习大模型微调的实践项目：使用 DialogSum 对话摘要数据，在单张 NVIDIA GPU 上对 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 进行 4-bit QLoRA 微调，并通过命令行输入对话生成英文摘要。

## 项目内容

- `my_finetune.py`：加载数据、4-bit 量化基础模型、添加 LoRA 并训练。
- `inference.py`：使用固定示例验证微调结果。
- `use_model.py`：交互式输入对话并生成摘要。
- `check_env.py`：检查 PyTorch、CUDA 和相关依赖。
- `setup_windows.ps1`：在 Windows 上创建虚拟环境并安装依赖。
- `outputs/my_dialogsum_qlora/final_adapter/`：训练得到的 LoRA adapter 和 tokenizer 文件。

仓库不包含 TinyLlama 完整基础模型。程序第一次运行时会从 Hugging Face 下载基础模型，并与仓库中的 LoRA adapter 组合使用。

## 环境要求

- Windows 10/11
- Python 3.10 或 3.11
- NVIDIA GPU（本项目在 RTX 5060 Laptop 8GB 上完成）
- 支持 CUDA 的显卡驱动

在 PowerShell 中运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

脚本会创建 `.venv`，安装 CUDA 13.0 版本的 PyTorch 2.9.1，以及 `requirements-local.txt` 中的训练依赖。

## 准备数据

本项目使用 DialogSum 格式的 CSV 数据。由于数据集不放入仓库，请在项目根目录创建 `data` 文件夹，并放入：

```text
data/
├── train.csv
├── validation.csv
└── test.csv
```

每个 CSV 至少需要包含 `dialogue` 和 `summary` 两列。

## 开始训练

```powershell
.\.venv\Scripts\python.exe .\my_finetune.py
```

训练完成后，LoRA adapter 会保存到：

```text
outputs/my_dialogsum_qlora/final_adapter/
```

训练配置包括 4-bit NF4、双重量化、BF16 计算，以及作用于 `q_proj`、`k_proj`、`v_proj`、`o_proj` 的 LoRA。

## 使用微调后的模型

```powershell
.\.venv\Scripts\python.exe .\use_model.py
```

输入多行英文对话，输入完成后单独输入 `END`；输入 `quit` 退出程序。

示例：

```text
#Person1#: Are you coming to the meeting tomorrow?
#Person2#: Yes, but I may arrive ten minutes late.
#Person1#: Please bring the sales report.
#Person2#: Sure, I will bring it.
END
```

当前模型使用英文 DialogSum 数据进行微调，因此更适合生成英文对话摘要。

## 模型文件说明

`final_adapter/adapter_model.safetensors` 只包含训练得到的 LoRA 参数，不是完整的 TinyLlama 模型。推理时仍需加载 `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 基础模型。

`checkpoint-*`、模型下载缓存、Python 虚拟环境及训练数据均被 `.gitignore` 排除，不会提交到 GitHub。

