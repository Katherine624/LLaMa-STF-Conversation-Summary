import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# 1. 设置基础模型和 LoRA 参数的位置
# ============================================================

# 原始的 TinyLlama 基础模型
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 你刚刚训练得到的 LoRA 参数
adapter_dir = "./outputs/my_dialogsum_qlora/final_adapter"

# 模型下载缓存位置
cache_dir = "./cache/huggingface"


# ============================================================
# 2. 配置 4-bit 量化
# ============================================================

compute_dtype = torch.bfloat16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 3. 加载分词器
# ============================================================

print("正在加载 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    adapter_dir,
)

# 有些模型没有专门的 pad_token
# 这里使用 eos_token 作为 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# 4. 加载原始 TinyLlama 基础模型
# ============================================================

print("正在加载 TinyLlama 基础模型...")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=compute_dtype,
)


# ============================================================
# 5. 把训练好的 LoRA 参数安装到基础模型上
# ============================================================

print("正在加载训练好的 LoRA 参数...")

model = PeftModel.from_pretrained(
    base_model,
    adapter_dir,
    is_trainable=False,
)

# 切换到推理模式
model.eval()

print("模型加载完成！")
print("接下来可以输入对话，让模型生成摘要。")
print("每次输入完一段对话后，单独输入 END。")
print("如果想退出程序，请直接输入 quit。")


# ============================================================
# 6. 持续接收用户输入
# ============================================================

while True:
    print("\n请输入需要总结的对话：")

    dialogue_lines = []

    while True:
        line = input()

        # 输入 quit，退出整个程序
        if line.strip().lower() == "quit":
            print("程序已退出。")
            raise SystemExit

        # 输入 END，代表当前对话输入完成
        if line.strip() == "END":
            break

        dialogue_lines.append(line)

    # 把多行对话拼接成一个字符串
    dialogue = "\n".join(dialogue_lines).strip()

    if not dialogue:
        print("你没有输入对话，请重新输入。")
        continue


    # ========================================================
    # 7. 按照训练时的格式制作 prompt
    # ========================================================

    # 必须和训练时使用的提示词格式保持一致
    prompt = f"""Instruct:
Please summarize the following conversation.

Input:
{dialogue}

Output:
"""


    # ========================================================
    # 8. 将文字转换成模型能够处理的 token
    # ========================================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # 把输入数据放到模型所在的 GPU
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)


    # ========================================================
    # 9. 让模型生成摘要
    # ========================================================

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


    # ========================================================
    # 10. 去掉原来的 prompt，只保留模型新生成的摘要
    # ========================================================

    prompt_length = input_ids.shape[1]

    summary_ids = generated_ids[0, prompt_length:]


    # ========================================================
    # 11. 把 token 转换回正常文字
    # ========================================================

    summary = tokenizer.decode(
        summary_ids,
        skip_special_tokens=True,
    )

    print("\n模型生成的 Summary：")
    print(summary.strip())

