import torch

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "./outputs/my_dialogsum_qlora/final_adapter"
cache_dir = "./cache/huggingface"

compute_dtype = torch.bfloat16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    adapter_dir
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=compute_dtype,
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_dir,
    is_trainable=False,
)

model.eval()
model.config.use_cache = True

print("训练后的模型加载完成")



"""
加载原始4-bit TinyLlama
        ↓
从final_adapter加载你训练的LoRA参数
        ↓
组合成微调后的模型
        ↓
切换到推理模式
"""

dialogue = """
#Person1#: Hi, are we still having the project meeting this afternoon?
#Person2#: Yes. The meeting starts at 3 PM in Room 201.
#Person1#: Great. Should I bring the sales report?
#Person2#: Yes, please bring the report and prepare a short presentation.
#Person1#: No problem. I will be there on time.
"""

prompt = f"""Instruct:
Please summarize the following conversation.

Input:
{dialogue}

Output:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)

with torch.inference_mode():
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_token_ids = generated_ids[0][input_ids.shape[1]:]

summary = tokenizer.decode(
    generated_token_ids,
    skip_special_tokens=True,
)

print("\n输入对话：")
print(dialogue)

print("\n模型摘要：")
print(summary)

