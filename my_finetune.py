import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

#加载数据+tokenization
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

data_files = {
    "train": "./data/train.csv",
    "validation": "./data/validation.csv",
    "test": "./data/test.csv",
}

dataset = load_dataset(
    "csv",
    data_files=data_files
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="./cache/huggingface"
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

"""
dataset：训练、验证和测试数据。
model_name：后面加载底座模型也会使用。
tokenizer：把文本转换成模型输入。
pad_token：训练组成 batch 时用于补齐。
padding_side="right"：在短样本右侧补齐。
"""


#预处理训练集和验证集
max_length = 512


def preprocess_batch(batch):
    texts = []
#zip() 用来把多个列表中相同位置的元素配在一起。
    for dialogue, summary in zip(
        batch["dialogue"],
        batch["summary"]
    ):
        text = f"""
Instruct:
Please summarize the following conversation.

Input:
{dialogue}

Output:
{summary}"""

        texts.append(text)

    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length
    )


train_dataset = dataset["train"].map(
    preprocess_batch,
    batched=True,
    remove_columns=dataset["train"].column_names
)

validation_dataset = dataset["validation"].map(
    preprocess_batch,
    batched=True,
    remove_columns=dataset["validation"].column_names
)

#以4-bit方式加载底座模型
#添加量化配置
compute_dtype = torch.bfloat16
#量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

#加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./cache/huggingface",
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=compute_dtype,
)

print("4-bit底座模型加载完成")


#给4-bit底座模型添加 LoRA
model = prepare_model_for_kbit_training(
    model,  #函数接收刚才加载的4-bit模型bu
    use_gradient_checkpointing=True  #正常训练时，模型前向计算产生的很多中间结果都会保存在显存里，以便反向传播使用。
)
#创建 LoRA 配置
lora_config = LoraConfig(
    r=16,  #r:秩
    lora_alpha=32,  #LoRA算出的修改量 ΔW 需要进行缩放。alpha / r
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],   #在模型的哪些层上添加 LoRA。
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    model,
    lora_config
)

model.config.use_cache = False
model.print_trainable_parameters()

#创建 Data Collator 和训练参数
#Data Collator 就是：把若干条独立数据整理成一个可以交给模型的 batch。
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,   #mlm=False 表示我们不是训练 BERT 那种“遮住一个词再猜”的模型，而是训练从左到右预测下一个 token 的因果语言模型。
)
#添加训练参数
training_args = TrainingArguments(
    output_dir="./outputs/my_dialogsum_qlora",

    num_train_epochs=1,  #表示完整遍历一次训练集。

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  #每4条更新一次

    learning_rate=2e-4,
    warmup_ratio=0.05,    #表示前5%的更新步骤用于学习率预热。学习率不会第一步就直接使用0.0002
    weight_decay=0.01,

    optim="paged_adamw_8bit",

    bf16=True,
    fp16=False,

    gradient_checkpointing=True,

    logging_steps=10,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    report_to="none",
    seed=42,
)


#创建 Trainer 并开始训练
model.config.pad_token_id = tokenizer.pad_token_id

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
)
"""
model              4-bit TinyLlama + LoRA
training_args      训练规则
train_dataset      训练数据
validation_dataset 验证数据
data_collator      动态padding并创建labels
"""

#正式开始训练
print("开始训练")

train_result = trainer.train()

print("训练完成")
print(train_result)

#保存最终 LoRA
final_adapter_dir = "./outputs/my_dialogsum_qlora/final_adapter"

trainer.model.save_pretrained(final_adapter_dir)
tokenizer.save_pretrained(final_adapter_dir)

print("LoRA Adapter已保存到：")
print(final_adapter_dir)
"""
保存的主要是：
LoRA参数
LoRA配置
Tokenizer配置和词表
"""






















#测试

