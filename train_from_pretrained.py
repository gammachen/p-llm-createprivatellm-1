# 完整改进方案
from transformers import BertTokenizer, GPT2LMHeadModel

# 1. 使用预训练中文模型
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-lyric")
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-lyric")

# 2. 扩展数据集
files = [f"text/{name}.txt" for name in 
         ["sanguoyanyi", "tian_long_ba_bu_all"]]

tokenizer.train(files)

# 3. 配置动态填充
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=64  # 显存优化
)

# 4. 添加验证集
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=2000,
    logging_steps=100,
    prediction_loss_only=True
)

# 5. 训练时监控
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,  # 必须添加
    data_collator=data_collator
)