import os
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    LineByLineTextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, pipeline, set_seed
)
from tokenizers import CharBPETokenizer
from tokenizers.decoders import BPEDecoder
import random

# 禁用wandb
os.environ["WANDB_DISABLED"] = "true"

# 定义保存路径
SAVE_PATH = "./chinese_novel_model"

# 1. 创建优化的中文分词器
def create_custom_tokenizer(files):
    """创建基于字符级BPE的中文分词器"""
    # 初始化字符级BPE分词器
    tokenizer = CharBPETokenizer(
        suffix="",  # 不添加后缀
        unk_token="<unk>",  # 未知词标记
    )
    
    # 添加特殊标记
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    # 训练分词器
    tokenizer.train(
        files,
        vocab_size=8000,  # 更小的词汇表
        min_frequency=2,  # 最小出现频率
        show_progress=True,
        special_tokens=special_tokens
    )
    
    # 将分词器转换为Hugging Face格式
    tokenizer.save_model(SAVE_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
    
    # 添加特殊标记映射
    tokenizer.add_special_tokens({
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    
    return tokenizer

# 2. 创建小型GPT-2模型
def create_small_gpt2_model(tokenizer):
    """创建小型GPT-2模型配置"""
    # 定义小型模型配置
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # 小型模型参数 - 减少层数和维度
        n_layer=6,          # 原始为12层 (减少50%)
        n_head=6,           # 原始为12头 (减少50%)
        n_embd=384,         # 原始为768 (减少50%)
        n_inner=1536,       # 原始为3072 (减少50%)
        n_positions=256,    # 原始为1024 (减少75%)
        # 其他优化参数
        resid_pdrop=0.1,    # 残差dropout
        embd_pdrop=0.1,     # 嵌入dropout
        attn_pdrop=0.1      # 注意力dropout
    )
    
    # 创建模型
    model = GPT2LMHeadModel(config)
    
    # 打印模型大小
    num_params = sum(p.numel() for p in model.parameters())
    print(f"创建小型GPT-2模型，参数数量: {num_params/1e6:.1f}M (原始GPT-2约为124M)")
    
    return model

# 3. 创建数据集
def create_datasets(tokenizer):
    """创建训练和验证数据集"""
    # 小说文件列表
    novel_files = [
        "text/sanguoyanyi.txt",
        "text/tian_long_ba_bu_all.txt"
    ]
    
    # 创建训练数据集
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=random.choice(novel_files),  # 随机选择一本
        block_size=128
    )
    
    # 创建验证数据集
    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=random.choice(novel_files),  # 使用不同小说
        block_size=128
    )
    
    return train_dataset, valid_dataset

# 4. 主训练函数
def train_model():
    """训练小型GPT-2模型"""
    # 小说文件列表
    novel_files = [
        "text/sanguoyanyi.txt",
        "text/tian_long_ba_bu_all.txt",
        # "text/other_novel1.txt",  # 添加更多小说
        # "text/other_novel2.txt"
    ]
    
    # 创建分词器
    tokenizer = create_custom_tokenizer(novel_files)
    
    # 创建小型模型
    model = create_small_gpt2_model(tokenizer)
    
    # 创建数据集
    train_dataset, valid_dataset = create_datasets(tokenizer)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=64  # 优化显存使用
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=10,  # 减少训练轮次
        per_device_train_batch_size=8,  # 适合小型模型的批次大小
        per_device_eval_batch_size=8,
        learning_rate=3e-5,  # 更小的学习率
        warmup_steps=500,  # 预热步骤
        weight_decay=0.01,  # 权重衰减
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        eval_strategy="steps",  # 定期评估
        eval_steps=500,  # 每500步评估一次
        prediction_loss_only=True,
        gradient_accumulation_steps=4,  # 梯度累积
        fp16=False  # 禁用混合精度训练以兼容macOS
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset  # 添加验证集
    )
    
    # 开始训练
    print("开始训练小型GPT-2模型...")
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"模型已保存至: {SAVE_PATH}")
    
    return model, tokenizer

# 5. 生成文本示例
def generate_text_examples(model, tokenizer):
    """生成多个文本示例"""
    generate_single_text(model, tokenizer, "吕布手持方天画戟", 100, multiple=True)

def generate_single_text(model, tokenizer, prompt=None, max_length=100, multiple=False):
    """生成单个文本或批量文本
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        prompt: 生成提示词，如果为None则使用默认提示
        max_length: 生成文本最大长度
        multiple: 是否生成多个示例
    """
    # 创建文本生成管道
    try:
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1  # 使用CPU，兼容macOS
        )
    except:
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer
        )
    
    # 设置随机种子
    set_seed(42)
    
    if multiple:
        # 测试不同提示
        prompts = [
            "吕布手持方天画戟",
            "乔峰降龙十八掌",
            "郭靖和黄蓉",
            "诸葛亮摇着羽扇"
        ]
    else:
        prompts = [prompt or "吕布手持方天画戟"]
    
    # 生成文本
    for p in prompts:
        print(f"\n提示: {p}")
        try:
            # 优化的生成参数
            output = generator(
                p,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            print(f"生成文本: {output[0]['generated_text']}\n")
        except Exception as e:
            print(f"生成文本时出错: {e}")
            # 简化参数重试
            try:
                output = generator(p, max_length=max_length, do_sample=True)
                print(f"生成文本: {output[0]['generated_text']}\n")
            except Exception as e2:
                print(f"重试也失败: {e2}")

# 主执行函数 - 分离训练和测试
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='小型GPT-2模型训练与文本生成')
    parser.add_argument('--mode', choices=['train', 'generate', 'both'], default='both',
                        help='运行模式：train=仅训练, generate=仅生成, both=训练+生成')
    parser.add_argument('--model_path', type=str, default=SAVE_PATH,
                        help='模型保存路径')
    parser.add_argument('--prompt', type=str, default="吕布手持方天画戟",
                        help='生成文本的提示词')
    parser.add_argument('--max_length', type=int, default=100,
                        help='生成文本的最大长度')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # 仅训练模型
        model, tokenizer = train_model()
        print("训练完成！模型已保存至:", args.model_path)
        
    elif args.mode == 'generate':
        # 仅生成文本，使用已训练的模型
        try:
            print(f"从 {args.model_path} 加载模型...")
            tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
            model = GPT2LMHeadModel.from_pretrained(args.model_path)
            
            # 生成单个文本示例
            generate_single_text(model, tokenizer, args.prompt, args.max_length)
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请先运行训练: python train_small_model.py --mode train")
            
    else:  # both
        # 训练并生成文本（原始行为）
        model, tokenizer = train_model()
        generate_text_examples(model, tokenizer)