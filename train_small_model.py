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

def read_file_utf8(file_path):
    """读取UTF-8格式文件（所有文件已统一为UTF-8）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return []

# 2. 创建小型GPT-2模型
def create_small_gpt2_model(tokenizer):
    """创建小型GPT-2模型配置"""
    # 创建中型GPT-2模型，参数数量: 约50M (平衡性能和资源)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,      # 增加上下文长度
        n_ctx=1024,
        n_embd=512,          # 增加嵌入维度
        n_layer=12,          # 增加层数
        n_head=8,            # 保持注意力头数量
        dropout=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    # 创建模型
    model = GPT2LMHeadModel(config)
    
    # 打印模型大小
    num_params = sum(p.numel() for p in model.parameters())
    print(f"创建小型GPT-2模型，参数数量: {num_params/1e6:.1f}M (原始GPT-2约为124M)")
    
    return model

# 3. 创建数据集
def create_datasets(tokenizer, novel_files):
    """创建训练和验证数据集
    
    Args:
        tokenizer: 分词器
        novel_files: 小说文件列表，确保与训练分词器使用的文件一致
    """
    print(f"正在使用 {len(novel_files)} 个文件创建数据集...")
    
    # 合并所有文本文件作为训练数据
    all_texts = []
    for file_path in novel_files:
        lines = read_file_utf8(file_path)
        all_texts.extend(lines)
    
    # 创建更大的训练数据集
    train_size = int(0.9 * len(all_texts))
    train_texts = all_texts[:train_size]
    valid_texts = all_texts[train_size:]
    
    print(f"总文本行数: {len(all_texts)}, 训练集: {len(train_texts)}, 验证集: {len(valid_texts)}")
    
    # 保存合并的文本用于训练
    with open("text/combined_train.txt", "w", encoding='utf-8') as f:
        f.writelines(train_texts)
    
    with open("text/combined_valid.txt", "w", encoding='utf-8') as f:
        f.writelines(valid_texts)
    
    # 创建训练数据集
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="text/combined_train.txt",
        block_size=256  # 增加块大小
    )
    
    # 创建验证数据集
    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="text/combined_valid.txt",
        block_size=256  # 增加块大小
    )
    
    return train_dataset, valid_dataset

# 4. 主训练函数
def train_model():
    """训练小型GPT-2模型"""
    # 获取所有小说文件列表 - 确保分词器和数据集使用相同的文件
    import glob
    novel_files = glob.glob("text/*.txt")
    
    # 过滤掉非小说文件
    exclude_files = ["text/目录.txt", "text/古龙武侠小说全集目录.txt"]
    novel_files = [f for f in novel_files if f not in exclude_files and f.endswith('.txt')]
    
    print(f"将使用 {len(novel_files)} 个武侠小说文件进行训练")
    
    # 创建分词器 - 使用所有文件
    tokenizer = create_custom_tokenizer(novel_files)
    
    # 创建小型模型
    model = create_small_gpt2_model(tokenizer)
    
    # 创建数据集 - 使用相同的文件
    train_dataset, valid_dataset = create_datasets(tokenizer, novel_files)
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=64  # 优化显存使用
    )
    
    # 训练参数 - 优化版本
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=50,  # 大幅增加训练轮次
        per_device_train_batch_size=4,  # 减小批次大小以适应更多数据
        per_device_eval_batch_size=4,
        learning_rate=5e-5,  # 标准学习率
        warmup_steps=1000,  # 增加预热步骤
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,  # 更频繁的日志记录
        save_steps=500,    # 更频繁的保存
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=250,    # 更频繁的评估
        prediction_loss_only=True,
        gradient_accumulation_steps=8,  # 增加梯度累积
        fp16=False,
        dataloader_num_workers=2,  # 数据加载优化
        load_best_model_at_end=True,  # 加载最佳模型
        metric_for_best_model="eval_loss",
        greater_is_better=False
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