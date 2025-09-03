#!/usr/bin/env python3
"""
增强版训练脚本 - 解决文本生成效果差的问题
"""

import os
import re
import json
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    TextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from transformers import set_seed
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTextPreprocessor:
    """文本预处理器 - 清理和增强训练数据"""
    
    def __init__(self):
        self.chapter_pattern = re.compile(r'第[一二三四五六七八九十百千万]+[章节回集]', re.UNICODE)
        self.dialogue_pattern = re.compile(r'[""""].*?[""""]', re.UNICODE)
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text):
        """清理文本"""
        # 移除多余空白
        text = self.whitespace_pattern.sub(' ', text)
        # 移除特殊字符但保留中文标点
        text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef\w\s。，！？：；""""]', '', text)
        return text.strip()
    
    def split_into_sentences(self, text):
        """按句子分割文本"""
        # 中文句子分割
        sentences = re.split(r'[。！？]', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def create_training_chunks(self, sentences, chunk_size=256):
        """创建训练数据块"""
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

def create_enhanced_dataset():
    """创建增强版数据集"""
    preprocessor = EnhancedTextPreprocessor()
    
    # 读取text目录下所有txt文件
    import glob
    text_files = glob.glob("text/*.txt")
    
    # 过滤掉非小说文件
    exclude_files = ["text/目录.txt", "text/古龙武侠小说全集目录.txt"]
    text_files = [f for f in text_files if f not in exclude_files and f.endswith('.txt')]
    
    logger.info(f"发现 {len(text_files)} 个文本文件用于训练")
    
    all_sentences = []
    
    def read_file_utf8(file_path):
        """读取UTF-8格式文件（所有文件已统一为UTF-8）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取文件 {file_path} 失败: {e}")
            return ""
    
    for file_path in text_files:
        if os.path.exists(file_path):
            logger.info(f"处理文件: {file_path}")
            try:
                content = read_file_utf8(file_path)
                
                # 清理文本
                cleaned = preprocessor.clean_text(content)
                
                # 分割成句子
                sentences = preprocessor.split_into_sentences(cleaned)
                all_sentences.extend(sentences)
                
                logger.info(f"从 {file_path} 提取了 {len(sentences)} 个句子")
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                continue
    
    # 创建训练数据块
    chunks = preprocessor.create_training_chunks(all_sentences, chunk_size=256)
    
    # 保存增强版训练数据
    train_size = int(0.9 * len(chunks))
    
    with open("text/enhanced_train.txt", "w", encoding='utf-8') as f:
        for chunk in chunks[:train_size]:
            f.write(chunk + "\n")
    
    with open("text/enhanced_valid.txt", "w", encoding='utf-8') as f:
        for chunk in chunks[train_size:]:
            f.write(chunk + "\n")
    
    total_size_mb = sum(os.path.getsize(f) for f in text_files) / (1024*1024)
    logger.info(f"总数据量: {total_size_mb:.1f}MB")
    logger.info(f"创建了增强版数据集: 训练集{train_size}条, 验证集{len(chunks)-train_size}条")
    logger.info(f"使用的小说文件: {len(text_files)}个")
    return "text/enhanced_train.txt", "text/enhanced_valid.txt"

def create_optimized_model(tokenizer):
    """创建优化版模型"""
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,          # 标准GPT-2大小
        n_layer=12,          # 标准层数
        n_head=12,           # 标准注意力头
        dropout=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        activation_function="gelu",
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )
    
    model = GPT2LMHeadModel(config)
    logger.info(f"创建优化模型: 参数量 {sum(p.numel() for p in model.parameters()):,}")
    return model

def train_enhanced_model():
    """训练增强版模型"""
    set_seed(42)
    
    # 创建增强版数据集
    train_file, valid_file = create_enhanced_dataset()
    
    # 创建分词器
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 添加中文词汇
    with open(train_file, 'r', encoding='utf-8') as f:
        texts = f.read()
    
    # 训练分词器
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    
    # 创建中文分词器
    tokenizer_chinese = Tokenizer(models.BPE())
    tokenizer_chinese.pre_tokenizer = pre_tokenizers.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=50000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    
    tokenizer_chinese.train([train_file], trainer)
    
    # 保存分词器
    tokenizer_chinese.save("chinese_tokenizer.json")
    
    # 创建优化模型
    model = create_optimized_model(tokenizer)
    
    # 创建数据集
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=256,
        overwrite_cache=True
    )
    
    valid_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=valid_file,
        block_size=256,
        overwrite_cache=True
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./enhanced_output",
        overwrite_output_dir=True,
        num_train_epochs=10,  # 大幅增加训练轮次
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        warmup_steps=2000,
        weight_decay=0.01,
        logging_dir="./enhanced_logs",
        logging_steps=25,
        save_steps=1000,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        dataloader_num_workers=4,
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    
    logger.info("开始训练增强版模型...")
    trainer.train()
    
    # 保存模型
    trainer.save_model("./enhanced_chinese_model")
    tokenizer.save_pretrained("./enhanced_chinese_model")
    
    logger.info("训练完成！模型已保存至: ./enhanced_chinese_model")
    return model, tokenizer

if __name__ == "__main__":
    train_enhanced_model()