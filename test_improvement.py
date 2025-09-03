#!/usr/bin/env python3
"""
测试改进效果脚本
"""

import os
import sys
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed

def test_generation_quality():
    """测试生成质量"""
    
    # 测试提示词
    test_prompts = [
        "吕布",
        "青光闪动",
        "乔峰",
        "郭靖和黄蓉",
        "在武侠世界中",
        "剑气纵横"
    ]
    
    # 检查模型是否存在
    model_paths = [
        "./enhanced_chinese_model",
        "./chinese_novel_model",
        "./output"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "config.json")):
            model_path = path
            break
    
    if not model_path:
        print("❌ 未找到训练好的模型")
        print("请先运行训练脚本:")
        print("  python train_small_model.py --mode train")
        print("或")
        print("  python enhance_training.py")
        return
    
    print(f"✅ 使用模型: {model_path}")
    
    try:
        # 加载模型和分词器
        print("正在加载模型...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # 创建生成器
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
        
        print("\n" + "="*50)
        print("开始测试文本生成质量")
        print("="*50)
        
        for prompt in test_prompts:
            print(f"\n📝 提示: {prompt}")
            print("-" * 30)
            
            try:
                # 使用优化的参数生成
                outputs = generator(
                    prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    num_return_sequences=3,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                for i, output in enumerate(outputs, 1):
                    text = output['generated_text']
                    # 计算生成长度
                    generated_part = text[len(prompt):].strip()
                    char_count = len(generated_part)
                    word_count = len(generated_part.split())
                    
                    print(f"  生成{i}: {text}")
                    print(f"  新增字符: {char_count}, 新增词汇: {word_count}")
                    
                    # 质量评估
                    if char_count > 10:
                        print("  ✅ 生成质量: 良好")
                    elif char_count > 5:
                        print("  ⚠️  生成质量: 一般")
                    else:
                        print("  ❌ 生成质量: 较差")
                    print()
                    
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")
                print()
    
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("请检查模型文件是否完整")

def diagnose_issues():
    """诊断常见问题"""
    
    print("\n" + "="*50)
    print("诊断报告")
    print("="*50)
    
    # 检查训练数据
    text_files = [
        "text/sanguoyanyi.txt",
        "text/tian_long_ba_bu_all.txt",
        "text/romeo_and_juliet.txt"
    ]
    
    total_lines = 0
    for file_path in text_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"📊 {file_path}: {lines} 行")
        else:
            print(f"❌ {file_path}: 文件不存在")
    
    print(f"📊 总文本行数: {total_lines}")
    
    if total_lines < 50000:
        print("⚠️  警告: 训练数据量可能不足")
        print("💡 建议: 增加更多文本文件")
    
    # 检查模型文件
    model_files = [
        "chinese_novel_model/config.json",
        "chinese_novel_model/model.safetensors",
        "output/config.json"
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"📦 {file_path}: {size:.1f} MB")
        else:
            print(f"❌ {file_path}: 不存在")
    
    print("\n💡 改进建议:")
    print("1. 增加训练数据量 (建议100MB+)")
    print("2. 增加训练轮次 (建议50-100轮)")
    print("3. 使用更大的模型架构")
    print("4. 优化文本预处理")
    print("5. 调整生成参数")

if __name__ == "__main__":
    print("🚀 文本生成质量测试工具")
    print("="*50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose_issues()
    else:
        test_generation_quality()