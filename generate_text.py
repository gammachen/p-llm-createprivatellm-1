#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的文本生成脚本
无需训练，直接使用已保存的模型进行文本生成
"""

import argparse
import argparse
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline, set_seed

# 默认模型路径
DEFAULT_MODEL_PATH = "./chinese_novel_model"

def load_model_and_tokenizer(model_path):
    """加载模型和分词器"""
    try:
        print(f"正在从 {model_path} 加载模型...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        print("模型加载成功！")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型已训练并保存在指定路径")
        return None, None

def generate_text(model, tokenizer, prompt, max_length=100, num_sequences=1):
    """生成文本"""
    if model is None or tokenizer is None:
        return []
    
    # 设置随机种子
    set_seed(42)
    
    # 生成文本 - 优化版
    try:
        # 使用优化的生成参数
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
        
        # 优化的生成参数
        generation_params = {
            'max_new_tokens': max_length,
            'do_sample': True,
            'temperature': 0.8,  # 稍微提高温度增加多样性
            'top_k': 50,        # 限制词汇选择
            'top_p': 0.9,       # 核心采样
            'repetition_penalty': 1.2,  # 更强的重复惩罚
            'no_repeat_ngram_size': 3,  # 避免3-gram重复
            'pad_token_id': tokenizer.eos_token_id,
            'num_return_sequences': num_sequences,
            'early_stopping': True
        }
        
        # 添加上下文提示
        enhanced_prompt = f"{prompt}"
        if len(prompt) < 10:  # 如果提示太短，添加一些上下文
            enhanced_prompt = f"在武侠世界中，{prompt}"
        
        outputs = generator(enhanced_prompt, **generation_params)
        
        results = []
        for output in outputs:
            generated_text = output['generated_text']
            results.append({'generated_text': generated_text})
        return results
        
    except Exception as e:
        print(f"生成文本时出错: {e}")
        # 简化参数重试
        try:
            generator = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer
            )
            outputs = generator(prompt, max_length=max_length, do_sample=True)
            results = []
            for output in outputs:
                generated_text = output['generated_text']
                results.append({'generated_text': generated_text})
            return results
        except Exception as e2:
            print(f"重试也失败: {e2}")
            return []

def interactive_mode(model, tokenizer):
    """交互式生成模式"""
    print("\n=== 中文小说文本生成器 ===")
    print("输入提示词开始生成，输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助信息\n")
    
    while True:
        try:
            prompt = input("请输入提示词: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("感谢使用，再见！")
                break
            
            if prompt.lower() == 'help':
                print("\n帮助信息:")
                print("- 直接输入中文提示词，如：吕布手持方天画戟")
                print("- 输入 'quit' 或 'exit' 退出程序")
                print("- 提示词越具体，生成的文本越相关\n")
                continue
            
            if not prompt:
                print("请输入有效的提示词！")
                continue
            
            print("正在生成文本...")
            outputs = generate_text(model, tokenizer, prompt, max_length=150)
            
            if outputs:
                print(f"\n提示: {prompt}")
                print("=" * 50)
                for i, output in enumerate(outputs, 1):
                    generated = output['generated_text']
                    # 清理输出，移除特殊token
                    generated = generated.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                    print(f"生成文本 {i}: {generated}\n")
            else:
                print("生成失败，请重试")
                
        except KeyboardInterrupt:
            print("\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")

def batch_generate(model, tokenizer, prompts, max_length=100):
    """批量生成文本"""
    results = []
    for prompt in prompts:
        outputs = generate_text(model, tokenizer, prompt, max_length)
        if outputs:
            results.append({
                'prompt': prompt,
                'generated': outputs[0]['generated_text']
            })
    return results

def main():
    parser = argparse.ArgumentParser(description='中文小说文本生成器')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='模型路径 (默认: ./chinese_novel_model)')
    parser.add_argument('--prompt', type=str, 
                        help='要生成的提示词')
    parser.add_argument('--max_length', type=int, default=100,
                        help='生成文本的最大长度')
    parser.add_argument('--num_sequences', type=int, default=1,
                        help='为每个提示词生成的文本数量')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式模式')
    parser.add_argument('--batch', nargs='+',
                        help='批量生成多个提示词')
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径 {args.model_path} 不存在")
        print("请确保模型已训练完成，或指定正确的模型路径")
        sys.exit(1)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    if model is None or tokenizer is None:
        sys.exit(1)
    
    # 根据不同模式运行
    if args.interactive:
        interactive_mode(model, tokenizer)
    
    elif args.batch:
        print("批量生成文本...")
        results = batch_generate(model, tokenizer, args.batch, args.max_length)
        for result in results:
            print(f"\n提示: {result['prompt']}")
            print("-" * 40)
            generated = result['generated'].replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
            print(f"生成: {generated}\n")
    
    elif args.prompt:
        print(f"生成文本: {args.prompt}")
        outputs = generate_text(model, tokenizer, args.prompt, args.max_length, args.num_sequences)
        
        for i, output in enumerate(outputs, 1):
            generated = output['generated_text'].replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
            print(f"\n生成文本 {i}: {generated}\n")
    
    else:
        # 默认生成示例
        print("使用默认提示词生成示例文本...")
        default_prompts = [
            "吕布手持方天画戟",
            "乔峰降龙十八掌",
            "郭靖和黄蓉",
            "诸葛亮摇着羽扇"
        ]
        
        for prompt in default_prompts:
            outputs = generate_text(model, tokenizer, prompt, args.max_length)
            if outputs:
                generated = outputs[0]['generated_text'].replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
                print(f"\n提示: {prompt}")
                print(f"生成: {generated}\n")

if __name__ == "__main__":
    main()