#!/usr/bin/env python3
"""
批量将text目录下的所有txt文件转换为UTF-8编码
支持ISO-8859、GB2312、GBK、GB18030、Big5等多种编码
"""

import os
import glob
import shutil
from pathlib import Path

def detect_and_convert_encoding(file_path):
    """检测文件编码并转换为UTF-8"""
    encodings = ['utf-8', 'gb2312', 'gbk', 'gb18030', 'big5', 'iso-8859-1']
    
    # 先尝试读取文件
    content = None
    detected_encoding = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                detected_encoding = encoding
                break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print(f"❌ 无法读取文件: {file_path}")
        return False
    
    # 如果已经是UTF-8，跳过转换
    if detected_encoding == 'utf-8':
        print(f"✅ {os.path.basename(file_path)} 已经是UTF-8格式")
        return True
    
    # 创建备份
    backup_path = f"{file_path}.backup"
    shutil.copy2(file_path, backup_path)
    
    # 转换为UTF-8
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"🔄 {os.path.basename(file_path)} 从 {detected_encoding} 转换为 UTF-8")
        return True
    except Exception as e:
        print(f"❌ 转换失败 {file_path}: {str(e)}")
        # 恢复备份
        shutil.copy2(backup_path, file_path)
        return False

def main():
    """主函数"""
    text_dir = "text"
    txt_files = glob.glob(os.path.join(text_dir, "*.txt"))
    
    if not txt_files:
        print("❌ 未找到任何txt文件")
        return
    
    print(f"📁 找到 {len(txt_files)} 个txt文件，开始转换为UTF-8...")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for file_path in sorted(txt_files):
        filename = os.path.basename(file_path)
        
        # 跳过备份文件
        if filename.endswith('.backup'):
            continue
            
        result = detect_and_convert_encoding(file_path)
        if result:
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n📊 转换完成统计:")
    print(f"   ✅ 成功转换: {success_count} 个文件")
    print(f"   ⏭️  跳过(已UTF-8): {skip_count} 个文件")
    print(f"   ❌ 转换失败: {error_count} 个文件")
    
    # 清理备份文件
    backup_files = glob.glob(os.path.join(text_dir, "*.txt.backup"))
    if backup_files:
        print(f"\n🗑️  清理 {len(backup_files)} 个备份文件...")
        for backup in backup_files:
            os.remove(backup)
        print("   备份文件已清理")
    
    print("\n🎉 UTF-8转换完成！")

if __name__ == "__main__":
    main()