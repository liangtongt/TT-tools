#!/usr/bin/env python3
"""
TT img enc 节点安装脚本
自动安装节点到 ComfyUI 的 custom_nodes 目录
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def find_comfyui_custom_nodes():
    """查找 ComfyUI 的 custom_nodes 目录"""
    possible_paths = [
        # 常见的 ComfyUI 安装路径
        "custom_nodes",
        "../custom_nodes",
        "../../custom_nodes",
        # Windows 用户目录
        os.path.expanduser("~/ComfyUI/custom_nodes"),
        os.path.expanduser("~/Desktop/ComfyUI/custom_nodes"),
        os.path.expanduser("~/Downloads/ComfyUI/custom_nodes"),
        # 当前目录的子目录
        "./ComfyUI/custom_nodes",
        "./custom_nodes"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None

def install_dependencies():
    """安装 Python 依赖包"""
    print("正在安装 Python 依赖包...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 依赖包安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖包安装失败: {e}")
        return False

def install_node(custom_nodes_path):
    """安装节点到 custom_nodes 目录"""
    print(f"正在安装节点到: {custom_nodes_path}")
    
    # 源文件路径
    source_file = "tt_img_enc_node.py"
    
    if not os.path.exists(source_file):
        print(f"✗ 找不到源文件: {source_file}")
        return False
    
    # 目标文件路径
    target_file = os.path.join(custom_nodes_path, "tt_img_enc_node.py")
    
    try:
        # 复制文件
        shutil.copy2(source_file, target_file)
        print(f"✓ 节点文件安装成功: {target_file}")
        
        # 复制 README 文件
        if os.path.exists("README.md"):
            readme_target = os.path.join(custom_nodes_path, "TT_img_enc_README.md")
            shutil.copy2("README.md", readme_target)
            print(f"✓ README 文件安装成功: {readme_target}")
        
        return True
    except Exception as e:
        print(f"✗ 节点安装失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    dirs = ["output", "temp"]
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✓ 创建目录: {dir_name}")

def main():
    """主安装流程"""
    print("=" * 50)
    print("TT img enc 节点安装程序")
    print("=" * 50)
    
    # 查找 custom_nodes 目录
    custom_nodes_path = find_comfyui_custom_nodes()
    
    if not custom_nodes_path:
        print("✗ 找不到 ComfyUI 的 custom_nodes 目录")
        print("\n请手动将 tt_img_enc_node.py 文件复制到 ComfyUI 的 custom_nodes 目录")
        print("或者运行此脚本时确保在正确的目录中")
        return False
    
    print(f"✓ 找到 custom_nodes 目录: {custom_nodes_path}")
    
    # 安装依赖
    if not install_dependencies():
        print("警告: 依赖包安装失败，节点可能无法正常工作")
        print("请手动运行: pip install -r requirements.txt")
    
    # 安装节点
    if not install_node(custom_nodes_path):
        return False
    
    # 创建目录
    create_directories()
    
    print("\n" + "=" * 50)
    print("安装完成！")
    print("=" * 50)
    print("下一步操作:")
    print("1. 重启 ComfyUI")
    print("2. 在节点列表中找到 'TT Tools' 分类")
    print("3. 拖拽 'TT img enc' 节点到工作区")
    print("4. 连接图片输入并运行工作流")
    print("\n如有问题，请查看 README.md 文件或检查 ComfyUI 控制台输出")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n安装被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n安装过程中发生错误: {e}")
        sys.exit(1)
