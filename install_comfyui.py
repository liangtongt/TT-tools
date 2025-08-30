#!/usr/bin/env python3
"""
ComfyUI 专用安装脚本
自动检测 ComfyUI 安装路径并安装 TT img enc 节点
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def find_comfyui_path():
    """查找 ComfyUI 安装路径"""
    possible_paths = [
        # 当前目录
        os.getcwd(),
        # 上级目录
        os.path.dirname(os.getcwd()),
        # 上上级目录
        os.path.dirname(os.path.dirname(os.getcwd())),
        # 用户目录
        os.path.expanduser("~"),
        # 桌面
        os.path.expanduser("~/Desktop"),
        # 下载目录
        os.path.expanduser("~/Downloads"),
        # 文档目录
        os.path.expanduser("~/Documents"),
    ]
    
    for path in possible_paths:
        # 检查是否是 ComfyUI 目录
        if os.path.exists(os.path.join(path, "main.py")) and os.path.exists(os.path.join(path, "custom_nodes")):
            return path
        # 检查是否有 ComfyUI 子目录
        comfyui_subdir = os.path.join(path, "ComfyUI")
        if os.path.exists(os.path.join(comfyui_subdir, "main.py")) and os.path.exists(os.path.join(comfyui_subdir, "custom_nodes")):
            return comfyui_subdir
    
    return None

def install_to_comfyui(comfyui_path):
    """安装到 ComfyUI"""
    custom_nodes_path = os.path.join(comfyui_path, "custom_nodes")
    target_path = os.path.join(custom_nodes_path, "tt-img-enc")
    
    print(f"正在安装到: {target_path}")
    
    # 如果目标目录已存在，先删除
    if os.path.exists(target_path):
        print("目标目录已存在，正在删除...")
        shutil.rmtree(target_path)
    
    # 复制当前项目到目标目录
    try:
        shutil.copytree(".", target_path, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", "output", "temp", "*.png", "*.jpg", "*.mp4"
        ))
        print(f"✓ 节点安装成功: {target_path}")
        return True
    except Exception as e:
        print(f"✗ 安装失败: {e}")
        return False

def install_dependencies():
    """安装 Python 依赖"""
    print("正在安装 Python 依赖包...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 依赖包安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖包安装失败: {e}")
        print("请手动运行: pip install -r requirements.txt")
        return False

def main():
    """主安装流程"""
    print("=" * 60)
    print("TT img enc - ComfyUI 节点安装程序")
    print("=" * 60)
    
    # 查找 ComfyUI 路径
    comfyui_path = find_comfyui_path()
    
    if not comfyui_path:
        print("✗ 找不到 ComfyUI 安装路径")
        print("\n请确保:")
        print("1. 在 ComfyUI 目录中运行此脚本")
        print("2. 或者在 ComfyUI 的 custom_nodes 目录中运行")
        print("3. 或者手动指定 ComfyUI 路径")
        print("\n手动安装方法:")
        print("1. 将整个项目文件夹复制到 ComfyUI/custom_nodes/ 目录")
        print("2. 重启 ComfyUI")
        return False
    
    print(f"✓ 找到 ComfyUI: {comfyui_path}")
    
    # 安装依赖
    install_dependencies()
    
    # 安装节点
    if not install_to_comfyui(comfyui_path):
        return False
    
    print("\n" + "=" * 60)
    print("安装完成！")
    print("=" * 60)
    print("下一步操作:")
    print("1. 重启 ComfyUI")
    print("2. 在节点列表中找到 'TT Tools' 分类")
    print("3. 拖拽 'TT img enc' 节点到工作区")
    print("4. 连接图片输入并运行工作流")
    print("\n示例工作流文件位于: examples/ 目录")
    print("如有问题，请查看 README.md 文件或检查 ComfyUI 控制台输出")
    
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
