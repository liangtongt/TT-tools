#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理ComfyUI和Python缓存
"""

import os
import shutil
import glob
import sys

def clear_cache():
    """清理所有可能的缓存"""
    
    print("🧹 开始清理缓存...")
    
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 清理当前目录的Python缓存
    print("\n📁 清理当前目录缓存...")
    cache_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd"
    ]
    
    for pattern in cache_patterns:
        if pattern == "__pycache__":
            # 删除__pycache__目录
            pycache_dir = os.path.join(current_dir, "__pycache__")
            if os.path.exists(pycache_dir):
                print(f"  删除: {pycache_dir}")
                shutil.rmtree(pycache_dir)
        else:
            # 删除.pyc等文件
            files = glob.glob(os.path.join(current_dir, pattern))
            for file in files:
                print(f"  删除: {file}")
                os.remove(file)
    
    # 2. 查找ComfyUI目录
    print("\n🔍 查找ComfyUI目录...")
    possible_comfyui_paths = [
        os.path.join(current_dir, "..", ".."),  # 当前目录的上两级
        os.path.join(current_dir, "..", "..", ".."),  # 当前目录的上三级
        "C:\\work\\runninghub\\pack\\ComfyUI",  # 用户提到的路径
    ]
    
    comfyui_found = False
    for path in possible_comfyui_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # 检查是否是ComfyUI目录
            if os.path.exists(os.path.join(path, "main.py")) or os.path.exists(os.path.join(path, "nodes.py")):
                print(f"  找到ComfyUI目录: {path}")
                comfyui_found = True
                
                # 清理ComfyUI的缓存
                comfyui_cache_dir = os.path.join(path, "__pycache__")
                if os.path.exists(comfyui_cache_dir):
                    print(f"  删除ComfyUI缓存: {comfyui_cache_dir}")
                    shutil.rmtree(comfyui_cache_dir)
                
                # 清理custom_nodes缓存
                custom_nodes_dir = os.path.join(path, "custom_nodes")
                if os.path.exists(custom_nodes_dir):
                    print(f"  检查custom_nodes缓存: {custom_nodes_dir}")
                    for item in os.listdir(custom_nodes_dir):
                        item_path = os.path.join(custom_nodes_dir, item)
                        if os.path.isdir(item_path):
                            pycache_path = os.path.join(item_path, "__pycache__")
                            if os.path.exists(pycache_path):
                                print(f"    删除 {item} 缓存: {pycache_path}")
                                shutil.rmtree(pycache_path)
                
                break
    
    if not comfyui_found:
        print("  ⚠️  未找到ComfyUI目录，请手动清理")
    
    # 3. 清理Python字节码缓存
    print("\n🐍 清理Python字节码缓存...")
    try:
        import py_compile
        # 强制重新编译
        print("  强制重新编译Python文件...")
        for py_file in glob.glob(os.path.join(current_dir, "*.py")):
            if py_file != __file__:  # 不重新编译自己
                try:
                    py_compile.compile(py_file, doraise=True)
                    print(f"    重新编译: {os.path.basename(py_file)}")
                except Exception as e:
                    print(f"    编译失败 {os.path.basename(py_file)}: {e}")
    except ImportError:
        print("  py_compile模块不可用")
    
    print("\n✅ 缓存清理完成！")
    print("\n📋 下一步操作:")
    print("  1. 重启ComfyUI")
    print("  2. 如果问题仍然存在，检查工作流配置")
    print("  3. 检查其他节点是否影响输出格式")

if __name__ == "__main__":
    clear_cache()
