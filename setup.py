#!/usr/bin/env python3
"""
TT img enc ComfyUI 节点安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements文件
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tt-img-enc",
    version="1.0.0",
    author="TT Tools",
    author_email="support@tttools.com",
    description="ComfyUI custom node for automatic image/video conversion with steganography",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tttools/tt-img-enc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    keywords=["comfyui", "node", "image", "video", "conversion", "steganography"],
    project_urls={
        "Bug Reports": "https://github.com/tttools/tt-img-enc/issues",
        "Source": "https://github.com/tttools/tt-img-enc",
        "Documentation": "https://github.com/tttools/tt-img-enc#readme",
    },
    include_package_data=True,
    zip_safe=False,
)
