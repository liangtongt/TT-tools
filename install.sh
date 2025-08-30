#!/bin/bash

echo "Installing ComfyUI Image Sequence Compressor Node..."
echo

# 检查是否在ComfyUI custom_nodes目录中
if [ ! -f "../../main.py" ]; then
    echo "Error: This script must be run from the ComfyUI custom_nodes directory"
    echo "Please navigate to your ComfyUI custom_nodes folder and run this script"
    exit 1
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo
echo "Installation completed!"
echo "Please restart ComfyUI to load the new node."
echo
