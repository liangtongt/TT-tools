# ComfyUI Image Sequence Compression Node

这是一个ComfyUI自定义节点，用于将图片序列压缩写入到一个图片文件中，并可以通过Python代码还原这些图片序列。

## 功能特性

- 将多张图片序列压缩到一个图片文件中
- 支持多种图片格式（PNG, JPG等）
- 提供Python还原脚本
- 可配置压缩质量
- 支持元数据存储

## 安装方法

### 方法1: Git Clone (推荐)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-image-sequence-compressor.git
cd comfyui-image-sequence-compressor
pip install -r requirements.txt
```

然后重启ComfyUI即可。

### 方法2: 手动安装

1. 将整个项目文件夹复制到你的ComfyUI `custom_nodes` 目录
2. 安装Python依赖：`pip install -r requirements.txt`
3. 重启ComfyUI
4. 在节点列表中找到 "Image Sequence Compressor" 节点

### 方法3: 使用安装脚本

**Windows:**
```bash
cd ComfyUI/custom_nodes/comfyui-image-sequence-compressor
install.bat
```

**Linux/macOS:**
```bash
cd ComfyUI/custom_nodes/comfyui-image-sequence-compressor
chmod +x install.sh
./install.sh
```

## 使用方法

### 在ComfyUI中使用

1. 添加 "Image Sequence Compressor" 节点
2. 连接输入：
   - `images`: 需要压缩的图片序列
   - `image`: 承载压缩数据的容器图片
3. 设置压缩参数（压缩级别、质量、格式等）
4. 运行工作流，输出为包含压缩数据的图像

### Python还原脚本

使用 `extract_from_image.py` 脚本来从图像中提取压缩的图片序列：

```bash
python extract_from_image.py output_image.png output_directory
```

**注意**: 现在节点直接输出图像，而不是保存文件。你可以：
1. 在ComfyUI中查看输出的图像
2. 保存图像到本地
3. 使用Python脚本从保存的图像中提取原始图片序列

## 文件结构

- `image_sequence_compressor.py` - ComfyUI节点主文件
- `extract_from_image.py` - 从图像中提取压缩数据的Python脚本
- `extract_images.py` - 从压缩文件中提取的Python脚本（兼容旧版本）
- `requirements.txt` - Python依赖
- `example_workflow.json` - 示例工作流
- `test_images/` - 测试图片目录

## 依赖

- ComfyUI
- Pillow (PIL)
- numpy
- base64
- json
