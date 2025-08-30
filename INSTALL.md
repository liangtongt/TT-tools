# 安装和使用说明

## 系统要求

- Python 3.7+
- ComfyUI
- Windows/Linux/macOS

## 安装步骤

### 1. 安装Python依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install Pillow numpy
```

### 2. 安装ComfyUI节点

1. 将 `image_sequence_compressor.py` 复制到你的ComfyUI `custom_nodes` 目录
2. 重启ComfyUI
3. 在节点列表中找到 "Image Sequence Compressor" 节点

### 3. 验证安装

运行演示脚本：

```bash
python demo.py
```

## 使用方法

### 在ComfyUI中使用

1. **添加节点**：在节点列表中找到 "Image Sequence Compressor"
2. **连接输入**：
   - `images`: 连接图片序列（可以是多个LoadImage节点的输出）
   - `filename`: 设置输出文件名
   - `compression_level`: 设置压缩级别（1-9）
   - `quality`: 设置图片质量（1-100）
   - `format`: 选择输出格式（PNG/JPEG/WEBP）
   - `include_metadata`: 是否包含元数据
   - `output_directory`: 输出目录
3. **运行工作流**：点击运行按钮

### 使用Python脚本提取

```bash
python extract_images.py compressed_image.png output_directory
```

### 参数说明

- `compression_level`: 压缩级别，数值越高压缩率越高，但处理时间越长
- `quality`: 图片质量，仅对JPEG和WEBP格式有效
- `format`: 输出格式，PNG支持无损压缩，JPEG和WEBP支持有损压缩

## 工作原理

### 压缩过程

1. 将每张图片转换为指定格式并压缩
2. 使用zlib进行二次压缩
3. 将压缩后的数据编码为base64
4. 将所有数据打包成JSON格式
5. 将JSON数据嵌入到第一张图片的底部像素中

### 提取过程

1. 读取图片底部的像素数据
2. 解析JSON数据
3. 解码base64数据
4. 使用zlib解压缩
5. 还原为原始图片格式

## 文件结构

```
tt_img/
├── image_sequence_compressor.py    # ComfyUI节点主文件
├── extract_images.py               # Python提取脚本
├── demo.py                         # 演示脚本
├── create_test_images.py           # 测试图片生成脚本
├── requirements.txt                # Python依赖
├── example_workflow.json           # 示例工作流
├── README.md                       # 项目说明
├── INSTALL.md                      # 安装说明
└── test_images/                    # 测试图片目录
```

## 故障排除

### 常见问题

1. **节点不显示**
   - 检查文件是否放在正确的 `custom_nodes` 目录
   - 重启ComfyUI
   - 检查Python依赖是否安装

2. **压缩失败**
   - 检查输入图片格式是否支持
   - 确保有足够的磁盘空间
   - 检查输出目录权限

3. **提取失败**
   - 确认压缩文件完整
   - 检查文件是否被修改
   - 验证Python脚本权限

### 性能优化

- 对于大量图片，建议分批处理
- 使用适当的压缩级别平衡文件大小和处理时间
- 选择合适的输出格式（PNG适合需要无损压缩的场景）

## 高级用法

### 自定义压缩参数

```python
from image_sequence_compressor import ImageSequenceCompressor

compressor = ImageSequenceCompressor()
compressor.compression_level = 9  # 最高压缩级别
compressor.quality = 80           # 较低质量，更小文件
```

### 批量处理

```python
import os
from PIL import Image

# 批量压缩目录中的所有图片
image_dir = "input_images"
output_dir = "compressed"
os.makedirs(output_dir, exist_ok=True)

# 这里可以集成到你的工作流中
```

## 技术支持

如果遇到问题，请检查：

1. Python版本兼容性
2. 依赖包版本
3. 文件权限
4. 磁盘空间

## 更新日志

- v1.0.0: 初始版本，支持基本的图片序列压缩和提取功能
- 支持PNG、JPEG、WEBP格式
- 支持可配置的压缩参数
- 提供完整的Python提取脚本
