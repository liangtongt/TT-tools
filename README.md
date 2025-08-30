# TT img enc - ComfyUI 节点

## 功能描述

TT img enc 是一个 ComfyUI 自定义节点，具有以下功能：

1. **自动格式转换**：
   - 输入多张图片 → 自动转换为 MP4 视频
   - 输入单张图片 → 自动转换为 JPG 格式

2. **造点图片生成**：创建带有随机噪点的图片

3. **文件嵌入**：将转换后的文件（MP4/JPG）嵌入到造点图片中

4. **ZIP 解压**：下载图片后，将文件后缀改为 .zip 即可直接解压出原始文件

## 安装方法

### 方法1：Git Clone（推荐）
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tttools/tt-img-enc.git
```
然后重启 ComfyUI，节点会自动加载。

### 方法2：手动下载
1. 下载项目文件
2. 将整个项目文件夹复制到 ComfyUI 的 `custom_nodes` 目录
3. 重启 ComfyUI

### 方法3：Pip 安装
```bash
pip install git+https://github.com/tttools/tt-img-enc.git
```

## 使用方法

### 节点参数

- **images**: 输入的图片（支持多张）
- **fps**: 视频帧率（1-60，默认30）
- **quality**: JPG 质量（1-100，默认95）
- **noise_density**: 噪点密度（0.01-0.5，默认0.1）
- **noise_size**: 噪点大小（1-5，默认2）

### 工作流程

1. 连接图片输入到 `images` 端口
2. 调整其他参数（可选）
3. 运行工作流
4. 下载输出的造点图片
5. 将图片后缀改为 `.zip`
6. 解压 ZIP 文件获得原始 MP4 或 JPG 文件

## 技术原理

### 图片转视频
- 使用 OpenCV 将多张图片合成为 MP4 视频
- 支持自定义帧率
- 自动处理 RGB/BGR 颜色空间转换

### 图片转 JPG
- 使用 Pillow 将图片转换为高质量 JPG 格式
- 支持自定义质量参数

### 文件嵌入
- 使用 LSB（最低有效位）隐写术
- 将 ZIP 文件数据嵌入到图片像素中
- 支持任意大小的文件（受图片尺寸限制）

### 造点图片
- 白色背景 + 随机彩色噪点
- 可调节噪点密度和大小
- 固定 512x512 像素尺寸

## 注意事项

1. **图片尺寸**：输出图片尺寸根据文件大小动态调整（512x512 到 2048x2048）
2. **输出数量**：无论输入多少张图片，都只输出一张造点图片
3. **文件大小限制**：支持最大约1.5MB的文件（2048x2048图片）
4. **临时文件**：处理过程中会创建临时文件，完成后自动清理
5. **错误处理**：如果处理失败，会输出错误提示图片

## 示例工作流

```
Load Image → TT img enc → Save Image
```

## 故障排除

### 常见问题

1. **依赖缺失**：确保安装了所有必需的 Python 包
2. **内存不足**：处理大量图片时可能需要更多内存
3. **文件权限**：确保 ComfyUI 有权限创建临时目录
4. **torch兼容性**：确保ComfyUI环境中有torch支持

### 常见错误

- **OpenCV error: src is not a numpy array**：这通常意味着输入格式问题，v1.0.2已修复
- **'numpy.ndarray' object has no attribute 'cpu'**：这通常意味着torch未正确安装或版本不兼容
- **ImportError: No module named 'cv2'**：需要安装opencv-python
- **ImportError: No module named 'PIL'**：需要安装Pillow

### 错误信息

- 检查 ComfyUI 控制台的错误输出
- 验证输入图片格式是否正确
- 确认输出目录有写入权限

## 更新日志

- v1.0.3: 修复存储容量问题，支持大文件（如21帧视频），动态图片尺寸
- v1.0.2: 修复torch张量输入处理问题，自动转换ComfyUI输入格式
- v1.0.1: 修复torch兼容性和输出格式问题，确保只输出一张图片
- v1.0.0: 初始版本，支持基本的图片转视频/图片功能
