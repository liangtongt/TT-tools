# TT img - ComfyUI 节点套件

## 功能描述

TT img 是一个 ComfyUI 自定义节点套件，包含编码和解码两个节点：

### TT img enc（编码节点）
1. **自动格式转换**：
   - 输入多张图片 → 自动转换为 MP4 视频
   - 输入单张图片 → 自动转换为 JPG 格式

2. **造点图片生成**：创建带有随机噪点的图片

3. **文件嵌入**：将转换后的文件（MP4/JPG）嵌入到造点图片中

### TT img dec（解码节点）
1. **文件提取**：从造点图片中提取隐藏的文件（MP4/JPG等）
2. **自动识别**：自动识别文件类型和扩展名
3. **智能解码**：使用LSB隐写技术提取数据
4. **状态反馈**：提供详细的提取状态和错误信息

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

### TT img enc（编码节点）

#### 节点参数
- **images**: 输入的图片（支持多张）
- **fps**: 视频帧率（1-60，默认30）
- **quality**: JPG 质量（1-100，默认95）

#### 工作流程
1. 连接图片输入到 `images` 端口
2. 调整其他参数（可选）
3. 运行工作流
4. 下载输出的造点图片

### TT img dec（解码节点）

#### 节点参数
- **image**: 包含隐藏文件的造点图片
- **output_filename**: 输出文件名（可选，默认为 "extracted_file"）

#### 工作流程
1. 连接造点图片到 `image` 端口
2. 设置输出文件名（可选）
3. 运行工作流
4. 查看提取状态和文件路径
5. 提取的文件保存在 `temp/` 目录下

### 完整工作流示例

**编码工作流**：
```
Load Image → TT img enc → Save Image
```

**解码工作流**：
```
Load Image → TT img dec → Preview Text
```

**完整循环**：
```
Load Image → TT img enc → Save Image → Load Image → TT img dec → Preview Text
```

### 文件提取

**方法1：使用ComfyUI解码节点（推荐）**
- 直接在ComfyUI中使用 `TT img dec` 节点
- 自动识别文件类型和扩展名
- 提供详细的提取状态反馈

**方法2：使用Python脚本**
```bash
python extract_zip.py <图片路径>
```

**方法3：指定输出路径**
```bash
python extract_zip.py <图片路径> <输出文件路径>
```

**示例：**
```bash
python extract_zip.py output_image.png
python extract_zip.py output_image.png my_video.mp4
python extract_zip.py output_image.png my_image.jpg
```

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
- 直接嵌入原始文件数据（不使用ZIP压缩）
- 保持文件扩展名信息
- 支持任意大小的文件（受图片尺寸限制）

### 存储图片
- 纯色背景（灰色，128值）
- 专为存储效率优化
- 动态图片尺寸（最小64x64，无上限）
- 根据文件大小自动调整

## 注意事项

1. **图片尺寸**：输出图片尺寸根据文件大小动态调整（最小64x64，无上限）
2. **输出数量**：无论输入多少张图片，都只输出一张存储图片
3. **文件大小限制**：无限制，根据文件大小自动调整图片尺寸
4. **临时文件**：处理过程中会创建临时文件，完成后自动清理
5. **错误处理**：如果处理失败，会输出错误提示图片
6. **文件提取**：必须使用提供的提取工具，不能直接改后缀名

## 示例工作流

### 编码工作流
```
Load Image → TT img enc → Save Image
```

### 解码工作流
```
Load Image → TT img dec → Preview Text
```

### 完整循环工作流
```
Load Image → TT img enc → Save Image → Load Image → TT img dec → Preview Text
```

参考 `examples/` 目录下的工作流文件：
- `basic_workflow.json`: 基本编码工作流
- `video_workflow.json`: 视频编码工作流  
- `decode_workflow.json`: 解码工作流

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

### v2.0.0 - 新增解码节点
- ✅ 新增 `TT img dec` 解码节点
- ✅ 基于 `extract_zip.py` 的成功解码逻辑
- ✅ 支持ComfyUI节点接口
- ✅ 完整的错误处理和状态反馈
- ✅ 自动文件类型识别

### v1.0.8 - 编码节点优化
- 移除噪点设置，优化存储效率，最小图片尺寸降至64x64
- 移除图片尺寸上限限制，支持任意大小文件
- 实现动态图片尺寸，支持大文件存储（如865KB MP4）
- 移除ZIP压缩，直接存储原始文件，提高存储效率
- 修复ZIP提取问题，提供专用提取工具，确保数据完整性
- 修复存储容量问题，支持大文件（如21帧视频），动态图片尺寸
- 修复torch张量输入处理问题，自动转换ComfyUI输入格式
- 修复torch兼容性和输出格式问题，确保只输出一张图片
- 初始版本，支持基本的图片转视频/图片功能
