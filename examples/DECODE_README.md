# TT img dec 解码节点使用说明

## 概述

TT img dec 是一个 ComfyUI 节点，用于从造点图片中提取隐藏的文件（MP4、JPG等）。这个节点基于 `extract_zip.py` 的成功解码逻辑开发。

## 功能特点

- ✅ 支持从造点图片中提取各种格式的文件
- ✅ 自动识别文件类型和扩展名
- ✅ 智能LSB隐写数据提取
- ✅ 完整的错误处理和状态反馈
- ✅ 支持自定义输出文件名

## 节点参数

### 输入参数

- **image** (IMAGE, 必需): 包含隐藏文件的造点图片
- **output_filename** (STRING, 可选): 输出文件名，默认为 "extracted_file"

### 输出参数

- **status** (STRING): 提取状态信息
- **file_path** (STRING): 提取文件的完整路径

## 使用方法

### 1. 基本使用

1. 在 ComfyUI 中添加 `LoadImage` 节点，加载造点图片
2. 添加 `TTImgDecNode` 节点
3. 连接 `LoadImage` 的 `IMAGE` 输出到 `TTImgDecNode` 的 `image` 输入
4. 运行工作流

### 2. 工作流示例

参考 `decode_workflow.json` 文件，这是一个完整的工作流示例：

```
LoadImage → TTImgDecNode → PreviewText
```

### 3. 输出文件

提取的文件会保存在ComfyUI的 `output/` 目录下，文件名格式为：
- 如果指定了 `output_filename`: `{output_filename}.{扩展名}`
- 如果未指定: `extracted_file.{扩展名}`

## 支持的文件格式

- **视频文件**: MP4, AVI, MOV 等
- **图片文件**: JPG, PNG, GIF 等
- **文档文件**: PDF, DOC, TXT 等
- **其他文件**: 任何二进制文件

## 技术原理

该节点使用 LSB (Least Significant Bit) 隐写技术：

1. **数据提取**: 从图片每个像素的RGB通道最低位提取二进制数据
2. **长度解析**: 前32位用于存储数据长度信息
3. **文件头解析**: 解析文件扩展名和实际数据
4. **数据重建**: 将二进制数据转换回原始文件

## 注意事项

1. **图片要求**: 必须是3通道RGB格式的造点图片
2. **文件完整性**: 确保图片完整下载，避免损坏
3. **存储空间**: 确保有足够的磁盘空间保存提取的文件
4. **权限**: 确保对ComfyUI的 `output/` 目录有写入权限

## 错误处理

节点会提供详细的错误信息：

- ❌ 图片格式错误
- ❌ 数据长度不足
- ❌ 文件头数据不完整
- ❌ LSB提取失败

## 使用场景

- 从社交媒体分享的造点图片中提取原文件
- 恢复被隐藏的重要文档
- 验证造点图片的完整性
- 批量处理多个造点图片

## 相关链接

- 教程: https://b23.tv/RbvaMeW
- B站: 我是小斯呀
- 编码节点: `tt_img_enc_node.py`

## 更新日志

- v1.0: 基于 `extract_zip.py` 创建初始版本
- 支持 ComfyUI 节点接口
- 完整的错误处理和状态反馈
