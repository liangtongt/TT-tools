# 示例工作流

这个目录包含了TT img节点套件的使用示例，包括编码和解码节点。

## 文件说明

### 编码节点示例
- `basic_workflow.json` - 基础工作流示例
- `video_workflow.json` - 视频生成工作流示例

### 解码节点示例
- `decode_workflow.json` - 解码工作流示例

### 完整循环示例
- `complete_workflow.json` - 编码+解码完整循环工作流

## 使用方法

1. 在ComfyUI中加载这些工作流文件
2. 根据需要修改输入图片路径
3. 运行工作流
4. 查看输出结果

## 工作流说明

### 编码节点示例

#### basic_workflow.json
最简单的使用方式，单张图片转JPG并嵌入造点图片。

#### video_workflow.json
多张图片转MP4视频并嵌入造点图片。

### 解码节点示例

#### decode_workflow.json
从造点图片中提取隐藏文件，展示解码节点的基本用法。

### 完整循环示例

#### complete_workflow.json
完整的编码+解码循环工作流，展示从输入图片到最终提取文件的完整流程。
