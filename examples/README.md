# 示例工作流

这个目录包含了TT img enc节点的使用示例。

## 文件说明

- `basic_workflow.json` - 基础工作流示例
- `video_workflow.json` - 视频生成工作流示例
- `batch_workflow.json` - 批量处理工作流示例

## 使用方法

1. 在ComfyUI中加载这些工作流文件
2. 根据需要修改输入图片路径
3. 运行工作流
4. 下载输出图片并改后缀为.zip解压

## 工作流说明

### basic_workflow.json
最简单的使用方式，单张图片转JPG并嵌入造点图片。

### video_workflow.json
多张图片转MP4视频并嵌入造点图片。

### batch_workflow.json
批量处理多组图片，每组生成对应的输出。
