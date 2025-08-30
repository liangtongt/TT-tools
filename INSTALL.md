# TT img enc - 安装说明

## 快速安装

### 方法1：Git Clone（推荐）
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tttools/tt-img-enc.git
```
然后重启 ComfyUI。

### 方法2：手动安装
1. 下载项目文件
2. 将整个项目文件夹复制到 `ComfyUI/custom_nodes/` 目录
3. 重启 ComfyUI

## 依赖说明

主要依赖：
- `opencv-python` - 图片处理和视频生成
- `Pillow` - 图片格式转换
- `numpy` - 数组操作
- `torch` - 张量操作（通常ComfyUI已包含）

如果遇到依赖问题：
```bash
pip install opencv-python Pillow numpy
```

## 使用方法

1. 重启 ComfyUI
2. 在节点列表中找到 "TT Tools" 分类
3. 拖拽 "TT img enc" 节点到工作区
4. 连接图片输入并运行工作流

## 功能说明

- **单张图片** → 自动转换为JPG并嵌入造点图片
- **多张图片** → 自动转换为MP4视频并嵌入造点图片
- **输出图片** → 下载后改后缀为.zip可直接解压

## 故障排除

### 节点不显示
- 检查 ComfyUI 控制台错误信息
- 确认项目在 `custom_nodes` 目录中

### 运行时错误
- 查看控制台输出
- 确认输入图片格式正确
- 检查临时目录权限

## 示例工作流

查看 `examples/` 目录中的工作流文件：
- `basic_workflow.json` - 基础使用
- `video_workflow.json` - 视频生成
