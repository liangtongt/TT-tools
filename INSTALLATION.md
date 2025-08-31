# ComfyUI 插件安装说明

## 安装步骤

### 1. 下载插件
将整个 `tt_img` 文件夹复制到ComfyUI的 `custom_nodes` 目录中。

### 2. 目录结构
```
ComfyUI/
├── custom_nodes/
│   └── tt_img/
│       ├── tt_img_enc_node.py
│       ├── tt_img_dec_node.py
│       ├── extract_zip.py
│       ├── __init__.py
│       └── ...
└── output/
    └── (提取的文件将保存在这里)
```

### 3. 重启ComfyUI
安装完成后，重启ComfyUI以加载新插件。

### 4. 验证安装
在ComfyUI中，您应该能在 `TT Tools` 分类下看到：
- **TT img enc**: 编码节点
- **TT img dec**: 解码节点

## 功能特性

### 编码节点 (TT img enc)
- 将文件隐藏到图片中
- 支持MP4、JPG等多种格式
- 自动预留水印区域（左上角50像素）
- 输出编码后的图片

### 解码节点 (TT img dec)
- 从图片中提取隐藏文件
- 自动保存到ComfyUI默认output目录
- 兼容带水印的图片
- 支持水印区域自动跳过

### 独立解码工具 (extract_zip.py)
- 命令行工具，无需ComfyUI环境
- 支持水印区域兼容性
- 使用方法：`python extract_zip.py <图片路径> [输出路径]`

## 使用示例

### 基本工作流
1. 添加图片到 **TT img enc** 节点
2. 设置参数（FPS、质量等）
3. 运行编码，获得隐藏文件的图片
4. 将编码图片连接到 **TT img dec** 节点
5. 设置输出文件名
6. 运行解码，文件自动保存到ComfyUI output目录

### 水印兼容性
- 编码时自动预留左上角50像素区域
- 解码时自动跳过水印区域
- 确保平台添加水印不影响数据提取

## 输出文件位置

提取的文件会自动保存到ComfyUI的默认output目录：
- 通常位于 `ComfyUI/output/` 目录
- 与其他ComfyUI生成的图片保存在同一位置
- 可以通过ComfyUI界面直接查看和下载

## 依赖要求

确保已安装以下Python包：
```bash
pip install numpy pillow opencv-python torch
```

## 故障排除

### 插件未显示
1. 检查文件是否放在正确的 `custom_nodes` 目录
2. 重启ComfyUI
3. 检查控制台是否有错误信息

### 文件保存失败
1. 检查ComfyUI output目录权限
2. 查看控制台错误信息
3. 确认ComfyUI有写入权限

### 提取失败
1. 确认图片由TT img enc节点生成
2. 检查图片格式（需要3通道RGB）
3. 查看详细错误信息

## 技术支持

- 教程：https://b23.tv/RbvaMeW
- B站：我是小斯呀
- 查看控制台输出获取详细错误信息
