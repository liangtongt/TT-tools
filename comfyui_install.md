# ComfyUI 安装说明

## 快速安装

### 方法1：Git Clone（推荐）
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tttools/tt-img-enc.git
```
然后重启 ComfyUI。

### 方法2：自动安装脚本
```bash
# 在项目目录中运行
python install_comfyui.py
```

### 方法3：手动安装
1. 下载项目文件
2. 将整个项目文件夹复制到 `ComfyUI/custom_nodes/` 目录
3. 重启 ComfyUI

## 依赖安装

如果遇到依赖问题，手动安装：
```bash
pip install opencv-python Pillow numpy
```

## 验证安装

1. 重启 ComfyUI
2. 在节点列表中找到 "TT Tools" 分类
3. 应该能看到 "TT img enc" 节点

## 故障排除

### 节点不显示
- 检查 ComfyUI 控制台是否有错误信息
- 确认项目文件夹在 `custom_nodes` 目录中
- 检查 Python 依赖是否安装成功

### 运行时错误
- 查看 ComfyUI 控制台输出
- 确认输入图片格式正确
- 检查临时目录权限

## 项目结构

```
tt-img-enc/
├── __init__.py              # 节点注册文件
├── tt_img_enc_node.py      # 主节点代码
├── requirements.txt         # Python 依赖
├── examples/               # 示例工作流
│   ├── basic_workflow.json
│   └── video_workflow.json
├── README.md               # 详细说明
└── install_comfyui.py     # 安装脚本
```

## 更新节点

```bash
cd ComfyUI/custom_nodes/tt-img-enc
git pull origin main
```
然后重启 ComfyUI。
