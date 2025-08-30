# 项目结构

```
tt_img/
├── tt_img_enc_node.py      # 核心节点文件
├── __init__.py             # ComfyUI节点注册文件
├── extract_zip.py          # 文件提取工具
├── README.md               # 项目说明文档
├── requirements.txt        # Python依赖
├── setup.py               # 安装配置
├── pyproject.toml         # 项目配置
├── MANIFEST.in            # 打包配置
├── LICENSE                # 开源协议
├── .gitignore             # Git忽略文件
├── examples/              # 示例工作流
│   ├── basic_workflow.json
│   ├── video_workflow.json
│   └── README.md
├── output/                # 输出目录（自动创建）
└── temp/                  # 临时目录（自动创建）
```

## 核心文件说明

- **tt_img_enc_node.py**: 主要的ComfyUI节点实现
- **__init__.py**: 节点注册和映射配置
- **extract_zip.py**: 从图片中提取隐藏文件的工具
- **examples/**: 包含ComfyUI工作流示例

## 自动创建的目录

- **output/**: 存储输出文件
- **temp/**: 存储临时文件（处理过程中自动清理）

## 安装和部署

项目已优化为最小化结构，专注于核心功能：
- 移除了所有测试文件
- 移除了冗余的安装脚本
- 保留了必要的配置和示例文件
