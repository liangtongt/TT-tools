"""
TT img - ComfyUI 自定义节点套件
自动图片格式转换和文件隐写术（编码+解码）
"""

from .tt_img_enc_node import TTImgEncNode
from .tt_img_dec_node import TTImgDecNode

# 注册节点类
NODE_CLASS_MAPPINGS = {
    "TT_img_enc": TTImgEncNode,
    "TT_img_dec": TTImgDecNode
}

# 注册节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TT_img_enc": "TT img enc",
    "TT_img_dec": "TT img dec"
}

# 节点分类
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
