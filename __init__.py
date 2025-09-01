"""
TT img - ComfyUI 自定义节点套件
自动图片格式转换和文件隐写术（编码+解码）
支持密码保护的文件隐藏和提取
"""

from .tt_img_enc_node import TTImgEncNode
from .tt_img_dec_node import TTImgDecNode
from .tt_img_enc_pw_node import TTImgEncPwNode
from .tt_img_dec_pw_node import TTImgDecPwNode
from .tt_img_reverse_node import TTImgReverseNode

# 注册节点类
NODE_CLASS_MAPPINGS = {
    "TT_img_enc": TTImgEncNode,
    "TT_img_dec": TTImgDecNode,
    "TT_img_enc_pw": TTImgEncPwNode,
    "TT_img_dec_pw": TTImgDecPwNode,
    "TT_img_reverse": TTImgReverseNode
}

# 注册节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TT_img_enc": "TT img enc",
    "TT_img_dec": "TT img dec",
    "TT_img_enc_pw": "TT img enc pw",
    "TT_img_dec_pw": "TT img dec pw",
    "TT_img_reverse": "TT img reverse"
}

# 节点分类
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 节点功能说明
NODE_DESCRIPTIONS = {
    "TT_img_enc": "将图片/视频转换为造点图片，隐藏文件数据",
    "TT_img_dec": "从造点图片中提取隐藏的文件",
    "TT_img_enc_pw": "将图片/视频转换为带密码保护的造点图片",
    "TT_img_dec_pw": "通用解码节点：支持带密码保护和无密码的图片，自动检测类型",
    "TT_img_reverse": "图像反向节点：支持水平翻转、垂直翻转和同时翻转"
}
