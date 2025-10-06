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
from .tt_img_color_reverse_node import TTImgColorReverseNode
from .tt_img_grayscale_node import TTImgGrayscaleNode
from .tt_img_brightness_contrast_node import TTImgBrightnessContrastNode
from .tt_img_rgb_adjust_node import TTImgRGBAdjustNode
from .tt_img_hsv_adjust_node import TTImgHSVAdjustNode
from .tt_img_lut_node import TTImgLUTNode
from .tt_img_enc_v2_node import TTImgEncV2Node

# 注册节点类
NODE_CLASS_MAPPINGS = {
    "TT_img_enc": TTImgEncNode,
    "TT_img_enc_v2": TTImgEncV2Node,
    "TT_img_dec": TTImgDecNode,
    "TT_img_enc_pw": TTImgEncPwNode,
    "TT_img_dec_pw": TTImgDecPwNode,
    "TT_img_reverse": TTImgReverseNode,
    "TT_img_color_reverse": TTImgColorReverseNode,
    "TT_img_grayscale": TTImgGrayscaleNode,
    "TT_img_brightness_contrast": TTImgBrightnessContrastNode,
    "TT_img_rgb_adjust": TTImgRGBAdjustNode,
    "TT_img_hsv_adjust": TTImgHSVAdjustNode,
    "TT_img_lut": TTImgLUTNode
}

# 注册节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TT_img_enc": "TT img enc",
    "TT_img_enc_v2": "TT img enc V2",
    "TT_img_dec": "TT img dec",
    "TT_img_enc_pw": "TT img enc pw",
    "TT_img_dec_pw": "TT img dec pw",
    "TT_img_reverse": "TT img reverse",
    "TT_img_color_reverse": "TT img color reverse",
    "TT_img_grayscale": "TT img grayscale",
    "TT_img_brightness_contrast": "TT img brightness contrast",
    "TT_img_rgb_adjust": "TT img RGB adjust",
    "TT_img_hsv_adjust": "TT img HSV adjust",
    "TT_img_lut": "TT img LUT"
}

# 节点分类
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# 节点功能说明
NODE_DESCRIPTIONS = {
    "TT_img_enc": "将图片/视频转换为造点图片，隐藏文件数据",
    "TT_img_enc_v2": "V2：支持每通道多位LSB并可跳过上下20%，更高容量",
    "TT_img_dec": "从造点图片中提取隐藏的文件",
    "TT_img_enc_pw": "将图片/视频转换为带密码保护的造点图片",
    "TT_img_dec_pw": "通用解码节点：支持带密码保护和无密码的图片，自动检测类型",
    "TT_img_reverse": "图像反向节点：支持水平翻转、垂直翻转和同时翻转",
    "TT_img_color_reverse": "图像颜色反向节点：支持完全反转、仅RGB反转、保留透明度反转",
    "TT_img_grayscale": "图像转灰度节点：支持多种灰度转换方法（亮度加权、平均值、单通道等）",
    "TT_img_brightness_contrast": "图像亮度和对比度调节节点：支持亮度(-1.0到1.0)和对比度(0.0到3.0)调节",
    "TT_img_rgb_adjust": "RGB三色相调节节点：支持独立调节红、绿、蓝三个通道(-1.0到1.0)",
    "TT_img_hsv_adjust": "HSV色彩调节节点：支持色相(-180°到180°)、饱和度(0.0到3.0)、明度(0.0到3.0)调节",
    "TT_img_lut": "LUT色彩查找表节点：支持.cube格式LUT文件，电影级调色效果"
}
