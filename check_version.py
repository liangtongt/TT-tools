#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查TT img节点版本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_version():
    """检查节点版本"""
    
    try:
        from image_sequence_compressor import TTImg
        
        print("=== 🔍 TT img 节点版本检查 ===\n")
        
        # 创建节点实例
        tt_img = TTImg()
        
        print(f"📋 节点信息:")
        print(f"  类名: {TTImg.__name__}")
        print(f"  版本: {TTImg.VERSION}")
        print(f"  构建日期: {TTImg.BUILD_DATE}")
        print(f"  预览功能: {'✅ 已启用' if TTImg.OUTPUT_NODE else '❌ 未启用'}")
        
        # 检查IS_CHANGED方法
        change_hash = TTImg.IS_CHANGED()
        print(f"  变更哈希: {change_hash}")
        
        # 检查代码文件修改时间
        import time
        file_path = "image_sequence_compressor.py"
        if os.path.exists(file_path):
            mod_time = os.path.getmtime(file_path)
            mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
            print(f"  代码文件修改时间: {mod_time_str}")
        
        print(f"\n🎯 版本检查完成!")
        print(f"  如果版本号显示为 {TTImg.VERSION}，说明节点已更新")
        print(f"  如果ComfyUI中仍然显示旧版本，需要重启ComfyUI")
        
        return True
        
    except Exception as e:
        print(f"❌ 版本检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_version()
    
    if success:
        print("\n✅ 版本检查成功！")
    else:
        print("\n💥 版本检查失败！")
