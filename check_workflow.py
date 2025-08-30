#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查ComfyUI工作流配置
"""

import json
import os

def check_workflow():
    """检查工作流配置"""
    
    print("=== 🔍 ComfyUI工作流检查 ===\n")
    
    # 检查工作流文件
    workflow_file = "example_workflow.json"
    if os.path.exists(workflow_file):
        print(f"📋 找到工作流文件: {workflow_file}")
        
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            print("\n🔍 工作流分析:")
            
            # 检查节点
            if 'nodes' in workflow:
                nodes = workflow['nodes']
                print(f"  总节点数: {len(nodes)}")
                
                tt_img_nodes = []
                save_image_nodes = []
                
                for node_data in nodes:
                    node_id = node_data.get('id', '')
                    node_type = node_data.get('type', '')
                    if node_type == 'TTImg':
                        tt_img_nodes.append(node_id)
                    elif node_type == 'SaveImage':
                        save_image_nodes.append(node_id)
                
                print(f"  TT img节点: {tt_img_nodes}")
                print(f"  SaveImage节点: {save_image_nodes}")
                
                # 检查连接
                if 'last_link_id' in workflow:
                    print(f"\n🔗 连接分析:")
                    
                    # 从节点数据中提取连接信息
                    for node_data in nodes:
                        node_id = node_data.get('id', '')
                        node_type = node_data.get('type', '')
                        
                        if node_type == 'TTImg':
                            print(f"  TT img节点 {node_id} 的输出连接:")
                            outputs = node_data.get('outputs', {})
                            for output_name, output_data in outputs.items():
                                links = output_data.get('links', [])
                                print(f"    {output_name}: {links}")
                        
                        elif node_type == 'SaveImage':
                            print(f"  SaveImage节点 {node_id} 的输入连接:")
                            inputs = node_data.get('inputs', {})
                            for input_name, input_data in inputs.items():
                                link = input_data.get('link', None)
                                print(f"    {input_name}: {link}")
                
                # 检查是否有问题
                if not tt_img_nodes:
                    print("\n❌ 问题: 工作流中没有TT img节点")
                elif not save_image_nodes:
                    print("\n❌ 问题: 工作流中没有SaveImage节点")
                elif len(tt_img_nodes) > 1:
                    print("\n⚠️  警告: 工作流中有多个TT img节点")
                elif len(save_image_nodes) > 1:
                    print("\n⚠️  警告: 工作流中有多个SaveImage节点")
                else:
                    print("\n✅ 工作流配置看起来正常")
                    
        except Exception as e:
            print(f"❌ 无法解析工作流文件: {e}")
    else:
        print(f"⚠️  未找到工作流文件: {workflow_file}")
    
    print(f"\n📋 建议的检查步骤:")
    print(f"  1. 确保ComfyUI已重启")
    print(f"  2. 检查工作流中是否有其他节点影响TT img输出")
    print(f"  3. 尝试简化工作流，只保留TT img和SaveImage节点")
    print(f"  4. 检查ComfyUI版本兼容性")
    print(f"  5. 检查其他自定义节点是否有冲突")

if __name__ == "__main__":
    check_workflow()
