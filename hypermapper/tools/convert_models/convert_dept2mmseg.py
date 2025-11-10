import torch

def remove_keys():
    # 加载原始权重文件
    checkpoint = torch.load('/data/DL/code/Rein-train-depth/checkpoints/depth_dinov2_patc16_880_convert.pth', map_location='cpu')
    
    # 创建新的state_dict
    new_state_dict = {}
    
    # 遍历并修改键名
    for k, v in checkpoint.items():
        if k.startswith('backbone.'):
            # 删除'backbone.'前缀
            new_key = k[9:]  # 'backbone.'有9个字符
            new_state_dict[new_key] = v
        else:
            # 其他键保持不变
            new_state_dict[k] = v
    
    # 保存新的权重文件
    torch.save(new_state_dict, '/data/DL/code/Rein-train-depth/checkpoints/depth_dinov2_patc16_880_convert_new.pth')
    
    # 打印示例修改（可选）
    print("Key modification examples:")
    for old_key, new_key in zip(list(checkpoint.keys())[:5], list(new_state_dict.keys())[:5]):
        print(f"Old: {old_key} -> New: {new_key}")

if __name__ == '__main__':
    remove_keys()