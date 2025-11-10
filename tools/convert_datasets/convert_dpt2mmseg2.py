import torch
import os

# 读取权重文件
weight_path = r"G:\Code\workspace\Rein-train\checkpoints\depth_anything_v2_vitl_with_meta_modified.pth"
checkpoint = torch.load(weight_path)

# 删除meta字典
if 'meta' in checkpoint:
    del checkpoint['meta']

# 定义精确的键名映射关系
key_mapping = {
    'backbone.pretrained.cls_token': 'backbone.cls_token',
    'backbone.pretrained.pos_embed': 'backbone.pos_embed',
    'backbone.pretrained.mask_token': 'backbone.mask_token',
    'backbone.pretrained.patch_embed.proj.weight': 'backbone.patch_embed.proj.weight',
    'backbone.pretrained.patch_embed.proj.bias': 'backbone.patch_embed.proj.bias'
}

# 添加blocks的映射
for i in range(24):  # 0-23
    base = f'backbone.pretrained.blocks.{i}'
    new_base = f'backbone.blocks.0.{i}'
    for suffix in [
        'norm1.weight', 'norm1.bias',
        'attn.qkv.weight', 'attn.qkv.bias',
        'attn.proj.weight', 'attn.proj.bias',
        'ls1.gamma',
        'norm2.weight', 'norm2.bias',
        'mlp.fc1.weight', 'mlp.fc1.bias',
        'mlp.fc2.weight', 'mlp.fc2.bias',
        'ls2.gamma'
    ]:
        key_mapping[f'{base}.{suffix}'] = f'{new_base}.{suffix}'

# 添加最后的norm层映射
key_mapping['backbone.pretrained.norm.weight'] = 'backbone.norm.weight'
key_mapping['backbone.pretrained.norm.bias'] = 'backbone.norm.bias'

# 创建新的状态字典
new_state_dict = checkpoint.copy()  # 保留原始字典，只修改需要映射的键

# 应用映射
for old_key, new_key in key_mapping.items():
    if old_key in checkpoint:
        new_state_dict[new_key] = checkpoint[old_key]
        # 仅在新键不同于旧键时删除旧键
        if new_key != old_key:
            del new_state_dict[old_key]

# 保存修改后的权重文件
save_path = os.path.splitext(weight_path)[0] + '_renamed.pth'
torch.save(new_state_dict, save_path)

print(f"处理完成！新权重文件已保存至：{save_path}")

# 打印确认信息
print("\n键名映射总数:", len(key_mapping))
print("成功映射的键数:", len(new_state_dict))
print("未映射的键数:", len(set(checkpoint.keys()) - set(key_mapping.keys())))

# 可选：打印未映射的键
unmapped_keys = list(set(checkpoint.keys()) - set(key_mapping.keys()))
print("\n未映射的键：")
for key in unmapped_keys:
    print(key)