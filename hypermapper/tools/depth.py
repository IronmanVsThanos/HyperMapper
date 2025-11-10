# import torch
# import torch.nn.functional as F
# import os
# import os.path as osp
# import tqdm
# import cv2
# import numpy as np
# from mmseg.registry import MODELS
# from mmengine import Config, mkdir_or_exist
# from mmengine.runner import load_checkpoint
# from mmseg.utils import get_classes, get_palette
# from PIL import Image
# import hypermapper
#
# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser(description="Depth Anything V2 Visualization")
#     parser.add_argument("--config",default=r"G:\Code\workspace\Rein-train\configs\_base_\models\depth_anything_v2.py", help="Path to the configuration file.")
#     parser.add_argument("--checkpoint", default=r"G:\Code\workspace\Rein-train\checkpoints\depth_anything_v2_vitl.pth", help="Path to the checkpoint file.")
#     parser.add_argument("--images", default=r"G:\Code\workspace\Rein-train\data\img\aachen_000003_000019.jpg",help="Directory or file path of images to be processed.")
#     parser.add_argument("--suffix", default=".jpg", help="File suffix to filter images.")
#     parser.add_argument("--save_dir", default="work_dirs/depth", help="Directory to save output images.")
#     parser.add_argument("--input-size", type=int, default=518, help="Input image size for preprocessing.")
#     parser.add_argument("--device", default="cuda", help="Device to use for computation.")
#     return parser.parse_args()
#
#
# def init_model(config, checkpoint, device='cuda'):
#     """Initialize model with checkpoint"""
#     cfg = Config.fromfile(config)
#
#     # 确保模型配置正确
#     if not hasattr(cfg, 'model'):
#         raise ValueError("Config must contain model configuration")
#
#     # 构建模型
#     model = MODELS.build(cfg.model)
#
#     # 加载权重
#     load_checkpoint(model, checkpoint, map_location=device)
#
#     model.to(device)
#     model.eval()
#     return model
#
#
# def preprocess_image(image_path, input_size=518):
#     """图像预处理"""
#     raw_image = cv2.imread(image_path)
#     image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
#
#     # 保持长宽比的缩放
#     h, w = image.shape[:2]
#     scale = input_size / max(h, w)
#     new_h, new_w = int(h * scale), int(w * scale)
#
#     image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#
#     # 归一化
#     image = image.astype(np.float32) / 255.0
#     image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
#
#     # 转换为张量
#     image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
#
#     return image, (h, w)
#
#
# def colorize_depth(depth):
#     """深度图着色"""
#     depth = (depth - depth.min()) / (depth.max() - depth.min())
#     depth_colored = cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
#     return depth_colored
#
#
# def main():
#     args = parse_args()
#
#     # 创建保存目录
#     mkdir_or_exist(args.save_dir)
#
#     # 初始化模型
#     model = init_model(args.config, args.checkpoint, args.device)
#
#     # 收集图像
#     images = []
#     if osp.isfile(args.images):
#         images.append(args.images)
#     elif osp.isdir(args.images):
#         for img in os.listdir(args.images):
#             if img.endswith(args.suffix):
#                 images.append(osp.join(args.images, img))
#
#     # 处理图像
#     for img_path in tqdm.tqdm(images):
#         # 预处理
#         image, (orig_h, orig_w) = preprocess_image(img_path, args.input_size)
#         image = image.to(args.device)
#
#         # 推理
#         with torch.no_grad():
#             depth = model(image)['pred_depth']
#             depth = F.interpolate(depth, (orig_h, orig_w), mode="bilinear", align_corners=True)[0, 0]
#             depth = depth.cpu().numpy()
#
#         # 着色和保存
#         depth_colored = colorize_depth(depth)
#
#         # 创建并保存并排图像
#         orig_image = cv2.imread(img_path)
#         vis_image = np.hstack([orig_image, depth_colored])
#
#         save_path = osp.join(args.save_dir, osp.basename(img_path))
#         cv2.imwrite(save_path, vis_image)
#
#     print(f"处理完成，结果保存在 {args.save_dir}")
#
#
# if __name__ == "__main__":
#     main()
from email.policy import strict

import matplotlib
import torch
import torch.nn.functional as F
from mmseg.apis import inference_model
from mmengine.runner import Runner, load_checkpoint
from mmengine import Config
from mmengine.registry import MODELS
from mmseg.utils import register_all_modules
import mmcv
import cv2
import numpy as np
from torchvision.transforms import Compose
import  hypermapper
from util.transform import resize_image, normalize_image, prepare_for_net


def init_model_without_meta(config, checkpoint=None, device='cuda:0', is_strict=False):
    """Initialize a model without meta information."""
    register_all_modules()

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    config.model.pretrained = None
    model = MODELS.build(config.model)

    # 直接使用 load_checkpoint 加载权重
    if checkpoint is not None:
        # 使用 load_checkpoint 替代手动加载
        checkpoint = load_checkpoint(model, checkpoint, map_location=device, strict=is_strict)

        # 如果需要，可以打印权重加载信息
        print(f"Loaded checkpoint from {checkpoint}")

    model.cfg = config
    model.to(device)
    model.eval()
    return model


def add_meta_to_checkpoint(checkpoint_path, save_path=None):
    """Add meta information to checkpoint file."""
    if save_path is None:
        save_path = checkpoint_path.replace('.pth', '_with_meta.pth')

    checkpoint = torch.load(checkpoint_path)

    if 'meta' not in checkpoint:
        checkpoint['meta'] = {
            'dataset_meta': {
                'classes': ['depth'],
                'palette': None
            }
        }
    torch.save(checkpoint, save_path)
    return save_path


def image2tensor(raw_image, input_size=518):
    """
    Preprocess raw_image into a tensor for model input.

    Args:
        raw_image (np.ndarray): Input raw image in BGR format.
        input_size (int): Target size for preprocessing.

    Returns:
        torch.Tensor: Preprocessed image tensor.
        tuple: Original image height and width.
    """
    # Step 1: Get original image dimensions
    h, w = raw_image.shape[:2]

    # Step 2: Convert image to RGB and normalize to [0, 1]
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    # Step 3: Resize image
    image = resize_image(
        image=image,
        width=input_size,
        height=input_size,
        keep_aspect_ratio=False,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        interpolation=cv2.INTER_CUBIC
    )

    # Step 4: Normalize image
    image = normalize_image(
        image=image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Step 5: Prepare for network
    processed_data = prepare_for_net(image=image)
    image = processed_data["image"]

    # Step 6: Convert to PyTorch tensor and add batch dimension
    image = torch.from_numpy(image).unsqueeze(0)

    # Step 7: Move tensor to the appropriate device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    image = image.to(DEVICE)

    return image, (h, w)

def main():
    cfg = Config.fromfile(r'G:\Code\workspace\Rein-train\configs\_base_\models\depth_anything_v2.py')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = r'G:\Code\workspace\Rein-train\checkpoints\depth_anything_v2_vitl_with_meta_modified.pth'
    model = init_model_without_meta(cfg, checkpoint_path, device=device, is_strict = False)
    # 加载并预处理图像
    img_path = r'G:\Code\workspace\Rein-train\data\img\aachen_000003_000019.jpg'  # 替换为实际图像路径
    raw_image = cv2.imread(img_path)
    image, (h, w) = image2tensor(raw_image,518)
    # 进行推理
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    with torch.no_grad():
        depth = model(image)['pred_depth']
        #1,1,252,518
        depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        depth = depth.cpu().numpy()

    # 处理深度图输出
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    # 保存结果
    cv2.imwrite('depth_result.png', depth)

if __name__ == '__main__':
    main()