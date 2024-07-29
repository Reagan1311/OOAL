import cv2
import os
import random
import torch
import numpy as np
from PIL import Image
from matplotlib import cm


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_gt(args):
    assert args.divide in ["Seen", "Unseen"], "The divide argument should be Seen or Unseen"
    files = os.listdir(args.mask_root)
    dict_1 = {}
    for file in files:
        file_path = os.path.join(args.mask_root, file)
        objs = os.listdir(file_path)
        for obj in objs:
            obj_path = os.path.join(file_path, obj)
            images = os.listdir(obj_path)
            for img in images:
                img_path = os.path.join(obj_path, img)
                mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                key = file + "_" + obj + "_" + img
                dict_1[key] = mask

    torch.save(dict_1, args.divide + "_gt.t7")


def normalize_map(atten_map, crop_size):
    atten_map = cv2.resize(atten_map, dsize=(crop_size, crop_size))
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)
    return atten_norm


def get_optimizer(model, args):
    lr = args.lr
    weight_list = []
    for name, value in model.named_parameters():
        if value.requires_grad:
            weight_list.append(value)

    optimizer = torch.optim.SGD([{'params': weight_list,
                                  'lr': lr}],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)
    return optimizer, scheduler


def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img
