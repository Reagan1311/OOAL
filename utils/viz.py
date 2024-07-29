import os
import torch
import numpy as np
from PIL import Image
from utils.util import overlay_mask
import matplotlib.pyplot as plt
import cv2


def viz_pred_test(args, image, ego_pred, GT_mask, aff_list, aff_label, img_name, epoch=None):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    gt = Image.fromarray(GT_mask)
    gt_result = overlay_mask(img, gt, alpha=0.5)
    aff_str = aff_list[aff_label.item()]

    os.makedirs(os.path.join(args.save_path, 'viz_gray'), exist_ok=True)
    gray_name = os.path.join(args.save_path, 'viz_gray', img_name + '.jpg')
    cv2.imwrite(gray_name, ego_pred * 255)

    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(img, ego_pred, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()

    ax[0].imshow(ego_pred)
    ax[0].set_title(aff_str)
    ax[1].imshow(gt_result)
    ax[1].set_title('GT')

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "iter" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()
