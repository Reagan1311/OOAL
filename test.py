import os
import argparse
from tqdm import tqdm

import cv2
import torch
import numpy as np
from models.ooal import Net as model

from utils.viz import viz_pred_test
from utils.util import set_seed, process_gt, normalize_map
from utils.evaluation import cal_kl, cal_sim, cal_nss

parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='./dataset/agd20k')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--save_path', type=str, default='./save_preds')
##  image
parser.add_argument('--divide', type=str, default='Seen')
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=256)
#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)

args = parser.parse_args()
args.mask_root = os.path.join(args.data_root, args.divide, "testset", "GT")

if args.viz:
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

if __name__ == '__main__':
    set_seed(seed=0)

    from data.agd20k_ego import TestData, SEEN_AFF, UNSEEN_AFF

    args.class_names = SEEN_AFF if args.divide == 'Seen' else UNSEEN_AFF

    testset = TestData(data_root=args.data_root, divide=args.divide, crop_size=args.crop_size)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model(args, 768, 512).cuda()

    KLs, SIM, NSS = [], [], []
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    state_dict = torch.load(args.model_file)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)

    GT_path = args.divide + "_gt.t7"
    if not os.path.exists(GT_path):
        process_gt(args)
    GT_masks = torch.load(GT_path)

    for step, (image, gt_aff, object, mask_path) in enumerate(tqdm(TestLoader)):
        ego_pred = model(image.cuda(), gt_aff=gt_aff)
        ego_pred = np.array(ego_pred.squeeze().data.cpu())
        ego_pred = normalize_map(ego_pred, args.crop_size)

        names = mask_path[0].split("/")
        key = names[-3] + "_" + names[-2] + "_" + names[-1]

        if key in GT_masks.keys():
            GT_mask = GT_masks[key]
            GT_mask = GT_mask / 255.0

            GT_mask = cv2.resize(GT_mask, (args.crop_size, args.crop_size))

            kld, sim, nss = cal_kl(ego_pred, GT_mask), cal_sim(ego_pred, GT_mask), cal_nss(ego_pred, GT_mask)

            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)

        if args.viz:
            img_name = key.split(".")[0]
            viz_pred_test(args, image, ego_pred, GT_mask, args.class_names, gt_aff, img_name)

    mKLD, mSIM, mNSS = sum(KLs) / len(KLs), sum(SIM) / len(SIM), sum(NSS) / len(NSS)

    print(
        "mKLD = " + str(round(mKLD, 3)) + " mSIM = " + str(round(mSIM, 3)) + " mNSS = " + str(round(mNSS, 3))
    )
