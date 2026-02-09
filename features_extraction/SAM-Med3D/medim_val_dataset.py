# -*- encoding: utf-8 -*-

import os.path as osp
from glob import glob

import medim
from tqdm import tqdm

from utils.infer_utils import validate_paired_img_gt

if __name__ == "__main__":
    ''' prepare the pre-trained model with local path or huggingface url '''
    # ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # or you can use a local path like:
    ckpt_path = "./ckpt/sam_med3d_turbo.pth"

    test_data_list = [
        dict(
            img_dir="./data/test/T1/harvard_oxford_atlas/imagesVal",
            gt_dir="./data/test/T1/harvard_oxford_atlas/labelsVal",
            out_dir="./data/test/T1/harvard_oxford_atlas/pred",
            ckpt_path="./ckpt/sam_med3d_turbo.pth",
        ),
    ]
    for test_data in test_data_list:
        model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=test_data["ckpt_path"])
        print("modelo cargado")
        gt_fname_list = sorted(glob(osp.join(test_data["gt_dir"], "*.nii.gz")))
        print(gt_fname_list)
        for gt_fname in tqdm(gt_fname_list):
            case_name = osp.basename(gt_fname).replace(".nii.gz", "")
            img_path = osp.join(test_data["img_dir"], f"{case_name}.nii.gz")
            print(img_path)
            gt_path = gt_fname
            out_path = osp.join(test_data["out_dir"], f"{case_name}.nii.gz")
            validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1,case_name=osp.basename(gt_path).replace(".nii.gz", ""))


