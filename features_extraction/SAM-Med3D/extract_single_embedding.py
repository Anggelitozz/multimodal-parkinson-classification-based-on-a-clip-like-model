# -*- encoding: utf-8 -*-

import medim

import torch
from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics

if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    ckpt_path = "./ckpt/sam_med3d_turbo.pth"
    import torch
    print("CUDA available:", torch.cuda.is_available())

    # or you can use a local path like:
    model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Model device:", next(model.parameters()).device)
    embeddings = {}

    def get_embeddings_hook(module, input, output):
        embeddings['image'] = output.detach().cpu()

    # suponer que sam_model.image_encoder es el encoder 3D
    model.image_encoder.register_forward_hook(get_embeddings_hook)

    ''' 2. read and pre-process your input data '''
    img_path = "test_data/amos_val_toy_data/imagesVal/PPMI_3392_MR_T1_resampled_brain_extracted_bias_corrected_rigid_registered.nii.gz"
    # img_path = "test_data/amos_val_toy_data/imagesVal/amos_0013.nii.gz"
    gt_path = "test_data/amos_val_toy_data/labelsVal/PPMI_3392_MR_T1_resampled_brain_extracted_bias_corrected_rigid_registered.nii.gz"
    #gt_path= "test_data/amos_val_toy_data/labelsVal/amos_0013.nii.gz"
    out_path = "test_data/amos_val_toy_data/pred/3392.nii.gz"
    #out_path = "test_data/amos_val_toy_data/pred/amos_0013.nii.gz"
    ''' 3. infer with the pre-trained SAM-Med3D model '''
    print("Validation start! plz wait for some times.")

    validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
    print("Validation finish! plz check your prediction.")

    torch.save(embeddings['image'], "3392.pt")

    ''' 4. compute the metrics of your prediction with the ground truth '''
    metrics = compute_metrics(
        gt_path=gt_path,
        pred_path=out_path,
        metrics=['dice'],
        classes=None,
    )
    print_computed_metrics(metrics)
