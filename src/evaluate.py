import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pprint import pprint
import argparse
import random

from sklearn.metrics import roc_auc_score, roc_curve, auc
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from src.datasets import build_dataset, build_transforms, EvalDataLoader
from src.mask import RandomMaskCollator
from src.models import build_model, load_checkpoint
from src.util import AverageMeter, pidx_to_pmask, gaussian_kernel, pidx_to_imask, \
    visualize_on_masked_area, save_tensor_image, patchify

def parse_args():
    parser = argparse.ArgumentParser(description="MAEDAY [Evaluation]")
    parser.add_argument("--data_root", type=str, default="/home/sakai/projects/MAEDAY/data/mvtec_ad")
    parser.add_argument("--class_name", type=str, default="bottle")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_masks", type=int, default=32)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--weights_path", type=str, default=None)  # path to the pre-trained weights
    parser.add_argument("--gaussian_filter", action="store_true", default=False)  # apply Gaussian filter to the error map
    parser.add_argument("--gaussian_sigma", type=float, default=1.4)
    parser.add_argument("--gaussian_ksize", type=int, default=7)
    parser.add_argument("--err_on_mask", action="store_true", default=False)  # calculate error only on masked patches
    parser.add_argument("--transform", type=str, default="default")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    return args

def calculate_mask_coverage(mask_batch, h, w):
    """Calculate mask coverage. 

    Args:
        mask_batch (tensor): Indices of masked patches. Shape: (B, N)
        h (int): Height of the feature map
        w (int): Width of the feature map
    Returns:
        mask_coverage (float): Mask coverage
    """
    mask = pidx_to_pmask(mask_batch, h, w)  # (B, H, W)
    mask_or = torch.any(mask, dim=0).float()  # (H, W)
    mask_coverage = torch.mean(mask_or)  # scalar
    return mask_coverage  

def gaussian_filter(err_map, sigma=1.4, ksize=7):
    """Apply Gaussian filter to the error map

    Args:
        err_map (tensor): Error map. Shape: (B, H, W)
        sigma (float, optional): Standard deviation of the Gaussian filter. Defaults to 1.4.
        ksize (int, optional): Kernel size of the Gaussian filter. Defaults to 7.
    Returns:
        err_map (tensor): Error map after applying Gaussian filter, Shape: (B, H, W)
    """
    err_map = err_map.detach().cpu()
    kernel = gaussian_kernel(ksize, sigma) 
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(err_map.device)  # (1, 1, ksize, ksize)
    padding = ksize // 2
    err_map = F.pad(err_map, (padding, padding, padding, padding), mode='reflect')
    err_map = F.conv2d(err_map.unsqueeze(1), kernel, padding=0).squeeze(1)
    return err_map

def compute_err_map(images, pred, mask_batch, patch_size, h, w, err_on_mask=False, normalized=False):
    """Compute error map
    Args:
        images (tensor): Original images. Shape: (B, C, H, W)
        pred (tensor): Predicted images. Shape: (B, C, H, W)
        mask_batch (tensor): Indices of masked patches. Shape: (B, N)
        h (int): Height of the feature map
        w (int): Width of the feature map
        patch_size (int): Patch size
        err_on_mask (bool, optional): Calculate error only on masked patches. Defaults to False.
        normalized (bool, optional): Normalize the error map. Defaults to False.
    Returns:
        err_map (tensor): Error map. Shape: (B, H, W) or (B, L)
    """
    H, W = images.shape[-2:]
    if normalized:
        images = patchify(images, patch_size)  # (B, L, patch_size**2 * 3)
        pred = patchify(pred, patch_size)  # (B, L, patch_size**2 * 3)
        mean = images.mean(dim=2, keepdim=True)  # (B, L, 1)
        var = images.var(dim=2, keepdim=True)  # (B, L, 1)
        images = (images - mean) / (var + 1e-6) ** 0.5
        err_map = torch.sum((images - pred) ** 2, dim=-1)  # (B, L)
        err_map = err_map.view(-1, h, w)  # (B, h, w)
        err_map = F.interpolate(err_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)  # (B, H, W)
    else:
        err_map = torch.sum((images - pred) ** 2, dim=1)  # (B, H, W)

    if err_on_mask:
        imasks = pidx_to_imask(mask_batch, h, w, patch_size).squeeze(1)  # (B, H, W)
        err_map = err_map * imasks.to(device=err_map.device)  
        
    return err_map

def main(args):
    assert args.weights_path is not None, "weights_path should be specified"
    assert args.num_masks > 0, "num_masks should be greater than 0"
    
    # logger.info(args)
    
    # Set seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Build model
    model = build_model(args)
    load_checkpoint(model, args.weights_path)
    patch_size = model.patch_embed.patch_size[0]
    h, w = args.img_size // patch_size, args.img_size // patch_size
    if "full" in args.weights_path:
        normalized = True  # If the model is trained with predicting normalized pixels.
    else:
        normalized = False
    
    # Build dataset
    args.split = "test"
    transform = build_transforms(args)
    dataset = build_dataset(args, transform, eval=True)
    mask_collator = RandomMaskCollator(args.mask_ratio, args.img_size, patch_size)
    dataloader = EvalDataLoader(dataset, args.num_masks, collate_fn=mask_collator)
    logger.info(f"Number of images in the dataset: {len(dataset)}")
    
    # Evaluation
    model.eval()
    model.to(args.device)
    
    logger.info("Evaluation started🚀")
    loss_meter = AverageMeter()
    mask_coverage_meter = AverageMeter()
    
    err_maps = []
    filenames = []
    clsnames = []
    labels = []
    anom_types = []
    for i, (batch, mask_batch) in enumerate(dataloader):   
        images = batch["samples"].to(args.device)  # (B, C, H, W)
        labels.append(batch["labels"][0].item())
        anom_types.append(batch["anom_type"][0])
        filenames.append(batch["filenames"][0])
        clsnames.append(batch["clsnames"][0])
        mask_coverage = calculate_mask_coverage(mask_batch, h, w)
        mask_coverage_meter.update(mask_coverage, 1)
        
        mask_batch = mask_batch.to(args.device)
        
        with torch.no_grad():
            loss, pred = model(images, mask_batch)   # (B, C, H, W)
            loss_meter.update(loss.item(), images.size(0))
            err_map = compute_err_map(images, pred, mask_batch, patch_size, h, w, args.err_on_mask, normalized=normalized)
            
            if args.gaussian_filter:
                err_map = gaussian_filter(err_map, args.gaussian_sigma, args.gaussian_ksize)
            err_map = torch.mean(err_map, dim=0)  # (H, W)
            err_maps.append(err_map)    
        
        if i % args.log_interval == 0:
            logger.info(f"Iter: {i}/{len(dataloader)}, Loss: {loss_meter.avg:.4f}, Mask coverage: {mask_coverage_meter.avg:.4f}")
            
    logger.info(f"Loss: {loss_meter.avg:.4f}, Mask coverage: {mask_coverage_meter.avg:.4f}")
    
    # Calculate the auROC score
    global_err_scores = [torch.max(err_map) for err_map in err_maps]
    global_err_scores = torch.stack(global_err_scores).cpu().numpy()
    
    auc = roc_auc_score(labels, global_err_scores)
    logger.info(f"\n-------auROC: {auc:.4f} on {args.class_name}------------")
    
    # Calculate the auROC score for each anomaly type
    unique_anom_types = list(sorted(set(anom_types)))
    normal_indices = [i for i, x in enumerate(anom_types) if x == "good"]
    for anom_type in unique_anom_types: 
        if anom_type == "good":
            continue
        anom_indices = [i for i, x in enumerate(anom_types) if x == anom_type]
        normal_scores = global_err_scores[normal_indices]
        anom_scores = global_err_scores[anom_indices]
        scores = np.concatenate([normal_scores, anom_scores])
        labels = [0] * len(normal_scores) + [1] * len(anom_scores)
        auc = roc_auc_score(labels, scores)
        logger.info(f"auROC: {auc:.4f} on {anom_type}")
    print("-------------------")
    
    logger.info("Evaluation finished🎉")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)