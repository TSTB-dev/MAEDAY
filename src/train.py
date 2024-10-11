import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse

import random 
import torch
from torch.utils.data import DataLoader
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

from src.datasets import build_dataset, build_transforms, TrainDataLoader
from src.mask import RandomMaskCollator
from src.models import build_model
from src.util import AverageMeter

def parse_args():
    parser = argparse.ArgumentParser(description="MAEDAY [Training]")
    parser.add_argument("--data_root", type=str, default="/home/sakai/projects/MAEDAY/data/mvtec_ad")
    parser.add_argument("--class_name", type=str, default="bottle")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)
    
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--weights_path", type=str, default=None)  # path to the pre-trained weights

    parser.add_argument("--num_normal_samples", type=int, default=-1)  # -1: use all normal samples
    parser.add_argument("--transform", type=str, default="default")
    parser.add_argument("--training_scheme", type=str, default="lora")  # lora or finetune
    parser.add_argument("--finetune_mode", type=str, default="decoder")  # full, decoder
    parser.add_argument("--lora_mode", type=str, default="kv")  # kv, all
    parser.add_argument("--lora_dim", type=int, default=32)  
    parser.add_argument("--apply_loss_on_vis", action="store_true", default=False)  # apply loss on unmasked and masked patches
    parser.add_argument("--num_iters", type=int, default=-1)
    parser.add_argument("--num_epochs", type=int, default=-1)  
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="sgd")  # sgd, adamw
    parser.add_argument("--scheduler", type=str, default=None)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="weights")

    return parser.parse_args()
    
def get_trainable_params(model, args):
    if args.training_scheme == "finetune":
        if args.finetune_mode == "full":
            params = model.parameters()
        elif args.finetune_mode == "decoder":
            params = []
            for name, param in model.named_parameters():
                if "decoder_embed" in name or "decoder_blocks" in name or "decoder_norm" in name or "decoder_pred" in name:
                    params.append(param)
                else:
                    param.requires_grad = False

    return params

def main(args):
    assert args.weights_path is not None, "weights_path should be specified"
    assert args.split in ["train", "val", "test"], "Invalid split"
    assert args.num_iters * args.num_epochs < 0, "num_iters and num_epochs should not be specified simultaneously"
    
    logger.info(args)
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Build dataset
    transform = build_transforms(args)
    dataset = build_dataset(args, transform)
    dataset_name = args.data_root.split("/")[-1]
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Number of samples: {len(dataset)}")
    
    # Load pre-trained weights
    model = build_model(args)
    model.load_state_dict(torch.load(args.weights_path, weights_only=True)["model"])
    patch_size = model.patch_embed.patch_size[0]
    
    # Create mask collator
    mask_collator = RandomMaskCollator(ratio=0.75, input_size=args.img_size, patch_size=patch_size)
    
    # Create dataloader
    dataloader = TrainDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=mask_collator
    )
    
    # Optimizer
    params = get_trainable_params(model, args)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")
    
    # Scheduler
    if args.scheduler is not None:
        if args.scheduler == "cosine":
            T_max = args.num_iters if args.num_iters > 0 else args.num_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        else:
            raise ValueError(f"Invalid scheduler: {args.scheduler}")
    else:
        scheduler = None
    
    # transfer model to device
    model.to(args.device)
    
    # Train
    i = 0
    dataloader_iter = iter(dataloader)
    loss_meter = AverageMeter()
    model.train()
    total_iters = args.num_iters if args.num_iters > 0 else args.num_epochs * len(dataloader)
    
    logger.info("Training started🚀") 
    while i < total_iters:
        try:
            batch, mask_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch, mask_batch = next(dataloader_iter)

        images = batch["samples"].to(args.device)
        mask_batch = mask_batch.to(args.device)
        
        # Forward
        loss, pred = model(images, mask_batch, args.apply_loss_on_vis)
        loss_meter.update(loss.item(), images.size(0))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Scheduler
        if scheduler is not None:
            scheduler.step()
        
        assert not torch.isnan(loss).any(), "Loss is NaN🐥"
        
        # Log
        if i % args.log_interval == 0:
            logger.info(f"Iter: {i}/{total_iters}, Loss: {loss_meter.avg}")
        
        i += 1
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_type = args.weights_path.split("_")[3].split(".")[0]
    save_path = os.path.join(args.save_dir, f"mae_{args.training_scheme}_vit_{model_type}_{args.class_name}_{dataset_name}.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved at {save_path}")
    
    logger.info("Training finished✅")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)