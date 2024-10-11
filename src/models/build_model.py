import torch
from .models_mae import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14

def build_model(args):
    if "vit_base" in args.weights_path:
        model = mae_vit_base_patch16()
    elif "vit_large" in args.weights_path:
        model = mae_vit_large_patch16()
    elif "vit_huge" in args.weights_path:
        model = mae_vit_huge_patch14()
    else:
        raise ValueError(f"Invalid weights_path: {args.weights_path}")
    return model

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    if "model" in checkpoint.keys():
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    return model