from .mvtec_ad import MVTecAD, AD_CLASSES
from .mvtec_loco import MVTecLOCO, LOCO_CLASSES

from torchvision import transforms
from torch.utils.data import Subset

import random
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def build_dataset(args, transform=None, eval=False):    
    if eval:
        args.num_normal_samples = -1
    
    assert transform is not None, "transform should not be specified"
    if 'mvtec_ad' in args.data_root:
        if args.class_name not in AD_CLASSES:
            raise ValueError(f"Invalid class_name: {args.class_name}")
        dataset = MVTecAD(args.data_root, args.class_name, args.img_size, args.split, custom_transforms=transform)
        if args.num_normal_samples > 0:
            dataset = Subset(dataset, random.sample(range(len(dataset)), args.num_normal_samples))
        
    elif 'mvtec_loco' in args.data_root:  
        if args.class_name not in LOCO_CLASSES:
            raise ValueError(f"Invalid class_name: {args.class_name}")
        dataset = MVTecLOCO(args.data_root, args.class_name, args.img_size, args.split, custom_transforms=transform)
        if args.num_normal_samples > 0:
            dataset = Subset(dataset, random.sample(range(len(dataset)), args.num_normal_samples))
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    return dataset

def build_transforms(args):
    default_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Random Crop, Random rotation
    crop_rot_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=args.img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if args.transform == 'default':
        return default_transform
    elif args.transform == 'crop_rot':
        return crop_rot_transform
    else:
        raise ValueError(f"Invalid transform: {args.transform}")


    
    