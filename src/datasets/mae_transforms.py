import torch
from PIL import Image
from torchvision import transforms

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]

# Normalization for ImageNet
class ImageNetTransforms():
    def __init__(self, input_res: int):
        
        self.mean = torch.Tensor(IMNET_MEAN).view(1, 3, 1, 1)
        self.std = torch.Tensor(IMNET_STD).view(1, 3, 1, 1)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((input_res, input_res)),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD)
        ])
    
    def __call__(self, img: Image) -> torch.Tensor:
        return self.img_transform(img)
    
    def inverse_affine(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(self.std.device)
        return img * self.std + self.mean