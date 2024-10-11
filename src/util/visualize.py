import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

def gaussian_kernel(size: int, sigma: float):
    """Generate a 2D Gaussian kernel
    Args:
        size (int): size of the kernel
        sigma (float): standard deviation of the Gaussian distribution
    Returns:
        kernel (tensor): 2D Gaussian kernel
    """
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x = torch.exp(-x**2 / (2 * sigma**2))
    x = x / x.sum()  # normalize the kernel
    kernel_2d = x[:, None] * x[None, :]
    return kernel_2d

def preds_to_image(preds, patch_size, h, w):
    """Convert MAE predictions to image
    Args:
        preds (tensor): predictions from MAE model, shape of (N, L, p*p*3)
        patch_size (int): the size of patch
        h (int): the height of the feature map
        w (int): the width of the feature map
    Returns:
        img (tensor): the image reconstructed from the predictions, shape of (N, C, H, W)
    """
    N, L, _ = preds.shape
    p = patch_size
    C = 3
    img = torch.zeros((N, C, h*p, w*p), dtype=torch.float32, device=preds.device)
    for i in range(N):
        for j in range(L):
            x, y = j % w, j // w
            img[i, :, y*p:(y+1)*p, x*p:(x+1)*p] = preds[i, j].view(C, p, p)
    return img

def visualize_on_masked_area(img, mask, preds, patch_size, h, w, inverse_transform=None):
    """Visualize the image on masked area,
    Args:
        img (tensor): original image, shape of (N, C, H, W)
        mask (tensor): binary mask of patches, shape of (N, H, W)
        preds (tensor): predictions from MAE model, shape of (N, C, H, W) or (N, L, p*p*3)
        patch_size (int): patch size
        h (int): the height of the feature map
        w (int): the width of the feature map
        inverse_transform (callable, optional): inverse transform for the image. Defaults to None.
    Returns:
        masked_img (tensor): the image with the predictions on the masked area, shape of (N, C, H, W)
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.shape[1] == 1:
        mask = mask.repeat(1, 3, 1, 1)
    if preds.shape == img.shape:    
        preds_img = preds
    else:
        preds_img = preds_to_image(preds, patch_size, h, w)  # (N, C, H, W)
    
    if inverse_transform:
        img = inverse_transform(img)
        preds_img = inverse_transform(preds_img)
    
    masked_img = img.clone()
    masked_img[mask] = preds_img[mask]
    return masked_img

def save_tensor_image(tensor, filename):
    """Save a tensor image to a file
    Args:
        tensor (torch.Tensor): Image tensor. shape of (C, H, W) or (B, C, H, W).
        filename (str): The file name to save the image.
    """
    np_grid = tensor.numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))  
    if np_grid.shape[-1] == 1:
        # For grayscale image
        np_grid = np.squeeze(np_grid, axis=-1)
    plt.imsave(filename, np_grid)

def show_tensor_image(tensor, nrow=1, normalize=True, title=None):
    """Display a tensor image

    Args:
        tensor (torch.Tensor): Image tensor. shape of (C, H, W) or (B, C, H, W).
        nrow (int, optional): _description_. Defaults to 1.
        normalize (bool, optional): _description_. Defaults to True.
        title (_type_, optional): _description_. Defaults to None.
    """
    if tensor.dim() == 4:
        grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=normalize)
    else:
        grid = tensor

    np_grid = grid.numpy()
    plt.imshow(np.transpose(np_grid, (1, 2, 0)))
    
    if title:
        plt.title(title)
    plt.axis('off') 
    plt.show()