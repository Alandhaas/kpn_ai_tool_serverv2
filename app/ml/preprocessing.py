import numpy as np
import torch
import cv2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(img_rgb: np.ndarray, img_size: int) -> torch.Tensor:
    """
    Input:
      img_rgb: uint8 RGB image [H,W,3]
    Output:
      torch.Tensor [1,3,H,W] normalized
    """
    img = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.unsqueeze(0)
