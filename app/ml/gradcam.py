import numpy as np
import torch
import torch.nn.functional as F
import cv2


def compute_gradcam(fused_feats, logits, concept_idx: int, img_size: int) -> np.ndarray:
    """
    Returns Grad-CAM heatmap in [0,1], resized to img_size.
    """
    logits[0, concept_idx].backward(retain_graph=True)

    grads = fused_feats.grad              # [1,C,H,W]
    acts = fused_feats.detach()            # [1,C,H,W]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1)[0]   # [H,W]
    cam = F.relu(cam)

    cam = cam.cpu().numpy()
    if cam.max() > 1e-8:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    cam = cv2.resize(cam, (img_size, img_size))
    return cam


def overlay_cam(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (1 - alpha) * img_rgb.astype(np.float32) + alpha * heat.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)
