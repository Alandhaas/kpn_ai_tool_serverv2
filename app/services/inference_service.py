import io
from typing import Dict, List
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.ml.preprocessing import preprocess_image
from app.ml.inference import run_inference, apply_thresholds
from app.ml.gradcam import compute_gradcam, overlay_cam


class InferenceService:
    def __init__(
        self,
        models,
        thresholds: Dict[str, float],
        concept_order: List[str],
    ):
        self.models = models
        self.thresholds = thresholds
        self.concept_order = concept_order
        self.device = settings.device_torch

    # ---------------------------------------------------------
    # Standard inference (NO Grad-CAM)
    # ---------------------------------------------------------
    def predict(self, img_rgb: np.ndarray) -> Dict:
        """
        Run inference on a single RGB image.
        Returns:
          {
            "final_ok": bool,
            "per_concept": {
              concept_name: { p_ok, threshold, pred_ok }
            }
          }
        """
        x = preprocess_image(img_rgb, settings.img_size).to(self.device)

        logits = run_inference(
            self.models.feature_extractor,
            self.models.concept_head,
            x,
        )

        return apply_thresholds(
            logits=logits,
            thresholds=self.thresholds,
            concept_order=self.concept_order,
        )

    # ---------------------------------------------------------
    # Inference + Grad-CAM (2x2 montage, base64 PNG)
    # ---------------------------------------------------------
    def predict_with_gradcam_bytes(self, img_rgb: np.ndarray) -> Tuple[Dict, bytes]:
        """
        Returns: (json_result, png_bytes)
        """
        x = preprocess_image(img_rgb, settings.img_size).to(self.device)

        feats = self.models.feature_extractor(x)
        logits, fused = self.models.concept_head(feats, return_fused=True)
        fused.retain_grad()

        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        tiles = []
        per_concept = {}
        final_ok = True

        # Belangrijk: overlay op resized image (zelfde size als cam)
        # Reuse dezelfde resize als preprocess (zonder normalisatie gedoe):
        # => simpel: maak een copy van img_rgb en resize op dezelfde manier als preprocess
        import cv2
        img_resized = cv2.resize(img_rgb, (settings.img_size, settings.img_size), interpolation=cv2.INTER_AREA)

        for idx, cname in enumerate(self.concept_order):
            p_ok = float(probs[idx])
            thr = self.thresholds[cname]
            pred_ok = p_ok >= thr
            final_ok &= bool(pred_ok)

            cam = compute_gradcam(fused_feats=fused, logits=logits, concept_idx=idx, img_size=settings.img_size)
            overlay = overlay_cam(img_rgb=img_resized, cam=cam, alpha=settings.alpha_overlay)
            tiles.append(overlay)

            per_concept[cname] = {
                "p_ok": p_ok,
                "threshold": thr,
                "pred_ok": bool(pred_ok),
            }

            # cleanup grads + re-forward
            self.models.feature_extractor.zero_grad(set_to_none=True)
            self.models.concept_head.zero_grad(set_to_none=True)
            fused.grad = None

            feats = self.models.feature_extractor(x)
            logits, fused = self.models.concept_head(feats, return_fused=True)
            fused.retain_grad()

        # 2x2 montage
        top = np.concatenate(tiles[:2], axis=1)
        bottom = np.concatenate(tiles[2:], axis=1)
        montage = np.concatenate([top, bottom], axis=0)

        buf = io.BytesIO()
        Image.fromarray(montage).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        json_result = {"final_ok": final_ok, "per_concept": per_concept}
        return json_result, png_bytes
