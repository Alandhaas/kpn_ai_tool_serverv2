import io
import json
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.ml.preprocessing import preprocess_image, IMAGENET_MEAN, IMAGENET_STD
from app.ml.inference import run_inference, apply_thresholds
from app.ml.gradcam import compute_gradcam, overlay_cam

import zipfile

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
        self.final_threshold = float(thresholds.get("threshold_final_classifier", 0.5))

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

        results = apply_thresholds(
            logits=logits,
            thresholds=self.thresholds,
            concept_order=self.concept_order,
        )

        final_logit = self.models.final_head(logits)
        final_prob = torch.sigmoid(final_logit)[0].item()
        final_pred = bool(final_prob >= self.final_threshold)

        results["final_prob"] = float(final_prob)
        results["final_pred"] = final_pred
        results["final_threshold"] = float(self.final_threshold)
        return results
    
    def predict_witch_gradcam(
        self, img_rgb: np.ndarray
    ) -> Tuple[List[np.ndarray], Dict, bool, Dict]:
        x = preprocess_image(img_rgb, settings.img_size).to(self.device)

        feats = self.models.feature_extractor(x)
        logits, fused = self.models.concept_head(feats, return_fused=True)
        fused.retain_grad()

        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
        final_logit = self.models.final_head(logits)
        final_prob = torch.sigmoid(final_logit)[0].item()
        final_pred = bool(final_prob >= self.final_threshold)

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

        final_result = {
            "final_prob": float(final_prob),
            "final_pred": final_pred,
            "final_threshold": float(self.final_threshold),
        }
        return tiles, per_concept, final_ok, final_result
            

    # ---------------------------------------------------------
    # Inference + Grad-CAM (2x2 montage, base64 PNG)
    # ---------------------------------------------------------
    def predict_with_gradcam_bytes(self, img_rgb: np.ndarray) -> Tuple[Dict, bytes]:
        """
        Returns: (json_result, png_bytes)
        """
        tiles, per_concept, final_ok, final_result = self.predict_witch_gradcam(img_rgb)

        if not tiles:
            raise ValueError("No Grad-CAM tiles generated")

        # Build montage in rows of 2, pad if odd.
        if len(tiles) % 2 == 1:
            blank = np.zeros_like(tiles[0])
            tiles = tiles + [blank]

        rows = [np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, len(tiles), 2)]
        montage = rows[0] if len(rows) == 1 else np.concatenate(rows, axis=0)

        buf = io.BytesIO()
        Image.fromarray(montage).save(buf, format="PNG")
        png_bytes = buf.getvalue()

        json_result = {"final_ok": final_ok, "per_concept": per_concept, **final_result}
        return json_result, png_bytes
    

    def predict_with_gradcam_zip_bytes(
        self,
        img_rgb_list: Union[List[np.ndarray], np.ndarray],
    ) -> bytes:
        """
        Returns: zipfile bytes containing gradcam PNGs and a JSON with all metadata.
        """
        if isinstance(img_rgb_list, np.ndarray):
            img_list = [img_rgb_list]
        else:
            img_list = list(img_rgb_list)

        results = []
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for idx, img_rgb in enumerate(img_list):
                tiles, per_concept, final_ok, final_result = self.predict_witch_gradcam(img_rgb)
                image_entry = {
                    "image_index": idx + 1,
                    "final_ok": final_ok,
                    "per_concept": per_concept,
                    **final_result,
                    "files": [],
                }
                x = preprocess_image(img_rgb, settings.img_size).squeeze(0)
                x = (x * IMAGENET_STD) + IMAGENET_MEAN
                x = x.clamp(0, 1)
                x = (x * 255.0).byte().permute(1, 2, 0).cpu().numpy()

                original_buf = io.BytesIO()
                Image.fromarray(x).save(original_buf, format="PNG")
                original_bytes = original_buf.getvalue()
                original_name = f"image_{idx+1:03d}_original.png"
                zip_file.writestr(original_name, original_bytes)
                image_entry["files"].append(
                    {"concept": "original", "filename": original_name}
                )
                for concept_idx, tile in enumerate(tiles):
                    concept_name = self.concept_order[concept_idx]
                    buf = io.BytesIO()
                    Image.fromarray(tile).save(buf, format="PNG")
                    png_bytes = buf.getvalue()
                    filename = f"image_{idx+1:03d}_{concept_name}_gradcam.png"
                    zip_file.writestr(filename, png_bytes)
                    image_entry["files"].append(
                        {"concept": concept_name, "filename": filename}
                    )
                results.append(image_entry)
            zip_file.writestr("metadata.json", json.dumps(results))

        return zip_buf.getvalue()
