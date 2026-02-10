import os, io, json, tempfile
import numpy as np
import dataiku
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

# ================= CONFIG =================

vars = dataiku.get_custom_variables()

DATASET_ID = "dataset_origin_split"
IMAGE_FOLDER_ID = "all_images_preprocessed"

SSL_FOLDER_ID = "MODEL_WEIGHTS"
SSL_FILE = vars["BACKBONE_WEIGHTS_FILE"]
BACKBONE_NAME = vars["BACKBONE_NAME"]

INPUT_CONCEPT_FOLDER_ID = "Concept_head_training_results"
INPUT_CONCEPT_HEAD = f"{BACKBONE_NAME}/backbone_concept_head.pth"
INPUT_CONCEPT_LAST_STAGE = f"{BACKBONE_NAME}/backbone_last_stage.pth"
INPUT_METRICS_JSON = f"{BACKBONE_NAME}/concept_head_metrics.json"

INPUT_FINAL_FOLDER_ID = "Final_classifier_training_results"
INPUT_FINAL_CLASSIFIER = f"{BACKBONE_NAME}/final_classifier_head.pth"

OUTPUT_FOLDER_ID = "Final_classifier_inference_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 320
BATCH_SIZE = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CONCEPTS = [
    ("rule_free_space_ok",   "rule_free_space"),
    ("rule_cable_routing_ok","rule_cable_routing"),
    ("rule_alignment_ok",    "rule_alignment"),
    ("rule_covering_ok",     "rule_covering"),
]

# ================= TRANSFORMS =================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ================= LOAD DATA =================

df = dataiku.Dataset(DATASET_ID).get_dataframe()
df = df[df["split"] == "test"].reset_index(drop=True)

IMG_COL = "path" if "path" in df.columns else "image_path"

IMG_FOLDER = dataiku.Folder(IMAGE_FOLDER_ID)
INPUT_CONCEPT_FOLDER = dataiku.Folder(INPUT_CONCEPT_FOLDER_ID)
INPUT_FINAL_FOLDER = dataiku.Folder(INPUT_FINAL_FOLDER_ID)
OUTPUT_FOLDER = dataiku.Folder(OUTPUT_FOLDER_ID)

# ================= CACHE IMAGES =================

needed = set(df[IMG_COL].tolist())
local_dir = tempfile.mkdtemp(prefix="cbm_test_")
local_files = {}

for p in IMG_FOLDER.list_paths_in_partition():
    if p in needed and p.lower().endswith((".jpg", ".jpeg", ".png")):
        with IMG_FOLDER.get_download_stream(p) as s:
            raw = s.read()
        safe = p.replace("/", "_")
        fp = os.path.join(local_dir, safe)
        with open(fp, "wb") as f:
            f.write(raw)
        local_files[p] = fp

# ================= DATASET =================

class TestDataset(Dataset):
    def __init__(self, df_):
        self.df = df_.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row[IMG_COL]
        if p not in local_files:
            return None

        img = Image.open(local_files[p]).convert("RGB")
        x = transform(img)

        y_concepts = torch.tensor(
            [int(row[c[0]] == 1) for c in CONCEPTS],
            dtype=torch.float32
        )
        y_final = y_concepts.min().unsqueeze(0)

        return x, y_final

def collate_skip_missing(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

loader = DataLoader(
    TestDataset(df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_skip_missing
)

# ================= MODEL =================

tmp = tempfile.mktemp(".safetensors")
with dataiku.Folder(SSL_FOLDER_ID).get_download_stream(SSL_FILE) as s:
    with open(tmp, "wb") as f:
        f.write(s.read())

convnext = timm.create_model("convnext_tiny", pretrained=False)
convnext.load_state_dict(load_file(tmp), strict=False)

class ConvNeXtFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.stem = backbone.stem
        self.stages = backbone.stages

    def forward(self, x):
        x = self.stem(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        f2 = self.stages[2](x)
        f3 = self.stages[3](f2)
        return f2, f3

backbone = ConvNeXtFeatures(convnext).to(DEVICE)

last_stage_state = torch.load(
    io.BytesIO(INPUT_CONCEPT_FOLDER.get_download_stream(INPUT_CONCEPT_LAST_STAGE).read())
)
convnext.stages[3].load_state_dict(last_stage_state, strict=True)

class ConceptHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce2 = nn.Conv2d(384, 384, 1)
        self.reduce3 = nn.Conv2d(768, 640, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Linear(256, 4)

    def forward(self, feats):
        f2, f3 = feats
        x = torch.cat([
            self.reduce2(f2),
            F.interpolate(self.reduce3(f3), size=f2.shape[-2:], mode="bilinear")
        ], dim=1)
        x = self.fuse(x)
        x = self.pool(x).flatten(1)
        return self.mlp(x)

head = ConceptHead().to(DEVICE)
head_state = torch.load(io.BytesIO(INPUT_CONCEPT_FOLDER.get_download_stream(INPUT_CONCEPT_HEAD).read()))
head.load_state_dict(head_state, strict=True)

# ================= LOAD THRESHOLDS =================

with INPUT_CONCEPT_FOLDER.get_download_stream(INPUT_METRICS_JSON) as s:
    metrics = json.loads(s.read().decode("utf-8"))

# use last epoch
metrics = metrics[-1]

thr_prob = [
    float(metrics["thresholds"][cname])
    for _, cname in CONCEPTS
]

def logit(p, eps=1e-6):
    p = torch.clamp(torch.tensor(p), eps, 1 - eps)
    return torch.log(p / (1 - p))

thr_logit = logit(thr_prob).to(DEVICE)

# ================= FINAL HEAD =================

class ThresholdAwareFinalHead(nn.Module):
    def __init__(self, thr_logit):
        super().__init__()
        K = thr_logit.numel()
        self.register_buffer("thr_logit", thr_logit)
        self.w = nn.Parameter(torch.ones(K))
        self.b = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(8.0))
        self.delta = nn.Parameter(torch.full((K,), 0.5))

    def forward(self, logits):
        d = logits - self.thr_logit
        neg = F.relu(-d)
        gate = torch.sigmoid(self.alpha * (torch.abs(d) - self.delta))
        penalty = (neg * gate * self.w).sum(dim=1)
        return self.b - penalty
    
ckpt = torch.load(
    io.BytesIO(INPUT_FINAL_FOLDER.get_download_stream(INPUT_FINAL_CLASSIFIER).read()),
    map_location=DEVICE
)
final_head = ThresholdAwareFinalHead(ckpt["thr_logit"]).to(DEVICE)
final_head.load_state_dict(ckpt["final_head_state"], strict=True)

# ================= INFERENCE =================

backbone.eval()
head.eval()
final_head.eval()

all_probs, all_true = [], []

with torch.no_grad():
    for x, y in loader:
        if x is None:
            continue
        x = x.to(DEVICE)
        y = y.to(DEVICE).squeeze(1)

        concept_logits = head(backbone(x))
        y_logit = final_head(concept_logits)
        y_prob = torch.sigmoid(y_logit)

        all_probs.append(y_prob.cpu())
        all_true.append(y.cpu())

y_prob = torch.cat(all_probs).numpy()
y_true = torch.cat(all_true).numpy().astype(int)

# ================= METRICS =================

y_pred = (y_prob >= 0.5).astype(int)

tp = ((y_true == 1) & (y_pred == 1)).sum()
tn = ((y_true == 0) & (y_pred == 0)).sum()
fp = ((y_true == 0) & (y_pred == 1)).sum()
fn = ((y_true == 1) & (y_pred == 0)).sum()

p, r, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

try:
    auroc = roc_auc_score(y_true, y_prob)
except ValueError:
    auroc = None

out = {
    "n_test": int(len(y_true)),
    "threshold": 0.5,
    "precision": float(p),
    "recall": float(r),
    "f1": float(f1),
    "auroc": auroc,
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
}

# ================= SAVE RESULTS =================

buf = io.BytesIO(json.dumps(out, indent=2).encode("utf-8"))
buf.seek(0)
OUTPUT_FOLDER.upload_stream(f"{BACKBONE_NAME}/final_test_metrics.json", buf)

print("[DONE] Test set evaluation complete.")
