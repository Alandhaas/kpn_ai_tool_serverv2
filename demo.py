# ============================================
# CBM Concept Training (Cost-aware + Augmented)
# ============================================

import os
import io
import json
import tempfile
import random
import numpy as np
import dataiku

from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import timm
from safetensors.torch import load_file

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)

import matplotlib.pyplot as plt

# ================= CONFIG =================

DATASET_ID = "dataset_origin_split"
IMAGE_FOLDER_ID = "all_images_preprocessed"
SSL_FOLDER_ID = "MODEL_WEIGHTS"
SSL_FILE = "timmconvnext_base.fb_in22k_ft_in1k.safetensors"

MODEL_FOLDER_ID = "conv-base_unfreezed_concept_head"
OUT_MODEL = "base_backbone_concept_head.pth"
LAST_STAGE_FILE = "convnext_base_last_stage_finetuned.pth"
METRICS_JSON = "base_backbone_concept_metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 320
BATCH_SIZE = 32
EPOCHS = 25

LR_HEAD = 3e-5
LR_BACKBONE_LAST_STAGE = 1e-5
WEIGHT_DECAY = 1e-4

DEBUG_LIMIT = None

# semantics: 1 = OK, 0 = violation
CONCEPTS = [
    ("rule_free_space_ok", "rule_free_space"),
    ("rule_cable_routing_ok", "rule_cable_routing"),
    ("rule_alignment_ok", "rule_alignment"),
    ("rule_covering_ok", "rule_covering"),
]

COST_FP = 8.0
COST_FN = 2.0
NEG_WEIGHT_MULT = 1.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ================= AUGMENTATIONS =================

train_transform_mild = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.08, 0.08, 0.06, 0.02),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_transform_strong = transforms.Compose([
    transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.12, 0.12, 0.10, 0.03),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ================= LOAD DATA =================

df = dataiku.Dataset(DATASET_ID).get_dataframe()
df = df[df["is_meterkast"] == True].reset_index(drop=True)

df_train = df[df["split"] == "train"].reset_index(drop=True)
df_val = df[df["split"] == "val"].reset_index(drop=True)

if DEBUG_LIMIT is not None:
    df_train = df_train.head(DEBUG_LIMIT)
    df_val = df_val.head(DEBUG_LIMIT)

for ok_col, _ in CONCEPTS:
    df_train[ok_col] = df_train[ok_col].astype(int)
    df_val[ok_col] = df_val[ok_col].astype(int)

IMG_COL = "path" if "path" in df.columns else "image_path"

IMG_FOLDER = dataiku.Folder(IMAGE_FOLDER_ID)
SSL_FOLDER = dataiku.Folder(SSL_FOLDER_ID)
MODEL_FOLDER = dataiku.Folder(MODEL_FOLDER_ID)

# ================= CACHE IMAGES =================

needed = set(df_train[IMG_COL].tolist() + df_val[IMG_COL].tolist())
local_dir = tempfile.mkdtemp(prefix="cbm_aug_")
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

class ConceptDataset(Dataset):
    def __init__(self, df_, is_train=False):
        self.df = df_.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = row[IMG_COL]
        if p not in local_files:
            return None

        img = Image.open(local_files[p]).convert("RGB")

        y = torch.tensor(
            [int(row[ok_col] == 1) for ok_col, _ in CONCEPTS],
            dtype=torch.float32
        )

        if self.is_train:
            n_viol = int((1 - y).sum().item())
            p_strong = min(0.2 + 0.15 * n_viol, 0.8)
            use_strong = random.random() < p_strong
            x = (train_transform_strong if use_strong else train_transform_mild)(img)
        else:
            x = val_transform(img)

        return x, y

def collate_skip_missing(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

# ================= SAMPLER =================

viol_counts = []
for _, r in df_train.iterrows():
    viol_counts.append(sum(int(r[c[0]] == 0) for c in CONCEPTS))

alpha = 2.5
weights = torch.tensor(1.0 + alpha * np.array(viol_counts), dtype=torch.double)

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)

train_loader = DataLoader(
    ConceptDataset(df_train, is_train=True),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    shuffle=False,
    collate_fn=collate_skip_missing
)

val_loader = DataLoader(
    ConceptDataset(df_val, is_train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_skip_missing
)

# ================= MODEL =================

tmp = tempfile.mktemp(".safetensors")
with SSL_FOLDER.get_download_stream(SSL_FILE) as s:
    with open(tmp, "wb") as f:
        f.write(s.read())

convnext = timm.create_model("convnext_base", pretrained=False)
convnext.load_state_dict(load_file(tmp), strict=False)
convnext.to(DEVICE)

for p in convnext.parameters():
    p.requires_grad = False

last_stage = convnext.stages[3]
for p in last_stage.parameters():
    p.requires_grad = True

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

class ConceptHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce2 = nn.Conv2d(512, 576, 1)
        self.reduce3 = nn.Conv2d(1024, 448, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Linear(256, len(CONCEPTS))

    def forward(self, feats):
        f2, f3 = feats
        x = torch.cat([
            self.reduce2(f2),
            F.interpolate(
                self.reduce3(f3),
                size=f2.shape[-2:],
                mode="bilinear"
            )
        ], dim=1)
        x = self.fuse(x)
        x = self.pool(x).flatten(1)
        return self.mlp(x)

head = ConceptHead().to(DEVICE)

# ================= LOSS =================

def apply_temperature(logits, T_vec):
    T = torch.tensor(T_vec).view(1, -1)
    return torch.sigmoid(logits / T)

def fit_temperature_binary(logits_1d, targets_1d, max_iter=50):
    logits = logits_1d.float()
    targets = targets_1d.float()
    logT = torch.zeros(1, requires_grad=True)

    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = torch.exp(logT)
        loss = F.binary_cross_entropy_with_logits(logits / T, targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())






def compute_neg_weights(df_):
    neg_w = []
    n = len(df_)
    for ok_col, _ in CONCEPTS:
        n_pos = df_[ok_col].sum()
        n_neg = max(n - n_pos, 1)
        w = (n_pos / n_neg) * NEG_WEIGHT_MULT * (COST_FP / COST_FN)
        neg_w.append(w)
    return torch.tensor(neg_w, device=DEVICE)

NEG_W = compute_neg_weights(df_train)
POS_W = torch.ones_like(NEG_W)

def weighted_bce(logits, targets):
    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    return -(POS_W * targets * log_p + NEG_W * (1 - targets) * log_1mp).mean()

# ================= TRAIN LOOP =================

optimizer = torch.optim.AdamW(
    [
        {"params": head.parameters(), "lr": LR_HEAD},
        {"params": last_stage.parameters(), "lr": LR_BACKBONE_LAST_STAGE},
    ],
    weight_decay=WEIGHT_DECAY
)

history = []

for epoch in range(EPOCHS):
    backbone.train()
    head.train()

    loss_sum, n_seen = 0.0, 0

    for x, y in train_loader:
        if x is None:
            continue

        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = head(backbone(x))
        loss = weighted_bce(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        n_seen += x.size(0)

    train_loss = loss_sum / max(n_seen, 1)
    history.append({"epoch": epoch + 1, "train_loss": train_loss})

# ================= SAVE =================

buf = io.BytesIO()
torch.save(head.state_dict(), buf)
buf.seek(0)
MODEL_FOLDER.upload_stream(OUT_MODEL, buf)

buf = io.BytesIO()
torch.save(last_stage.state_dict(), buf)
buf.seek(0)
MODEL_FOLDER.upload_stream(LAST_STAGE_FILE, buf)

buf = io.BytesIO(json.dumps(history, indent=2).encode("utf-8"))
buf.seek(0)
MODEL_FOLDER.upload_stream(METRICS_JSON, buf)

print("[DONE] Training complete.")
