"""Model, transforms and scoring shared by train.py, eval_heldout.py and the notebook.

Input handling (aspect-neutral on purpose):
  * every image is first cropped to its picture area (padding removed, see normalize.content_box)
  * TRAIN : random SQUARE window, side = sqrt(U(0.5,1)) * short side (a random-resized-crop that never
            stretches the aspect ratio), small rotation, JPEG recompress q60-95, slight blur, colour
            jitter, resized to `size` (512). NO horizontal flip (text).
  * EVAL  : up to 3 evenly spaced square windows of side = short side (whole receipt coverage), resized
            to `size`; logits are averaged.
"""
import math, random, io
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms as T

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalize as N

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_norm = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
_jitter = T.ColorJitter(0.25, 0.25, 0.2, 0.02)
KMAX = 3


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_model(pretrained=True):
    import timm
    try:
        m = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)
        got = pretrained
    except Exception as e:  # offline etc.
        print("WARNING: could not load pretrained weights:", repr(e)[:200])
        m, got = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1), False
    m.pretrained_loaded = got
    return m


def load_content(path, cw, ch):
    im = Image.open(path).convert("RGB")
    if im.size == (N.LONG, N.LONG) and (cw, ch) != im.size:
        im = im.crop(N.content_box(int(cw), int(ch), N.LONG))
    return im


def train_view(im, size):
    w, h = im.size
    short = min(w, h)
    side = max(32, int(short * math.sqrt(random.uniform(0.5, 1.0))))
    x0, y0 = random.randint(0, w - side), random.randint(0, h - side)
    c = im.crop((x0, y0, x0 + side, y0 + side))
    c = c.rotate(random.uniform(-5, 5), resample=Image.Resampling.BICUBIC, fillcolor=(128, 128, 128))
    c = c.resize((size, size), Image.Resampling.LANCZOS)
    if random.random() < 0.6:  # JPEG re-compression q 60-95
        b = io.BytesIO(); c.save(b, "JPEG", quality=random.randint(60, 95)); b.seek(0); c = Image.open(b).convert("RGB")
    if random.random() < 0.25:
        c = c.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
    c = _jitter(c)
    return _norm(c)


def eval_views(im, size):
    w, h = im.size
    short, long_ = min(w, h), max(w, h)
    k = int(min(KMAX, max(1, round(long_ / short))))
    offs = [0] if k == 1 else [round(i * (long_ - short) / (k - 1)) for i in range(k)]
    views = []
    for o in offs:
        box = (0, o, short, o + short) if h >= w else (o, 0, o + short, short)
        views.append(_norm(im.crop(box).resize((size, size), Image.Resampling.LANCZOS)))
    while len(views) < KMAX:  # repeat to a fixed number of views; mean over duplicates is unchanged
        views.append(views[len(views) % k])
    return torch.stack(views)


class DS(torch.utils.data.Dataset):
    def __init__(self, df, norm_dir, size, train):
        self.df, self.dir, self.size, self.train = df.reset_index(drop=True), Path(norm_dir), size, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        im = load_content(self.dir / r["path"], r["cw"], r["ch"])
        x = train_view(im, self.size) if self.train else eval_views(im, self.size)
        y = float(r["label"]) if "label" in r else 0.0
        return x, torch.tensor(y)


@torch.no_grad()
def predict_logits(model, loader, device):
    """Mean logit over eval windows. Loader must be built with train=False."""
    model.eval(); out = []
    for x, _ in loader:
        b, k = x.shape[:2]
        x = x.to(device).flatten(0, 1)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            z = model(x).float().view(b, k)
        out.append(z.mean(1).cpu())
    return torch.cat(out).numpy()


def score_files(model, df, norm_dir, size, device, temperature=1.0, batch=16, workers=0):
    """df: path, cw, ch -> calibrated probabilities (np array)."""
    dl = torch.utils.data.DataLoader(DS(df, norm_dir, size, False), batch_size=batch, shuffle=False, num_workers=workers)
    z = predict_logits(model, dl, device)
    from scipy.special import expit
    return expit(z / temperature), z
