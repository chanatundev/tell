"""Leak check: can a logistic regression on NON-CONTENT file stats separate AI from real?

Fit on train, AUROC on val. Run on RAW files and on NORMALISED files.
  raw       : width, height, aspect, file bytes, mean JPEG quantisation (0 = lossless/PNG)
  norm      : the same stats of the normalised files (+ blockiness)
  norm_content_aspect : aspect of the picture area inside the padded canvas (residual shape cue)
  blockiness_only     : 8x8 JPEG grid strength of the normalised files (double-compression cue)
After normalisation the METADATA stats (w, h, aspect, quantisation) are constant, so that row is 0.5 by
construction. File bytes after a fixed-quality encode measure image compressibility (a CONTENT property:
texture/noise/smoothness) and blockiness measures earlier JPEG compression; both can legitimately differ
between AI and real images and cannot be removed by re-encoding. They are reported separately, not hidden:
a high value there is a residual-shortcut warning (e.g. real Commons JPEGs vs PNG fakes), not a bug.

  python -I src/leak_check.py --splits data/splits.csv --stats data/raw_stats.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalize as N


def blockiness(path, cw, ch):
    """Ratio of mean |horizontal diff| across 8-px block boundaries vs inside blocks (>1 => JPEG grid)."""
    im = Image.open(path).convert("L")
    l, t, r, b = N.content_box(cw, ch, im.size[0])
    a = np.asarray(im.crop((l, t, r, b)), dtype=np.float32)
    a = a[:, : (a.shape[1] // 8) * 8]
    d = np.abs(np.diff(a, axis=1))  # d[:, k] between col k and k+1
    k = np.arange(d.shape[1])
    on = d[:, (k % 8) == 7].mean()
    off = d[:, (k % 8) != 7].mean()
    return float(on / (off + 1e-6))


def norm_features(df, norm_dir):
    rows = []
    for _, r in df.iterrows():
        p = norm_dir / r["path"]
        with Image.open(p) as im:
            w, h = im.size
        rows.append(dict(w=w, h=h, aspect=w / h, logbytes=np.log(p.stat().st_size), quant=float(
            np.mean([np.mean(v) for v in Image.open(p).quantization.values()])),
            block=blockiness(p, int(r["cw"]), int(r["ch"])),
            c_aspect=r["cw"] / r["ch"]))
    return pd.DataFrame(rows, index=df.index)


def auroc(X, y, split):
    tr, va = (split == "train").values, (split == "val").values
    if len(set(y[va])) < 2 or len(set(y[tr])) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
    return float(roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits.csv"))
    ap.add_argument("--stats", default=str(ROOT / "data" / "raw_stats.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "leak_check.json"))
    a = ap.parse_args()
    st = pd.read_csv(a.stats)
    y, split = st["label"].values, st["split"]
    raw = pd.DataFrame(dict(w=st.raw_w, h=st.raw_h, aspect=st.raw_w / st.raw_h, logbytes=np.log(st.raw_bytes),
                            quant=st.raw_mean_quant))
    nf = norm_features(st, ROOT / "data" / "norm")
    res = {
        "raw_all_stats": auroc(raw.values, y, split),
        "raw_size_only(w,h,aspect)": auroc(raw[["w", "h", "aspect"]].values, y, split),
        "raw_bytes_only": auroc(raw[["logbytes"]].values, y, split),
        "norm_metadata_stats(w,h,aspect,quant)": auroc(nf[["w", "h", "aspect", "quant"]].values, y, split),
        "norm_all_file_stats(+bytes)": auroc(nf[["w", "h", "aspect", "logbytes", "quant"]].values, y, split),
        "norm_bytes_only(=image compressibility)": auroc(nf[["logbytes"]].values, y, split),
        "norm_content_aspect_only": auroc(nf[["c_aspect"]].values, y, split),
        "norm_blockiness_only": auroc(nf[["block"]].values, y, split),
        "n_val": int((split == "val").sum()), "n_val_ai": int(((split == "val") & (y == 1)).sum()),
    }
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:50s} {v if isinstance(v, int) else round(v, 3)}")


if __name__ == "__main__":
    main()
