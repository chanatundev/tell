"""Normalise all training sources and build the split file.

  python -I src/prepare.py                 # needs data/raw/gpt4o_receipt/*.png (gated download)
  python -I src/prepare.py --allow-synth   # pipeline smoke test with the synthetic stand-in AI class
                                           # -> writes data/splits_smoke.csv (NEVER use for real results)

Outputs
  data/norm/<source>/*.jpg          normalised images (see normalize.py)
  data/raw_stats.csv                per-file stats of the RAW files (used by leak_check / EXIF baseline)
  data/splits.csv                   path (relative to data/norm), label (1=AI, 0=real), source, split, group, cw, ch, raw_path
"""
import argparse, json, re, random, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalize as N

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NORM = ROOT / "data" / "norm"


def raw_stats(p: Path) -> dict:
    with Image.open(p) as im:
        w, h = im.size
        fmt = im.format
        q = getattr(im, "quantization", None)
        mq = float(np.mean([np.mean(v) for v in q.values()])) if q else 0.0  # 0 => lossless / no JPEG tables
        exif = 0
        try:
            ex = im.getexif()
            exif = int(any(k in ex for k in (271, 272, 306, 36867, 33434, 34850)))  # make/model/software/date/exposure
            if 34665 in ex:  # Exif sub-IFD pointer
                exif = int(exif or len(ex.get_ifd(34665)) > 0)
        except Exception:
            pass
    return dict(raw_w=w, raw_h=h, raw_bytes=p.stat().st_size, raw_format=fmt, raw_mean_quant=mq, raw_has_camera_exif=exif)


def gpt_group(name: str) -> str:
    s = re.sub(r"^receipt_", "", Path(name).stem)
    s = re.sub(r"_?\d+$", "", s)
    return s if s else Path(name).stem  # un-named files are singleton groups


def group_split(df: pd.DataFrame, frac=0.2, seed=0) -> pd.Series:
    rng = random.Random(seed)
    groups = df.groupby("group").size().to_dict()
    keys = sorted(groups); rng.shuffle(keys)
    target = frac * len(df); val, n = set(), 0
    for g in keys:
        if n >= target:
            break
        if n + groups[g] <= target * 1.15:
            val.add(g); n += groups[g]
    return df["group"].map(lambda g: "val" if g in val else "train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-synth", action="store_true")
    ap.add_argument("--no-pad", action="store_true")
    ap.add_argument("--skip-normalize", action="store_true")
    a = ap.parse_args()

    items = []  # (source, label, raw Path, group)
    gp = sorted((RAW / "gpt4o_receipt").glob("*.png"))
    for p in gp:
        items.append(("gpt4o_receipt", 1, p, gpt_group(p.name)))
    smoke = False
    if not gp:
        if not a.allow_synth:
            sys.exit("No GPT4o-Receipt images in data/raw/gpt4o_receipt/ (dataset is gated; see DATA.md). "
                     "Use --allow-synth only for a pipeline smoke test.")
        smoke = True
        for p in sorted((RAW / "_synth_smoke").glob("*.png")):
            items.append(("synth_smoke", 1, p, p.stem))
    for p in sorted((RAW / "cord_v2" / "images").glob("*.png")):
        items.append(("cord_v2", 0, p, p.stem))
    for it in json.load(open(ROOT / "data" / "commons_train_ok.json")):
        items.append(("commons", 0, ROOT / it["path"], "artist:" + (it["artist_norm"] or it["id"])))

    rows = []
    for src, lab, p, g in items:
        dst = NORM / src / (p.stem + ".jpg")
        if a.skip_normalize and dst.exists():
            with Image.open(dst) as _: pass
            with Image.open(p) as im:
                s = N.LONG / max(im.size); cw, ch = (N.LONG, round(im.size[1] * s)) if im.size[0] >= im.size[1] else (round(im.size[0] * s), N.LONG)
            m = dict(cw=cw, ch=ch)
        else:
            m = N.normalize_file(p, dst, not a.no_pad)
        st = raw_stats(p)
        rows.append(dict(path=f"{src}/{dst.name}", label=lab, source=src, group=g, cw=m["cw"], ch=m["ch"],
                         raw_path=str(p.relative_to(ROOT)), **st))
    df = pd.DataFrame(rows)
    df["split"] = ""
    for src, sub in df.groupby("source"):
        df.loc[sub.index, "split"] = group_split(sub, 0.2, seed=0).values
    out = ROOT / "data" / ("splits_smoke.csv" if smoke else "splits.csv")
    df[["path", "label", "source", "split", "group", "cw", "ch", "raw_path"]].to_csv(out, index=False)
    df.to_csv(ROOT / "data" / ("raw_stats_smoke.csv" if smoke else "raw_stats.csv"), index=False)
    print(df.groupby(["source", "split", "label"]).size().unstack(fill_value=0))
    print("wrote", out)


if __name__ == "__main__":
    main()
