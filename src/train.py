"""Fine-tune EfficientNet-B0 on normalised images, calibrate on val, freeze threshold.

  python -I src/train.py --splits data/splits.csv --norm-dir data/norm --out runs/main
  python -I src/train.py --splits data/splits_smoke.csv --norm-dir data/norm --out runs/smoke --smoke

Writes: model.pt, frozen.json (threshold, temperature, timestamp, git hash, train counts), report.json, report.md.
Threshold = calibrated probability above which at most 5% of VAL REAL images are flagged.
NOTE: val is also used to pick the epoch, temperature and threshold, so val numbers are optimistic;
the sealed held-out set is the honest test.
"""
import argparse, json, hashlib, os, subprocess, sys, time, datetime, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tellmodel as TM
from metrics import summarize, threshold_at_fpr, fmt_prop, wilson

ROOT = Path(__file__).resolve().parents[1]


def git_hash():
    if os.environ.get("TELL_GIT_HASH"):
        return os.environ["TELL_GIT_HASH"]
    for p in (Path(__file__).resolve().parent / "GIT_HASH.txt", ROOT / "GIT_HASH.txt"):
        if p.exists():
            return p.read_text().strip()
    try:
        h = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "src"], stderr=subprocess.DEVNULL).decode().strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def fit_temperature(z, y):
    """Temperature scaling: minimise NLL of sigmoid(z/T) on val."""
    zt, yt = torch.tensor(z, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(); l = bce(zt / logT.exp(), yt); l.backward(); return l
    opt.step(closure)
    return float(logT.detach().exp().clamp(0.05, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--norm-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-far", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU/MPS run: 64 images, 1 epoch, 256 px")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.splits)
    if a.smoke:
        a.epochs, a.size, a.batch, a.workers = 1, 256, 16, 0
        parts = []
        for sp, n in (("train", 32), ("val", 16)):
            for lab in (0, 1):
                parts.append(df[(df.split == sp) & (df.label == lab)].sample(n // 2, random_state=0))
        df = pd.concat(parts)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    nonsq = df[(df.cw < 1024) | (df.ch < 1024)]
    pad_square = bool(len(nonsq)) and __import__('PIL.Image', fromlist=['Image']).open(Path(a.norm_dir) / nonsq.iloc[0]['path']).size == (1024, 1024)
    dev = TM.pick_device()
    print(f"device={dev} train={len(tr)} val={len(va)} size={a.size} epochs={a.epochs}")
    model = TM.make_model(pretrained=True).to(dev)
    if dev == "cuda":
        model = model.to(memory_format=torch.channels_last)
    kw = dict(num_workers=a.workers, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)
    dl_tr = torch.utils.data.DataLoader(TM.DS(tr, a.norm_dir, a.size, True), batch_size=a.batch, shuffle=True, drop_last=len(tr) > a.batch, **kw)
    dl_va = torch.utils.data.DataLoader(TM.DS(va, a.norm_dir, a.size, False), batch_size=max(4, a.batch // 2), shuffle=False, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    total = a.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))
    bce = nn.BCEWithLogitsLoss()
    best, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            if dev == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                loss = bce(model(x).squeeze(1).float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); n += len(y)
        z = TM.predict_logits(model, dl_va, dev)
        m = summarize(z, va.label.values, threshold_at_fpr(z[va.label.values == 0], a.max_far))
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, val_auroc=m["auroc"], secs=round(time.time() - t0)))
        print(hist[-1], flush=True)
        if not np.isnan(m["auroc"]) and m["auroc"] > best or best_state is None:
            best, best_state = m["auroc"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    # ---- calibration + threshold on VAL
    z = TM.predict_logits(model, dl_va, dev)
    y = va.label.values
    T = fit_temperature(z, y)
    p = expit(z / T)
    thr = threshold_at_fpr(p[y == 0], a.max_far)
    m = summarize(p, y, thr)
    nll = float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))

    torch.save(best_state, out / "model.pt")
    sha = hashlib.sha256((out / "model.pt").read_bytes()).hexdigest()
    frozen = dict(threshold=thr, temperature=T, timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  git_hash=git_hash(), model_sha256=sha, input_size=a.size, arch="efficientnet_b0",
                  pretrained_weights_loaded=bool(getattr(model, "pretrained_loaded", False)), max_val_far=a.max_far,
                  train_counts=dict(n=int(len(tr)), ai=int((tr.label == 1).sum()), real=int((tr.label == 0).sum()),
                                    by_source={k: int(v) for k, v in tr.source.value_counts().items()}),
                  val_counts=dict(n=int(len(va)), ai=int((va.label == 1).sum()), real=int((va.label == 0).sum())),
                  smoke_test=bool(a.smoke), pad_square=pad_square, epochs=a.epochs, splits_file=str(a.splits))
    json.dump(frozen, open(out / "frozen.json", "w"), indent=1)

    per = {}
    for s in va.source.unique():
        mk = (va.source == s).values
        k, n_ = int((p[mk] > thr).sum()), int(mk.sum())
        per[s] = dict(n=n_, flagged=k, rate=k / max(n_, 1), ci=wilson(k, n_), kind="recall" if va.label[mk].iloc[0] == 1 else "false-alarm")
    rep = dict(val=m, nll=nll, temperature=T, threshold=thr, per_source=per, history=hist, best_val_auroc_epoch_select=best)
    json.dump(rep, open(out / "report.json", "w"), indent=1, default=float)
    md = [f"# Validation report ({'SMOKE TEST - not a result' if a.smoke else 'main run'})", "",
          f"- val AUROC: **{m['auroc']:.3f}**",
          f"- recall at threshold {thr:.4f}: {fmt_prop(m['tp'], m['n_ai'])}",
          f"- false-alarm rate on val reals: {fmt_prop(m['fp'], m['n_real'])}",
          f"- temperature {T:.3f}, val NLL {nll:.3f}", "", "| source | kind | flagged / n | rate (Wilson 95%) |", "|---|---|---|---|"]
    for s, d in per.items():
        md.append(f"| {s} | {d['kind']} | {d['flagged']}/{d['n']} | {d['rate']:.3f} ({d['ci'][0]:.3f}-{d['ci'][1]:.3f}) |")
    md += ["", "Caveat: epoch, temperature and threshold were chosen on this same val set, so these numbers are optimistic.",
           "This project is a prototype and is not financial advice."]
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md)); print("saved ->", out, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
