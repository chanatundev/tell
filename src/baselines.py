"""Baselines for Tell. All (b)/(c) see only the NORMALISED images, same as the model.

 (a) "no camera EXIF" rule  - on raw files vs on normalised files (stripped => chance by construction)
 (b) logistic regression on FFT radial power spectrum + pixel-intensity stats
 (c) bookkeeper's checklist: Tesseract OCR + rules (sum mismatch, "12 50" prices, no total, phash duplicate)

Importable by eval_heldout.py:  freq_features(), fit_freq(), run_checklist(), checklist_score()
CLI:  python -I src/baselines.py --splits data/splits.csv --stats data/raw_stats.csv

Tesseract needs the tesseract binary on PATH.
If the binary is missing, (c) is skipped and reported as SKIPPED.  Only English traineddata is assumed,
so non-Latin scripts mostly fail to parse - that is reported as coverage, not hidden.
"""
import argparse, json, os, re, sys, shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from PIL import Image
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import normalize as N
from metrics import summarize, threshold_at_fpr, fmt_prop

# ---------------------------------------------------------------- (b) frequency features
NB = 24


def load_content(path, cw, ch, pad_square=True):
    im = Image.open(path).convert("RGB")
    if im.size == (N.LONG, N.LONG) and (cw, ch) != im.size:
        im = im.crop(N.content_box(cw, ch, N.LONG))
    return im


def freq_features(path, cw, ch):
    im = load_content(path, cw, ch).convert("L")
    w, h = im.size
    s = min(512, w, h)
    l, t = (w - s) // 2, (h - s) // 2
    a = np.asarray(im.crop((l, t, l + s, t + s)), dtype=np.float64)
    win = np.outer(np.hanning(s), np.hanning(s))
    F = np.fft.fftshift(np.fft.fft2((a - a.mean()) * win))
    P = np.log1p(np.abs(F) ** 2)
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - s / 2, xx - s / 2) / (s / 2)  # 0..~1.41, 1 = Nyquist
    edges = np.linspace(0, 1.0, NB + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, NB)
    keep = idx < NB
    prof = np.bincount(idx[keep], weights=P.ravel()[keep], minlength=NB) / np.maximum(np.bincount(idx[keep], minlength=NB), 1)
    v = a.ravel()
    gx, gy = np.abs(np.diff(a, axis=1)).mean(), np.abs(np.diff(a, axis=0)).mean()
    ints = [v.mean(), v.std(), sps.skew(v), sps.kurtosis(v), *np.percentile(v, [1, 5, 50, 95, 99]), gx, gy]
    return np.concatenate([prof, ints])


def feats_for(df, norm_dir, workers=8):
    args = [(str(norm_dir / r.path), int(r.cw), int(r.ch)) for r in df.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(_ff, args, chunksize=8)))


def _ff(a):
    return freq_features(*a)


def fit_freq(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"))
    return clf.fit(X, y)


# ---------------------------------------------------------------- (c) bookkeeper's checklist
PRICE_DEC = re.compile(r"(?<![\d.,])(\d{1,6}[.,]\d{2})(?![\d])")
PRICE_THOU = re.compile(r"(?<![\d.,])(\d{1,3}(?:[.,]\d{3})+)(?![\d])")
SPACE_PRICE = re.compile(r"(?:[$€£¥]\s?|\b)(\d{1,4}) (\d{2})\s*$")
RE_TOTAL = re.compile(r"\b(grand\s*total|total\s*due|amount\s*due|balance\s*due|total|jumlah|totaal|summe|importe|sum)\b", re.I)
RE_SUBTOTAL = re.compile(r"sub\s*-?\s*total", re.I)
RE_TAX = re.compile(r"\b(tax|ppn|vat|gst|hst|pajak|service|serv\.?\s*chg|mwst)\b", re.I)
RE_OTHER = re.compile(r"\b(cash|change|visa|master\s*card|mastercard|amex|card|tendered|payment|debit|credit|tip|kembali|tunai|bayar|discount|disc|saving|points?|balance|round)\b", re.I)


def parse_price(tok):
    t = tok.strip()
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", t):  # thousands grouping, e.g. 45.000 / 1,234
        return float(re.sub(r"[.,]", "", t))
    return float(t.replace(",", "."))


def line_price(line):
    ms = list(PRICE_DEC.finditer(line)) or list(PRICE_THOU.finditer(line))
    if not ms:
        return None
    return parse_price(ms[-1].group(1))


def tesseract_ok():
    try:
        import pytesseract
        return shutil.which("tesseract") is not None and bool(pytesseract.get_tesseract_version())
    except Exception:
        return False


def ocr(path, cw, ch, cache_dir=None):
    cp = None
    if cache_dir is not None:
        cp = Path(cache_dir) / (str(path).replace("/", "__") + ".txt")
        if cp.exists():
            return cp.read_text()
    import pytesseract
    os.environ["OMP_THREAD_LIMIT"] = "1"
    txt = pytesseract.image_to_string(load_content(path, cw, ch), config="--psm 6")
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(txt)
    return txt


def analyse_text(txt, tol=0.02):
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    items, taxes, total, subtotal = [], [], None, None
    seen_total = False
    for l in lines:
        p = line_price(l)
        if p is None:
            continue
        if RE_SUBTOTAL.search(l):
            subtotal = p if subtotal is None else subtotal
        elif RE_TOTAL.search(l) and not RE_OTHER.search(l.replace("total", "")):
            if total is None:
                total = p
            seen_total = True
        elif RE_TAX.search(l):
            taxes.append(p)
        elif RE_OTHER.search(l) or seen_total:
            continue
        else:
            items.append(p)
    space = any(SPACE_PRICE.search(l) for l in lines)
    parsed = total is not None and len(items) >= 2
    mismatch = False
    if parsed:
        s, t, tx = sum(items), total, sum(taxes)
        tol_abs = tol if total < 1000 else 1.0  # integer currencies (IDR etc.): allow rounding
        if subtotal is not None:
            ok = abs(s - subtotal) <= tol_abs and (abs(subtotal - t) <= tol_abs or abs(subtotal + tx - t) <= tol_abs)
        else:
            ok = abs(s - t) <= tol_abs or abs(s + tx - t) <= tol_abs
        mismatch = not ok
    return dict(no_total=total is None, sum_mismatch=mismatch, space_price=space, parsed=parsed,
                n_items=len(items), has_text=len(lines) >= 3)


def _ocr_job(a):
    return analyse_text(ocr(*a))


def run_checklist(df, norm_dir, cache_dir, workers=6, phash_dist=6):
    """df needs columns path, cw, ch. Returns DataFrame of rule flags (aligned to df.index)."""
    import imagehash
    args = [(str(norm_dir / r.path), int(r.cw), int(r.ch), str(cache_dir)) for r in df.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_ocr_job, args, chunksize=4))
    out = pd.DataFrame(res, index=df.index)
    hs = [imagehash.phash(load_content(a[0], a[1], a[2])) for a in args]
    dup = np.zeros(len(hs), bool)
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            if hs[i] - hs[j] <= phash_dist:
                dup[i] = dup[j] = True
    out["near_dup"] = dup
    return out


def checklist_score(f, strict=False):
    """Number of rules triggered. Lenient (default) ignores 'no total found' (it fires on every non-receipt)."""
    s = f.sum_mismatch.astype(int) + f.space_price.astype(int) + f.near_dup.astype(int)
    return s + (f.no_total.astype(int) if strict else 0)


# ---------------------------------------------------------------- CLI
def row(name, m):
    return (f"| {name} | {m['auroc']:.3f} | {m['recall']:.3f} ({m['recall_ci'][0]:.2f}-{m['recall_ci'][1]:.2f}) "
            f"| {m['far']:.3f} ({m['far_ci'][0]:.2f}-{m['far_ci'][1]:.2f}) |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits.csv"))
    ap.add_argument("--stats", default=str(ROOT / "data" / "raw_stats.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "baselines_val.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--norm-dir", default=str(ROOT / "data" / "norm"))
    ap.add_argument("--ocr-cache", default=str(ROOT / "data" / "ocr_cache"))
    a = ap.parse_args()
    norm_dir = Path(a.norm_dir)
    have_raw = Path(a.stats).exists()  # without raw-file stats, baseline (a) on raw files is skipped
    st = pd.read_csv(a.stats if have_raw else a.splits)
    tr, va = st[st.split == "train"], st[st.split == "val"]
    res, lines = {}, ["| method | AUROC | recall (Wilson) | false-alarm (Wilson) |", "|---|---|---|---|"]

    # (a) no-camera-EXIF rule: flag if no EXIF. raw vs normalised (all stripped => flags everything)
    y = va.label.values
    for name, sc in [("(a) no-EXIF rule, RAW files", 1 - va.raw_has_camera_exif.values if have_raw else None),
                     ("(a) no-EXIF rule, NORMALISED files", np.ones(len(va)))]:
        if sc is None:
            lines.append(f"| {name} | (raw stats not available here) | | |"); continue
        m = summarize(sc, y, 0.5); res[name] = m; lines.append(row(name, m))

    # (b) frequency LR
    Xtr, Xva = feats_for(tr, norm_dir), feats_for(va, norm_dir)
    clf = fit_freq(Xtr, tr.label.values)
    s = clf.decision_function(Xva)
    thr = threshold_at_fpr(s[y == 0], 0.05)
    m = summarize(s, y, thr); res["(b) freq LR @5% val-FAR"] = m; lines.append(row("(b) FFT+intensity LR, thr@5% val FAR", m))
    per = {}
    for src in va.source.unique():
        mk = (va.source == src).values
        per[src] = dict(n=int(mk.sum()), flagged=int((s[mk] > thr).sum()))
    res["(b) per-source flagged"] = per

    # (c) checklist
    if tesseract_ok():
        f = run_checklist(va, norm_dir, Path(a.ocr_cache), a.workers)
        cov = dict(parsed=float(f.parsed.mean()), has_text=float(f.has_text.mean()))
        cov["by_source"] = {src: dict(n=int(m.sum()), parsed=float(f.parsed[m].mean()), has_text=float(f.has_text[m].mean()),
                                      no_total=float(f.no_total[m].mean()))
                            for src in va.source.unique() for m in [(va.source == src).values]}
        res["(c) coverage"] = cov
        for strict in (False, True):
            sc = checklist_score(f, strict).values
            nm = f"(c) checklist {'strict (+no-total)' if strict else 'lenient'}, flag if >=1 rule"
            m = summarize(sc, y, 0.5); res[nm] = m; lines.append(row(nm, m))
        res["(c) rule flag rates by class"] = {c: {k: float(f[k][y == c].mean()) for k in
                                               ["no_total", "sum_mismatch", "space_price", "near_dup", "parsed"]} for c in (0, 1)}
    else:
        res["(c)"] = "SKIPPED: tesseract binary not found"
        lines.append("| (c) checklist | SKIPPED (no tesseract) | | |")
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print("\n".join(lines)); print(json.dumps({k: v for k, v in res.items() if k.startswith("(c) c") or k.startswith("(c) r") or "per-source" in k}, indent=1))


if __name__ == "__main__":
    main()
