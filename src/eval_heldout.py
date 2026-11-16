"""SEALED held-out evaluation. Run ONCE, after model.pt + frozen.json are frozen and committed.

  python src/eval_heldout.py --unseal --model runs/main/model.pt --frozen runs/main/frozen.json \
        --splits data/splits.csv --norm-dir data/norm --val-report runs/main/report.json

Refuses to start unless --unseal is given AND model + frozen files exist and match (sha256).
Uses only records with split == "test" in heldout_sealed/real_selected.json (spares are ignored).
Expected layout: heldout_sealed/real/<id>.jpg and heldout_sealed/ai/<id>_twin.<ext> (or <id>.<ext>)  (the twin of real <id>).

Writes results/: heldout_results.md/.json, heldout_pairs.csv, examples/ + examples.md
(it copies example files and writes a table; it never displays any image).
Nothing here is used for training, tuning or thresholds: threshold and temperature come from frozen.json.
"""
import argparse, hashlib, json, shutil, subprocess, sys, datetime
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def refuse(msg):
    sys.exit(f"REFUSING TO RUN: {msg}\nThe held-out set is sealed until the model and threshold are frozen and committed. "
             "Run with: --unseal --model model.pt --frozen frozen.json")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseal", action="store_true", help="explicit confirmation that the model + threshold are frozen")
    ap.add_argument("--model"); ap.add_argument("--frozen")
    ap.add_argument("--heldout", default=str(ROOT / "heldout_sealed"))
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits.csv"), help="training splits (to fit baseline b)")
    ap.add_argument("--norm-dir", default=str(ROOT / "data" / "norm"))
    ap.add_argument("--val-report", help="report.json from train.py (in-distribution row)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    # ---- guards run BEFORE anything under --heldout is touched
    if not a.unseal:
        refuse("--unseal not given.")
    if not a.model or not a.frozen:
        refuse("--model and --frozen are both required.")
    mp, fp = Path(a.model), Path(a.frozen)
    if not mp.is_file() or not fp.is_file():
        refuse("model or frozen file not found.")
    fz = json.load(open(fp))
    for k in ("threshold", "temperature", "timestamp", "git_hash", "model_sha256"):
        if k not in fz:
            refuse(f"frozen.json lacks '{k}'.")
    if hashlib.sha256(mp.read_bytes()).hexdigest() != fz["model_sha256"]:
        refuse("model.pt does not match the sha256 recorded in frozen.json.")
    if fz.get("smoke_test"):
        refuse("frozen.json comes from a smoke test, not a real run.")
    return a, fz


def committed(path):
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-1", "--", str(Path(path).resolve())],
                           capture_output=True, text=True)
        return bool(r.stdout.strip())
    except Exception:
        return False


def main():
    a, fz = parse()
    import torch
    import normalize as N, tellmodel as TM, baselines as B
    from metrics import wilson, bootstrap_ci, fmt_prop, threshold_at_fpr, summarize
    from sklearn.metrics import roc_auc_score
    out = Path(a.out); (out / "examples").mkdir(parents=True, exist_ok=True)
    held = Path(a.heldout)
    meta = {r["id"]: r for r in json.load(open(held / "real_selected.json")) if r.get("split") == "test"}
    # ---- collect pairs
    pairs = []
    for i, r in sorted(meta.items()):
        real = held / "real" / f"{i}.jpg"
        twins = [p for p in list((held / "ai").glob(f"{i}_twin.*")) + list((held / "ai").glob(f"{i}.*")) if p.suffix.lower() in N.EXTS]
        if real.exists() and twins:
            pairs.append(dict(id=i, real=real, twin=twins[0], country=r.get("country", "?"), capture=r.get("capture", "?"),
                              script=r.get("language_script", "?")))
    n = len(pairs)
    print(f"test pairs found: {n} (of {len(meta)} test records)")
    if n == 0:
        sys.exit("no test pairs found")
    # ---- normalise with the SAME normalize.py
    pad = fz.get("pad_square", True)
    rows = []
    for p in pairs:
        for kind in ("real", "twin"):
            dst = out / "heldout_norm" / kind / f"{p['id']}.jpg"
            m = N.normalize_file(p[kind], dst, pad)
            rows.append(dict(id=p["id"], kind=kind, label=int(kind == "twin"), path=str(dst), cw=m["cw"], ch=m["ch"]))
    df = pd.DataFrame(rows)
    # ---- Tell
    dev = TM.pick_device()
    model = TM.make_model(pretrained=False)
    model.load_state_dict(torch.load(a.model, map_location="cpu")); model.to(dev).eval()
    dfp = df.assign(path=[str(Path(x).resolve()) for x in df.path])  # absolute -> norm_dir join is a no-op
    prob, _ = TM.score_files(model, dfp, "/", fz["input_size"], dev, fz["temperature"])
    thr = fz["threshold"]
    df["tell"] = prob
    pv = df.pivot(index="id", columns="kind", values="tell").loc[[p["id"] for p in pairs]]

    def paired(real_s, twin_s):
        w = (twin_s > real_s).astype(float) + 0.5 * (twin_s == real_s)
        return float(w.mean()), bootstrap_ci(w)

    res = {"n_pairs": n, "frozen": {k: fz[k] for k in ("threshold", "temperature", "timestamp", "git_hash", "model_sha256")},
           "frozen_json_committed_in_git": committed(a.frozen)}
    if not res["frozen_json_committed_in_git"]:
        print("WARNING: frozen.json is not committed in git; commit it BEFORE looking at results.")
    acc, ci = paired(pv["real"].values, pv["twin"].values)
    k_tw, k_re = int((pv["twin"] > thr).sum()), int((pv["real"] > thr).sum())
    res["tell"] = dict(paired_acc=acc, paired_ci=ci, recall_twins=(k_tw, n, wilson(k_tw, n)), false_alarm_reals=(k_re, n, wilson(k_re, n)),
                       auroc=float(roc_auc_score(df.label, df.tell)))

    # ---- baseline (b): frequency LR trained on the training split only, threshold at 5% val FAR
    bres = {}
    try:
        sp = pd.read_csv(a.splits)
        tr, va = sp[sp.split == "train"], sp[sp.split == "val"]
        clf = B.fit_freq(B.feats_for(tr, Path(a.norm_dir), a.workers), tr.label.values)
        sv = clf.decision_function(B.feats_for(va, Path(a.norm_dir), a.workers))
        thr_b = threshold_at_fpr(sv[va.label.values == 0], 0.05)
        hb = df.assign(path=[str(Path(x).resolve()) for x in df.path])
        sb = clf.decision_function(np.stack([B.freq_features(r.path, r.cw, r.ch) for r in hb.itertuples()]))
        df["freq_lr"] = sb
        pvb = df.pivot(index="id", columns="kind", values="freq_lr").loc[pv.index]
        acc_b, ci_b = paired(pvb["real"].values, pvb["twin"].values)
        kt, kr = int((pvb["twin"] > thr_b).sum()), int((pvb["real"] > thr_b).sum())
        bres["freq_lr"] = dict(paired_acc=acc_b, paired_ci=ci_b, recall_twins=(kt, n, wilson(kt, n)), false_alarm_reals=(kr, n, wilson(kr, n)),
                               auroc=float(roc_auc_score(df.label, df.freq_lr)))
    except Exception as e:
        bres["freq_lr"] = f"SKIPPED: {e!r}"
    # ---- baseline (c): checklist
    if B.tesseract_ok():
        f = B.run_checklist(dfp, Path("/"), out / "ocr_cache", a.workers)
        for nm, strict in (("checklist_lenient", False), ("checklist_strict", True)):
            df[nm] = B.checklist_score(f, strict).values
            pvc = df.pivot(index="id", columns="kind", values=nm).loc[pv.index]
            acc_c, ci_c = paired(pvc["real"].values, pvc["twin"].values)
            kt, kr = int((pvc["twin"] >= 1).sum()), int((pvc["real"] >= 1).sum())
            bres[nm] = dict(paired_acc=acc_c, paired_ci=ci_c, recall_twins=(kt, n, wilson(kt, n)), false_alarm_reals=(kr, n, wilson(kr, n)),
                            auroc=float(roc_auc_score(df.label, df[nm])))
        bres["checklist_coverage"] = dict(parsed_twins=float(f.parsed[df.kind == "twin"].mean()), parsed_reals=float(f.parsed[df.kind == "real"].mean()))
    else:
        bres["checklist"] = "SKIPPED: tesseract binary not found"
    res["baselines"] = bres

    # ---- group table (false alarms on REALS)
    pr = pd.DataFrame(pairs).set_index("id").loc[pv.index]
    pr["fa"] = (pv["real"].values > thr)
    groups = {}
    for col in ("country", "capture", "script"):
        t = pr.groupby(col)["fa"].agg(["sum", "count"]).reset_index()
        groups[col] = [dict(group=r[col], false_alarms=int(r["sum"]), n=int(r["count"]), ci=wilson(int(r["sum"]), int(r["count"]))) for _, r in t.iterrows()]
    res["groups"] = groups

    # ---- examples: 3 most confident misses (lowest-scoring twins) and 3 most confident false alarms (highest-scoring reals)
    miss = pv["twin"].sort_values().head(3)
    fa = pv["real"].sort_values(ascending=False).head(3)
    md_ex = ["| kind | id | score | country | capture | file |", "|---|---|---|---|---|---|"]
    for kind, ser, key in (("miss (AI twin scored low)", miss, "twin"), ("false alarm (real scored high)", fa, "real")):
        for i, s in ser.items():
            dst = out / "examples" / f"{'miss' if key == 'twin' else 'falsealarm'}_{i}_{key}.jpg"
            shutil.copy(df[(df.id == i) & (df.kind == key)].path.iloc[0], dst)
            md_ex.append(f"| {kind} | {i} | {s:.3f} | {pr.loc[i, 'country']} | {pr.loc[i, 'capture']} | {dst.relative_to(out)} |")
    (out / "examples.md").write_text("\n".join(md_ex))
    pd.DataFrame({"id": pv.index, "p_real": pv["real"].values, "p_twin": pv["twin"].values, "country": pr.country.values,
                  "capture": pr.capture.values}).to_csv(out / "heldout_pairs.csv", index=False)

    # ---- report
    def line(name, d):
        if isinstance(d, str):
            return f"| {name} | {d} | | | |"
        r, f_ = d["recall_twins"], d["false_alarm_reals"]
        return (f"| {name} | {d['paired_acc']:.3f} ({d['paired_ci'][0]:.2f}-{d['paired_ci'][1]:.2f}) | {r[0]}/{r[1]} = {r[0]/r[1]:.2f} ({r[2][0]:.2f}-{r[2][1]:.2f}) "
                f"| {f_[0]}/{f_[1]} = {f_[0]/f_[1]:.2f} ({f_[2][0]:.2f}-{f_[2][1]:.2f}) | {d['auroc']:.3f} |")
    md = [f"# Held-out twin evaluation ({datetime.datetime.now().isoformat(timespec='seconds')})", "",
          f"{n} test pairs (real Commons photo + AI twin from a newer generator). Model {fz['model_sha256'][:12]}, git {fz['git_hash']}, "
          f"frozen {fz['timestamp']}. frozen.json committed in git: {res['frozen_json_committed_in_git']}.", "",
          "| method | paired accuracy (bootstrap 95% CI) | recall on twins (Wilson) | false alarms on reals (Wilson) | AUROC (all images) |", "|---|---|---|---|---|",
          line("**Tell** (frozen threshold)", res["tell"])]
    for k, v in bres.items():
        if k != "checklist_coverage":
            md.append(line(k, v))
    if a.val_report and Path(a.val_report).exists():
        v = json.load(open(a.val_report))["val"]
        md += ["", f"In-distribution (GPT-4o val split): AUROC {v['auroc']:.3f}, recall {v['recall']:.3f}, false-alarm {v['far']:.3f} "
               "(val was also used to pick epoch/threshold, so optimistic)."]
    md += ["", "Paired accuracy = share of pairs where the twin scores strictly higher than its real partner (ties = 0.5); chance = 0.5. "
           "Baseline thresholds: freq LR flagged at its own 5%-val-false-alarm threshold; checklist flags at >= 1 rule.", ""]
    for col, rows_ in groups.items():
        md += [f"### False alarms on the {n} reals by {col}", "", "| group | false alarms / n | Wilson 95% |", "|---|---|---|"]
        md += [f"| {r['group']} | {r['false_alarms']}/{r['n']} | {r['ci'][0]:.2f}-{r['ci'][1]:.2f} |" for r in sorted(rows_, key=lambda r: -r["n"])]
        md.append("")
    md += ["### Most confident failures (files in results/examples/; not displayed here)", "", *md_ex, "",
           "**Caveats:** small, team-made set; synthetic twins are not real fraud; a flag is not proof of fraud; group rows are warning signs, "
           "not proof of unfairness. This project is a prototype and is not financial advice."]
    (out / "heldout_results.md").write_text("\n".join(md))
    json.dump(res, open(out / "heldout_results.json", "w"), indent=1, default=float)
    print("\n".join(md[:14])); print("wrote", out)


if __name__ == "__main__":
    main()
