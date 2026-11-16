"""Post-unseal development evaluation using v1's cached normalized twins.
No training or threshold selection is performed here. Never writes v1 results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
import tellmodel as TM
from metrics import summarize, bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/v2')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/post_unseal_v2')
    args = parser.parse_args()
    if args.out.resolve() == (ROOT / 'results/heldout_v1').resolve():
        raise ValueError('Cannot overwrite original v1 evidence')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output must be new or empty; preserve earlier evaluations')
    fz = json.loads((args.run / 'frozen.json').read_text())
    assert not fz['smoke_test']
    assert hashlib.sha256((args.run / 'model.pt').read_bytes()).hexdigest() == fz['model_sha256']
    pairs = pd.read_csv(ROOT / 'results/heldout_v1/heldout_pairs.csv')
    rows = []
    # All normalized inputs already use the frozen v1 normalization implementation.
    from PIL import Image
    import normalize as N
    for ident in pairs.id:
        for kind in ['real', 'twin']:
            raw = ROOT / 'heldout_sealed' / ('real' if kind == 'real' else 'ai')
            candidates = list(raw.glob(ident + ('.jpg' if kind == 'real' else '_twin.*')))
            if not candidates and kind == 'twin':
                candidates = list(raw.glob(ident + '.*'))
            assert len(candidates) == 1, (ident, kind, candidates)
            with Image.open(candidates[0]) as im:
                _, (cw, ch) = N.normalize_image(im, fz['pad_square'])
            path = ROOT / 'results/heldout_v1/heldout_norm' / kind / (ident + '.jpg')
            assert path.is_file()
            rows.append(dict(id=ident, kind=kind, path=str(path), cw=cw, ch=ch, label=int(kind=='twin')))
    df = pd.DataFrame(rows)
    device = TM.pick_device()
    model = TM.make_model(False).to(device)
    model.load_state_dict(torch.load(args.run / 'model.pt', map_location=device, weights_only=True))
    prob, _ = TM.score_files(model, df, '/', fz['input_size'], device, fz['temperature'], batch=4)
    df['score'] = prob
    pv = df.pivot(index='id', columns='kind', values='score')
    wins = (pv.twin > pv.real).astype(float) + .5*(pv.twin == pv.real)
    report = dict(evaluation='POST-UNSEAL DEVELOPMENT — NOT FRESH HELDOUT',
                  model_sha256=fz['model_sha256'], n_pairs=len(pv),
                  paired_accuracy=float(wins.mean()), paired_ci=bootstrap_ci(wins.to_numpy()),
                  metrics=summarize(prob, df.label.to_numpy(), fz['threshold']), frozen=fz)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / 'scores.csv', index=False)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2))
    m=report['metrics']
    (args.out / 'report.md').write_text(
        '# V2 post-unseal development evaluation\n\nNOT a fresh heldout result. '
        'V1 results were known before v2 development.\n\n'
        f"Pairs: {len(pv)}; paired accuracy: {wins.mean():.3f}; bootstrap 95% CI: {report['paired_ci']}.\n\n"
        f"Detected twins: {m['tp']}/{m['n_ai']}; flagged reals: {m['fp']}/{m['n_real']}.\n")
    print((args.out / 'report.md').read_text())

if __name__ == '__main__':
    main()
