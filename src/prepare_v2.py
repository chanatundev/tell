"""Extend the saved v1 split without modifying it or reading sealed photos."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import normalize as N
from prepare import group_split

ROOT = Path(__file__).resolve().parents[1]

def main():
    base = pd.read_csv(ROOT / 'data/splits.csv')
    rows = []
    wild = ROOT / 'data/raw/wildreceipt_dl/wildreceipt'
    # The upstream OCR test partition is validation here, never a new heldout test.
    for name, split in [('train', 'train'), ('test', 'val')]:
        for line in (wild / (name + '.txt')).read_text().splitlines():
            rec = json.loads(line)
            p = wild / rec['file_name']
            rows.append(dict(source='wildreceipt', label=0, raw_path=str(p.relative_to(ROOT)),
                             group='wild:' + p.stem, split=split))
    rejected = []
    fake = ROOT / 'data/raw/newgen_fakes'
    for p in sorted(fake.glob('n*.png')):
        log = fake / 'logs' / (p.stem + '.json')
        r = json.loads(log.read_text()) if log.exists() else {}
        note = r.get('note', '').lower()
        if not r.get('ok') or any(s in note for s in ['contains real', 'includes real', 'requirement is not met', 'does not meet']):
            rejected.append(p.name)
            continue
        rows.append(dict(source='newgen_fakes', label=1, raw_path=str(p.relative_to(ROOT)),
                         group='newgen:' + p.stem, split=''))
    extra = pd.DataFrame(rows)
    ids = extra.source == 'newgen_fakes'
    if ids.any():
        extra.loc[ids, 'split'] = group_split(extra[ids], seed=2026).values
    seen = {}
    for idx, r in extra.iterrows():
        p = ROOT / r.raw_path
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate new image: {p} and {seen[digest]}')
        seen[digest] = str(p)
        dst = ROOT / 'data/norm' / r.source / (p.stem + '.jpg')
        if dst.exists():
            # Existing normalized pixels may be in use by the interim training loader.
            with Image.open(p) as im:
                _, (cw, ch) = N.normalize_image(im)
            m = dict(cw=cw, ch=ch)
        else:
            m = N.normalize_file(p, dst)
        extra.loc[idx, 'path'] = str(dst.relative_to(ROOT / 'data/norm'))
        extra.loc[idx, 'cw'], extra.loc[idx, 'ch'] = m['cw'], m['ch']
    full = pd.concat([base, extra[base.columns]], ignore_index=True)
    assert not full.path.duplicated().any()
    assert full.groupby(['source', 'group']).split.nunique().max() == 1
    assert not full.raw_path.str.contains('heldout_sealed').any()
    pd.testing.assert_frame_equal(full.iloc[:len(base)].reset_index(drop=True), base, check_dtype=False)
    full.to_csv(ROOT / 'data/splits_v2.csv', index=False)
    counts = full.groupby(['source', 'split']).size()
    print(counts.to_string(), flush=True)
    print('Excluded new fakes:', rejected, flush=True)
    (ROOT / 'data/v2_manifest.json').write_text(json.dumps(dict(
        evaluation='post-unseal development; not fresh heldout',
        base_split_sha256=hashlib.sha256((ROOT / 'data/splits.csv').read_bytes()).hexdigest(),
        rejected_fakes=rejected, counts={str(k): int(v) for k,v in counts.items()},
        new_image_sha256=seen), indent=2))

if __name__ == '__main__':
    main()
