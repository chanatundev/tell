"""Build tell/heldout_sealed/real/, real_selected.json and README.md from heldout_selection_data.py.
Run:  python -I scripts/heldout_build_real.py   (cwd = tell/ ; -I means scripts dir is NOT on sys.path,
so we add it explicitly below -- the data module is our own code, not downloaded data)."""
import json, re, html, shutil, os, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from heldout_selection_data import TEST, SPARE
ROOT = os.path.dirname(HERE)
OUT = f'{ROOT}/heldout_sealed'; os.makedirs(f'{OUT}/real', exist_ok=True)
meta = {x['id']: x for x in json.load(open(f'{ROOT}/commons_filtered.json'))}

def artist(a):
    txt = re.sub(r'<[^>]+>', '', html.unescape(a or '')).strip()
    txt = re.sub(r'\s+', ' ', txt)
    m = re.search(r'User:([^&"\s>]+)', html.unescape(a or ''))
    if (not txt or len(txt) < 2 or '<' in (a or '')[-3:]) and m: txt = m.group(1).replace('_', ' ')
    if m and ('action=edit' in (a or '') and not re.search(r'>[^<]{2,}<', a or '')): txt = m.group(1).replace('_', ' ')
    return txt

# commons_filtered.json truncates the artist HTML at 120 chars; fix the cut-off ones by hand.
# The Commons file page (commons_url) is the authoritative attribution record.
ARTIST_FIX = {
  **{i: 'Donald Trung Quoc Don (Chữ Hán: 徵國單) [name truncated in metadata; see file page]' for i in ('r0283','r0303','r0388','r0123','r0198','r0128','r0446')},
  'r0746': 'Emited by Becex Trade s.r.o.; scanned by ŠJů',
  'r0024': 'User EVOW Sahgyune Paiem KAASEI … [name truncated in metadata; see file page]',
  'r0078': '[author name truncated in metadata; see file page]',
  'r0753': 'William J Sisti', 'r0604': 'Stéphane M. Grueso', 'r0692': 'Martin Lewison',
  'r0015': 'Kuruman (Flickr)', 'r0326': '芳蘭 徐芳蘭 (Flickr)', 'r0321': '芳蘭 徐芳蘭 (Flickr)',
  'r0157': 'Guw Foiursanm Longum [name possibly truncated; see file page]',
}
recs = []
for split, rows in (('test', TEST), ('spare', SPARE)):
    for i, country, lang, shop, cap, blur, prompt in rows:
        m = meta[i]; title = m['title'].replace('File:', '', 1)
        wc = len(prompt.split())
        assert 60 <= wc <= 120, (i, wc)
        assert 'Jan 26' not in prompt and '26.01.2025' not in prompt and '1/26/2025' not in prompt
        shutil.copyfile(f'{ROOT}/data/commons_raw/{i}.jpg', f'{OUT}/real/{i}.jpg')
        recs.append(dict(id=i, title=title,
            commons_url='https://commons.wikimedia.org/wiki/File:' + title.replace(' ', '_'),
            license=m['license'], artist=ARTIST_FIX.get(i) or artist(m['artist']), country=country, language_script=lang,
            shop_type=shop, capture=cap,
            needs_blur=(False if blur is False else blur), split=split, twin_prompt=prompt))
assert len(recs) == 60 and sum(r['split']=='test' for r in recs) == 50 and len({r['id'] for r in recs}) == 60
json.dump(recs, open(f'{OUT}/real_selected.json', 'w'), ensure_ascii=False, indent=2)
print('wrote', len(recs))
for k in ('license', 'country', 'capture'):
    print(k, dict(collections.Counter(r[k] for r in recs)))
print('test only', dict(collections.Counter(r['country'] for r in recs if r['split']=='test')))
print('blur', sum(1 for r in recs if r['needs_blur'] is not False))
print('artists', sorted({r['artist'] for r in recs}))

# ---------------- README ----------------
lic = collections.Counter(r['license'] for r in recs)
lic_t = collections.Counter(r['license'] for r in recs if r['split']=='test')
cty = collections.Counter(r['country'] for r in recs if r['split']=='test')
cap = collections.Counter(r['capture'] for r in recs if r['split']=='test')
shop = collections.Counter(r['shop_type'].split(' / ')[0].split(' (')[0] for r in recs if r['split']=='test')
blur = [r['id'] for r in recs if r['needs_blur'] is not False]
def tab(c): return '\n'.join(f'| {k} | {v} |' for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
readme = f"""# Tell - sealed REAL held-out set

> **SEALED: the model builder should not view this folder until the model and threshold are frozen.**

## What this is
The REAL half of the held-out test set for "Tell" (AI-generated receipt-image detector). It holds
**60 genuine receipt photos/scans from Wikimedia Commons**: 50 `test` + 10 `spare` (replacements if a twin fails or an
image is later rejected). Each real image gets an AI "twin" generated from the `twin_prompt` written in
`real_selected.json`. Twins are matched on country, shop type and photo style (phone photo vs flat scan, paper condition,
framing), so the detector cannot win by spotting style differences.

* `real/rXXXX.jpg` - the 60 images (1024px-wide Commons thumbnails, names as in `data/commons_raw/`).
* `real_selected.json` - one record per image: id, title, commons_url, license, artist, country, language_script,
  shop_type, capture, needs_blur, split, twin_prompt.
* Built by `scripts/heldout_build_real.py` from `scripts/heldout_selection_data.py` (selection + prompts); contact-sheet
  helper: `scripts/heldout_contact_sheets.py`.

## Selection criteria
Included: modern (2008+) printed shop / restaurant / cafe / supermarket / pharmacy / fuel-station receipts, shown whole or
nearly whole and legible, as a photo or scan of the receipt itself. Excluded: tickets, invoices (hand-written or
formal), handwritten notes, historical documents, receipts tiny within a scene, collages, any visible face, any unredacted
full card number or clearly visible customer full name. Receipts with no personal data were preferred.
Diversity: 30 countries/territories, many scripts (Latin, Cyrillic, Hangul, Hanzi/kana, Thai, Hebrew), mix of phone photos and flat
scans, mix of shop types, some crumpled/faded. At most 6 per country (Japan), mostly 1-4; no two from the same shop+day.
The `twin_prompt` texts use only INVENTED shop names, addresses, tax ids and items (nothing copied from the real receipt).

Test split: {len([r for r in recs if r['split']=='test'])} images - capture: {dict(cap)}.

### Test split by country
| Country | n |
|---|---|
{tab(cty)}

### Test split by shop type
| Shop type | n |
|---|---|
{tab(shop)}

## Licences (all 60 images; test-only in brackets)
| Licence | n (test) |
|---|---|
""" + '\n'.join(f'| {k} | {v} ({lic_t.get(k,0)}) |' for k, v in sorted(lic.items(), key=lambda kv: (-kv[1], kv[0]))) + f"""

All licences are permissive/free (CC0 and public domain need no attribution; CC BY / CC BY-SA require it).
**Attribution requirement:** in the project write-up cite each used file's **artist and licence** (fields `artist`,
`license`, link `commons_url` in `real_selected.json`) - e.g. a table "Image id - Commons file page - artist - licence".
For CC BY-SA files (the majority) state the licence and link to it; derivative/redistributed copies of those images must carry the
same licence. Note: `commons_filtered.json` cuts the artist HTML at 120 characters, so a few `artist` values were completed or marked
"truncated; see file page" - the Commons file page is the authoritative record, so check it when writing the credits.

## Privacy notes
No faces. Where a receipt shows a partial (masked / last-4) card number, a staff or sole-trader name, a prescription number
or a partly shown loyalty number, `needs_blur` says what to blur before showing the real image publicly (ids: {', '.join(blur)}).
No unredacted full card numbers or customer names were found in the chosen files (re-check before publishing).
"""
open(f'{OUT}/README.md', 'w').write(readme)
print('readme written')
