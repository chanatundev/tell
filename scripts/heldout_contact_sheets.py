"""Build labelled contact sheets of candidate Commons receipt thumbnails.
usage: python -I heldout_contact_sheets.py <out_dir> <ids-file or comma list>
Reads tell/data/commons_raw/*.jpg (untrusted; only decoded as images)."""
import sys, os
from PIL import Image, ImageDraw
raw = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'commons_raw')
out = sys.argv[1]; ids = sys.argv[2].split(',')
os.makedirs(out, exist_ok=True)
COLS, ROWS, CW, CH = 5, 3, 330, 400
per = COLS*ROWS
for s in range(0, len(ids), per):
    sheet = Image.new('RGB', (COLS*CW, ROWS*CH), 'white'); d = ImageDraw.Draw(sheet)
    for k, i in enumerate(ids[s:s+per]):
        im = Image.open(f'{raw}/{i}.jpg').convert('RGB'); im.thumbnail((CW-6, CH-28))
        x, y = (k % COLS)*CW, (k//COLS)*CH
        sheet.paste(im, (x+3, y+26)); d.rectangle([x, y, x+CW-1, y+CH-1], outline='red')
        d.text((x+6, y+6), i, fill='red')
    sheet.save(f'{out}/sheet_{s//per:02d}.jpg', quality=82)
print(len(ids), 'ids', (len(ids)+per-1)//per, 'sheets')
