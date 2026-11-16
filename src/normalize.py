"""Normalise EVERY image identically before anything (model, baselines, leak check) sees it.

Steps (same for train, val, held-out reals and held-out twins):
  1. open, apply EXIF orientation (so the picture is upright), convert to RGB
     (transparency composited on white)
  2. drop ALL metadata: EXIF, ICC, C2PA/JUMBF, PNG text chunks, dpi  (we re-encode from raw pixels)
  3. resize so the LONG side is exactly 1024 px (aspect kept; LANCZOS up or down)
  4. optional --pad-square (default ON): centre on a 1024x1024 neutral-grey canvas so width,
     height and aspect ratio are identical for every file (removes file-level size shortcuts).
     The real picture area ("content box") is recorded (cw, ch) so models can crop inside it
     and never see the padding.
  0. (before 1) images that were never JPEG (PNG etc.) get one JPEG pass at q70-95 at original
     size, so real and AI images share a 'compressed once before' history (see equalize_jpeg_history)
  5. encode JPEG quality 85, 4:2:0, baseline, no optimisation (fixed quantisation tables)

Importable: normalize_image(img) / normalize_file(src, dst) / content_box(cw, ch).
"""
import argparse, csv, io, sys, hashlib
from pathlib import Path
from PIL import Image, ImageOps

LONG = 1024
QUALITY = 85
PAD_COLOR = (128, 128, 128)
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _to_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def normalize_image(img: Image.Image, pad_square: bool = True):
    """Return (normalised PIL RGB image, (cw, ch)) where (cw, ch) is the content size."""
    img = _to_rgb(img)
    w, h = img.size
    s = LONG / max(w, h)
    cw, ch = max(1, round(w * s)), max(1, round(h * s))
    if w >= h:
        cw = LONG
    else:
        ch = LONG
    img = img.resize((cw, ch), Image.Resampling.LANCZOS)
    if pad_square:
        canvas = Image.new("RGB", (LONG, LONG), PAD_COLOR)
        canvas.paste(img, ((LONG - cw) // 2, (LONG - ch) // 2))
        img = canvas
    return img, (cw, ch)


def content_box(cw: int, ch: int, size: int = LONG):
    """Box of the real picture inside a padded size x size canvas (left, top, right, bottom)."""
    l, t = (size - cw) // 2, (size - ch) // 2
    return (l, t, l + cw, t + ch)


def encode_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    # no exif/icc/comment arguments -> nothing is written besides pixels + quant tables
    img.save(buf, "JPEG", quality=QUALITY, subsampling="4:2:0", optimize=False, progressive=False)
    return buf.getvalue()


def equalize_jpeg_history(img: Image.Image, fmt: str, key: str) -> Image.Image:
    """Give never-JPEG'd images (PNG/WebP/...: the AI fakes and twins) one JPEG compression at
    their ORIGINAL size, with a quality from 70-95 chosen deterministically from the file name.
    Real photos are already camera/upload JPEGs. Without this, 'has old JPEG blockiness' would
    separate real (JPEG) from AI (PNG) even after re-encoding: a format shortcut, not an AI tell.
    The rule depends only on the input FORMAT, never on the label, so it is applied identically to
    train, val and the held-out set."""
    if (fmt or "").upper() in ("JPEG", "MPO"):
        return img
    q = 70 + int(hashlib.md5(key.encode()).hexdigest(), 16) % 26
    buf = io.BytesIO()
    _to_rgb(img).save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def normalize_file(src: Path, dst: Path, pad_square: bool = True) -> dict:
    src, dst = Path(src), Path(dst)
    with Image.open(src) as im:
        ow, oh = im.size
        fmt = im.format
        im2 = equalize_jpeg_history(im, fmt, src.name)
        norm, (cw, ch) = normalize_image(im2, pad_square)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(encode_jpeg(norm))
    return dict(src=str(src), dst=str(dst), orig_w=ow, orig_h=oh, orig_format=fmt,
                orig_bytes=src.stat().st_size, cw=cw, ch=ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of raw images")
    ap.add_argument("--out", required=True, help="output folder (data/norm/<source>)")
    ap.add_argument("--no-pad", action="store_true", help="keep aspect ratio, no square canvas")
    ap.add_argument("--manifest", help="optional CSV with sizes of what was written")
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.src).iterdir()):
        if p.suffix.lower() in EXTS:
            rows.append(normalize_file(p, Path(a.out) / (p.stem + ".jpg"), not a.no_pad))
    print(f"normalised {len(rows)} images -> {a.out}")
    if a.manifest and rows:
        with open(a.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
