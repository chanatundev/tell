"""Make a SYNTHETIC STAND-IN for the AI-image class, for pipeline smoke tests ONLY.

Renders plain text receipts with PIL at fixed GPT-image-like sizes (PNG). They are NOT AI-generated
receipts and say nothing about Tell's accuracy. Used only when GPT4o-Receipt is unavailable
(it is gated). Some have wrong arithmetic or malformed prices so the checklist code is exercised.
Output: data/raw/_synth_smoke/*.png
"""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "_synth_smoke"
FONTS = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Courier.ttc",
         "/System/Library/Fonts/Monaco.ttf"]
ITEMS = ["MILK 2L", "BREAD", "EGGS 12", "APPLES", "COFFEE", "RICE 5KG", "SOAP", "BATTERIES", "PASTA", "TEA"]


def font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make(i, rng):
    W, H = rng.choice([(1024, 1536), (1536, 1024), (1024, 1024)])
    bg = Image.new("RGB", (W, H), (rng.randint(60, 120),) * 3)
    pw = int(min(W, H) * 0.7)
    paper = Image.new("RGB", (pw, int(H * 0.9)), (245, 243, 238))
    d = ImageDraw.Draw(paper)
    f = font(26)
    y = 30
    d.text((30, y), f"SHOP {rng.randint(1, 99):02d} MARKET", font=font(34), fill=(20, 20, 20)); y += 60
    n = rng.randint(4, 8)
    prices = [round(rng.uniform(1, 25), 2) for _ in range(n)]
    for name, p in zip(rng.sample(ITEMS, min(n, len(ITEMS))), prices):
        ps = f"{p:.2f}"
        if rng.random() < 0.08:
            ps = ps.replace(".", " ")
        d.text((30, y), f"{name:<14}{ps:>8}", font=f, fill=(20, 20, 20)); y += 36
    sub = round(sum(prices), 2)
    tax = round(sub * 0.0825, 2)
    total = round(sub + tax + (rng.choice([0, 0, 0, 1.5, -2.0]) if rng.random() < 0.3 else 0), 2)
    y += 10
    for lab, v in [("SUBTOTAL", sub), ("TAX 8.25%", tax), ("TOTAL", total)]:
        d.text((30, y), f"{lab:<14}{v:>8.2f}", font=f, fill=(20, 20, 20)); y += 36
    bg.paste(paper, ((W - pw) // 2, int(H * 0.05)))
    return bg.filter(ImageFilter.GaussianBlur(0.6))


def main(n=150):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for i in range(n):
        make(i, rng).save(OUT / f"synth_{i:04d}.png")
    print("wrote", n, "->", OUT)


if __name__ == "__main__":
    main()
