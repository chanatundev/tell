"""Download public datasets into data/raw/<name>/. Never executes anything downloaded.

  python src/download.py cord          # CORD v2 (public, CC BY 4.0)
  python src/download.py gpt4o         # GPT4o-Receipt (GATED on Hugging Face)

GPT4o-Receipt is gated: a human must accept the terms on huggingface.co with their OWN
account and export HF_TOKEN=<their read token> before running this. Without a token this
script prints that and exits non-zero; it never asks for / sends personal details.
"""
import os, sys, io, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def cord():
    from huggingface_hub import snapshot_download
    out = RAW / "cord_v2"
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download("naver-clova-ix/cord-v2", repo_type="dataset", local_dir=str(out),
                      allow_patterns=["data/*.parquet", "README.md"])
    # Extract images from parquet (pure data; no pickle) -> data/raw/cord_v2/images/
    import pyarrow.parquet as pq
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    n = 0
    for pf in sorted((out / "data").glob("*.parquet")):
        split = pf.name.split("-")[0]  # train / validation / test
        t = pq.read_table(pf, columns=["image"]).to_pylist()
        for row in t:
            b = row["image"]["bytes"]
            (img_dir / f"cord_{n:04d}_{split}.png").write_bytes(b)
            n += 1
    print("CORD images extracted:", n)


def gpt4o():
    from huggingface_hub import snapshot_download
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("GPT4o-Receipt is gated. A human must accept the terms on huggingface.co/datasets/"
              "Scam-AI/gpt4o-receipt with their own account, then: export HF_TOKEN=... and rerun.")
        sys.exit(2)
    out = RAW / "gpt4o_receipt"
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download("Scam-AI/gpt4o-receipt", repo_type="dataset", local_dir=str(out),
                      allow_patterns=["*.png", "README.md"], token=tok)
    print("PNG count:", len(list(out.glob("*.png"))))


if __name__ == "__main__":
    {"cord": cord, "gpt4o": gpt4o}[sys.argv[1]]()
