# Tell: flagging AI-generated receipt photos for expense review

**Veritas AI Hackathon 2026 · AI + Finance track · Team Tee to Three (Chanatun Kohsuwan and Arush Wadhawan)**

All of this work, V1 and V2 included, was done within the hackathon time limit.

> **This project is a prototype and is not financial advice.**

Judges: start with [JUDGES.md](JUDGES.md) (results, setup, demo) and the two-page writeup (summary and methods), [`output/pdf/tell-writeup.pdf`](output/pdf/tell-writeup.pdf).

Tell is an image classifier (a fine-tuned EfficientNet-B0) for small-business owners and bookkeepers who approve expense claims from receipt photos. It gives each photo a score for how AI-generated it looks. Above a frozen threshold, it suggests asking for the original receipt or a card statement before paying. It was trained on AI receipts from GPT-4o-era tools and tested on matched "twins" from a newer generator.

We built the receipt preparation pipeline, the data-leakage checks, three comparison baselines, the training and calibration code, the matched-pair evaluation and a local review interface. Tell fine-tunes an existing EfficientNet-B0 checkpoint. Dataset, model and AI-tool credits are linked below and in [CREDITS.md](CREDITS.md).

Tell looks for AI-generated images, which is only one kind of expense fraud. A real receipt can still be altered, reused or for personal spending, and a flagged photo can be genuine.

## Results (50 pairs: one real receipt photo + one AI twin from a newer generator)

| model | AI scored above its real pair | AI twins caught | real receipts wrongly flagged |
|---|---|---|---|
| v1, frozen before the pairs were opened | 78% | 37/50 (74%) | 15/50 (30%) |
| v2, built after the v1 test | 92% | 38/50 (76%) | 5/50 (10%) |

The first column shows how often the AI image scored above the real photo in its pair. The other two count flags at each model's fixed threshold.

V1 is the held-out result ([`results/heldout_v1/heldout_results.md`](results/heldout_v1/heldout_results.md)). The example images listed in those reports are not included. V2 added WildReceipt photos and new fakes after that test and was scored on the same pairs ([`results/post_unseal_v2/report.md`](results/post_unseal_v2/report.md)). A new set of pairs would give it an independent score.

## Limitations
* The held-out test is small (50 pairs) and team-made, and the confidence intervals are wide.
* The twins are synthetic and no real expense-fraud cases were used. These numbers don't measure real-world effectiveness.
* Results on the validation split are optimistic (epoch, temperature and threshold are chosen on it) and the validation split mixes sources with different countries/styles (see the country confound in `DATA.md`). The v1 twin result is the number to rely on.
* The "bookkeeper's checklist" baseline relies on Tesseract with English data only. It parses only a minority of receipts and almost none in non-Latin scripts. `DATA.md` reports its coverage.
* The v1 threshold did not carry over to new photos. It flagged 4.7% of real validation receipts but 30% of the real test photos, and scores can be overconfident.

## Reproduce (about 1 hour, plus ~20 min of GPU training)

Receipt images and model weights are not in this repository (dataset licences; the held-out twins stay private). These commands rebuild from the public sources.
```bash
python3 -m venv .venv && .venv/bin/pip install numpy pillow scikit-learn pandas scipy imagehash pytesseract \
    huggingface_hub torch torchvision timm pyarrow matplotlib nbformat nbclient ipykernel
brew install tesseract                      # macOS;  Colab/Ubuntu: apt-get install -y tesseract-ocr

# 1. data (every dataset in its own data/raw/<name>/ folder; run Python on downloads with -I)
.venv/bin/python src/download.py cord                       # public, CC BY 4.0
export HF_TOKEN=...                                          # your own read token, after accepting the dataset terms
.venv/bin/python src/download.py gpt4o                       # gated, CC BY-NC-SA 4.0
mkdir -p data/commons_raw && .venv/bin/python -I scripts/commons_dl.py commons_filtered.json data/commons_raw   # Commons thumbnails
.venv/bin/python -I src/commons_filter.py                    # held-out exclusions; without the private set it reuses ATTRIBUTION_commons.csv

# 2-3. normalise (strip metadata, long side 1024, padded 1024x1024 canvas, JPEG q85) + 80/20 grouped split
.venv/bin/python -I src/prepare.py                           # -> data/norm/<source>/, data/splits.csv, data/raw_stats.csv

# 4-5. leak check and baselines (all on the validation split)
.venv/bin/python -I src/leak_check.py
.venv/bin/python -I src/baselines.py

# 6. train (Apple MPS / CUDA / CPU); notebooks/tell_train.ipynb wraps the same command for a GPU runtime
.venv/bin/python -I src/train.py --splits data/splits.csv --norm-dir data/norm --out runs/main

# 7. freeze, THEN evaluate once on a sealed held-out set (ours is private)
cp runs/main/frozen.json frozen/frozen.json && git add frozen/frozen.json && git commit -m "freeze model + threshold"
.venv/bin/python src/eval_heldout.py --unseal --model runs/main/model.pt --frozen frozen/frozen.json \
        --val-report runs/main/report.json            # writes results/heldout_results.md + examples/
```
Smoke test (needs steps 1-3, so the gated GPT4o-Receipt data; under a minute, a few dozen images, 256 px, not a result): `.venv/bin/python -I src/train.py --splits data/splits.csv --norm-dir data/norm --out runs/smoke --smoke`

## What each piece does
| file | job |
|---|---|
| `src/normalize.py` | every image (train, val, held-out) is re-encoded identically: RGB, all metadata dropped, long side 1024, centred on a grey 1024x1024 canvas, JPEG q85 |
| `src/leak_check.py` | logistic regression on width/height/aspect/bytes/JPEG quantisation: AUROC on raw vs normalised files |
| `src/baselines.py` | (a) no-camera-EXIF rule, (b) FFT + intensity logistic regression, (c) bookkeeper's checklist (Tesseract + rules + perceptual-hash duplicates) |
| `src/tellmodel.py`, `src/train.py` | model, aspect-neutral square-window augmentation (no flips), temperature scaling, threshold = at most 5% of val reals flagged, `frozen.json` |
| `notebooks/tell_train.ipynb` | notebook wrapper around `train.py` (local or GPU runtime), with a guarded Grad-CAM cell |
| `src/eval_heldout.py` | sealed twin evaluation; refuses to run without `--unseal`, a model and a matching frozen file |
| `src/prepare_v2.py`, `src/eval_v2.py` | v2 data (WildReceipt + accepted new fakes) and post-unseal evaluation, kept apart from the v1 results |
| `src/demo.py` | local Gradio demo: next step, model score and frozen threshold |

## The sealed held-out set
The original v1 evaluation used 50 matched pairs after freezing. The set has now been unsealed. It must never be used for training, tuning or thresholds; v2 evaluation is explicitly post-unseal development. The generated fakes and held-out images are not included in this repository.

## Data and attributions

Full linked dataset, pretrained-model, software and AI-assistance credits are in [CREDITS.md](CREDITS.md). Compact PDF credit links are in [CREDITS.json](CREDITS.json). Counts, splits and limitations are in [DATA.md](DATA.md).

- [GPT4o-Receipt, Scam.AI](https://huggingface.co/datasets/Scam-AI/gpt4o-receipt): CC BY-NC-SA 4.0, non-commercial research; images not redistributed.
- [CORD v2, NAVER CLOVA](https://huggingface.co/datasets/naver-clova-ix/cord-v2): CC BY 4.0.
- [Wikimedia Commons](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia): individual file licences. Training credits: [ATTRIBUTION_commons.csv](ATTRIBUTION_commons.csv); selected test/spare source credits: [CREDITS.md](CREDITS.md#selected-commons-receipt-source-credits).
- [WildReceipt](https://github.com/open-mmlab/mmocr/blob/main/dataset_zoo/wildreceipt/metafile.yml): primary dataset metadata lists licence **N/A**, so receipt-image rights are unspecified; do not confuse the MMOCR code's Apache-2.0 licence with a dataset-image licence. Images are not redistributed.
- [EfficientNet-B0 ImageNet-1k checkpoint](https://huggingface.co/timm/efficientnet_b0.ra_in1k), via [timm](https://github.com/huggingface/pytorch-image-models): pretrained initialization, then fine-tuned in Tell.

## Local demo and v2

Only the demo line runs from public assets, and only once you have the weights. `prepare_v2.py` also needs WildReceipt and the project's generated fakes (not redistributed); `eval_v2.py` needs the private held-out pairs. They are listed so the v2 pipeline is auditable.

```bash
.venv/bin/pip install gradio==6.32.0
.venv/bin/python src/prepare_v2.py           # adds WildReceipt + accepted new fakes -> data/splits_v2.csv
.venv/bin/python -I src/train.py --splits data/splits_v2.csv --norm-dir data/norm --out runs/v2 --epochs 3 --batch 8 --workers 1
.venv/bin/python src/eval_v2.py              # post-unseal evaluation; never writes to results/heldout_v1
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python src/demo.py --run runs/v2
```

The demo needs trained weights (`runs/v2/model.pt`, about 16 MB), which are not in this repository. The demo checks the file against the SHA-256 in `runs/v2/frozen.json`. It binds to 127.0.0.1 only, with no public sharing link and no bundled receipts. Use `--run runs/v1` for the original frozen model. The two-page writeup (summary and methods) is `output/pdf/tell-writeup.pdf`.

## Licence

Code and documentation written for Tell: [Apache License 2.0](LICENSE), copyright 2026 Chanatun Kohsuwan and Arush Wadhawan (see [NOTICE](NOTICE)). Datasets, Commons files and the pretrained checkpoint keep their own licences, listed above and in [CREDITS.md](CREDITS.md).
