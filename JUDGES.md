# Tell: guide for judges

Tell flags receipt photos that look AI-generated. When it does, the business can ask for the original receipt or a card statement before paying. A flag is a reason to ask for proof, and the decision stays with the reviewer. **This project is a prototype and is not financial advice.**

## Five-minute review (no setup)

| what | where |
|---|---|
| Two-page writeup (summary and methods) | [`output/pdf/tell-writeup.pdf`](output/pdf/tell-writeup.pdf) |
| Original frozen evaluation (v1, 50 sealed pairs) | [`results/heldout_v1/heldout_results.md`](results/heldout_v1/heldout_results.md) |
| v2 on the same pairs | [`results/post_unseal_v2/report.md`](results/post_unseal_v2/report.md) |
| Frozen v1 threshold and model hash | [`frozen/`](frozen/) |
| Data sources, licences, counts | [`DATA.md`](DATA.md), [`CREDITS.md`](CREDITS.md) |

Headline numbers on 50 pairs (one real photo + one AI twin each):

| model | AI scored above its real pair | AI twins caught | real receipts wrongly flagged |
|---|---|---|---|
| v1, frozen before unsealing | 78% | 37/50 (74%) | 15/50 (30%) |
| v2, built after the v1 test | 92% | 38/50 (76%) | 5/50 (10%) |

We froze V1 before opening the test images and scored it once on these pairs. V2 was built afterwards and scored on the same pairs. A new set of pairs would give it an independent score.

## What is not in this repository

* **Receipt images.** GPT4o-Receipt is CC BY-NC-SA and gated, WildReceipt's image licence is unspecified, and Commons files carry per-file licences, so no images are redistributed. The held-out twins were generated for this project and are kept private.
* **Model weights** (`runs/v1/model.pt`, `runs/v2/model.pt`). `frozen.json` records each file's SHA-256; the demo refuses weights that do not match.

## Setup (tested on macOS with Python 3.14)

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pillow scikit-learn pandas scipy imagehash pytesseract \
    huggingface_hub torch torchvision timm pyarrow matplotlib gradio==6.32.0
brew install tesseract          # Ubuntu: sudo apt-get install -y tesseract-ocr (only for the OCR baseline)
```

## Run the demo (needs weights)

Place the weights at `runs/v2/model.pt`, then:

```bash
GRADIO_ANALYTICS_ENABLED=False .venv/bin/python src/demo.py --run runs/v2   # http://127.0.0.1:7860
```

Upload a receipt photo. The demo shows the next step, the model score and the validation-selected threshold. It runs locally only, with no public link. Use `--run runs/v1` for the original frozen model.

## Rebuild from public data (about 1 hour, plus GPU training)

```bash
.venv/bin/python src/download.py cord                 # CC BY 4.0, public
export HF_TOKEN=...                                    # your own read token, after accepting the dataset terms
.venv/bin/python src/download.py gpt4o                # gated, CC BY-NC-SA 4.0
mkdir -p data/commons_raw && .venv/bin/python -I scripts/commons_dl.py commons_filtered.json data/commons_raw
.venv/bin/python -I src/commons_filter.py             # reuses the published Commons selection (ATTRIBUTION_commons.csv)
.venv/bin/python -I src/prepare.py                    # strip metadata, normalise, grouped 80/20 split
.venv/bin/python -I src/leak_check.py                 # can file statistics alone separate the classes?
.venv/bin/python -I src/baselines.py                  # no-EXIF rule, frequency model, OCR checklist
.venv/bin/python -I src/train.py --splits data/splits.csv --norm-dir data/norm --out runs/main
```

Training takes roughly 10-25 minutes on a T4-class GPU.

### Training in the notebook (optional)

`notebooks/tell_train.ipynb` runs the same training, then shows the validation report and an optional Grad-CAM view.

* It needs a data folder holding `norm/` and `splits.csv`, both written by `src/prepare.py`. By default it uses this repo's `data/`; set `TELL_DATA` to use another folder. `TELL_WORK` sets the working folder (default `nb_work/`) and `TELL_SPLITS` picks a different splits file.
* On Colab, upload the repo and your data folder, choose a GPU runtime (Runtime -> Change runtime type -> T4 GPU), then Run all. Keep any copy of the data private: it contains the AI-generated images, which must not be redistributed.
* Set `TELL_SMOKE=1` for a one-minute check that the pipeline runs (a few dozen images, one epoch, 256 px); its numbers mean nothing.
* Outputs (`model.pt`, `frozen.json`, `report.md`) land in `<TELL_WORK>/run`. Freeze `frozen.json` before evaluating on any held-out set.

### Other data

The v2 additions (WildReceipt and newer-generator fakes) are described in `DATA.md`. Rerunning `src/eval_heldout.py` or `src/eval_v2.py` requires the private held-out pairs, so the saved reports above are the record.
