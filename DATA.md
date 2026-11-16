# Data sources, licences, counts

*Current status: GPT4o-Receipt was downloaded and v1 trained/evaluated. The original counts, exclusions and initial gate notes below are retained as historical provenance. V2 adds 1,739 annotated WildReceipt photos (1,267 train / 472 validation) and 32 accepted unrelated new generated images (25 train / 7 validation); 68 generated candidates were excluded by the recorded checks. No extra Commons images were added. V2 twin evaluation is post-unseal development, never fresh heldout.*

| source | role | label | licence (as found) | count | notes |
|---|---|---|---|---|---|
| **GPT4o-Receipt** `Scam-AI/gpt4o-receipt` | train/val AI | 1 | CC BY-NC-SA 4.0 (YAML + card) | **776 AI PNG, 0 real** (card: 776 PNG + 764 txt; 608 matched pairs; reals of the full benchmark are *not* in this repo). **Downloaded for v1; initial gate resolved.** | GPT-4o text + GPT-Image-1 render, US-style, 8.25% tax. NC-SA: the project is non-commercial; do not publish the fakes. |
| **CORD v2** `naver-clova-ix/cord-v2` | train/val real | 0 | **CC BY 4.0** (YAML on the dataset card, confirmed) | **1,000** receipts (800 train / 100 validation / 100 test shards; we re-split 80/20 ourselves) | Indonesian shop receipts, photos. Cite Park et al., "CORD: A Consolidated Receipt Dataset for Post-OCR Parsing", NeurIPS 2019 Workshop on Document Intelligence. |
| **Wikimedia Commons** (`commons_filtered.json`, `data/commons_raw/`) | train/val real | 0 | per-file, see below | 768 downloaded -> **389 usable** | see exclusion rules |

## GPT4o-Receipt access
GPT4o-Receipt is gated on Hugging Face. It was downloaded after accepting its access terms, and all 776 AI images were used.

## Commons reals: what was excluded and why
Counts (`data/commons_filter_stats.json`, `src/commons_filter.py`):

| step | images |
|---|---|
| downloaded (Category:Receipts and sub-categories) | 768 |
| removed: `id` is in the sealed held-out set (60 = 50 test + 10 spare) | -60 |
| removed: same artist/uploader as any held-out image (43 distinct held-out artists; normalised, case/markup-insensitive, substring match for names >= 5 chars) | -319 |
| **kept as training/val reals** | **389** (16 have an empty artist field and were kept: a same-uploader overlap cannot be proven or excluded) |

Only the `id` and `artist` fields of `heldout_sealed/real_selected.json` were read for this; no held-out image was opened. (One early inspection command also printed a single full record of that JSON, text only; no image data. Noted for the record.)
The aggressive artist rule removes a lot (one or two prolific Polish uploaders dominate the category); that is intended: it prevents near-duplicate leakage (same shop, same camera, same day) between training reals and held-out reals.

Licences of the 389 kept: CC BY-SA 4.0 (131), CC BY 4.0 (72), CC0 (51), CC BY-SA 2.0 (37), CC BY 2.0 (35), Public domain (32), CC BY-SA 3.0 (22), No restrictions (4), CC BY 3.0 (3), CC BY 2.5 dk (1), "Copyrighted free use" (1).
Mix by category: Receipts of Poland 105, Japan 65, Category:Receipts 52, Germany 22, restaurant checks (Japan 14, China 13, other ~30), plus a long tail (Austria, Philippines, China, Russia, Czechia, UK, Netherlands, US, India, Nigeria, ...).

**Caveat:** Commons "Receipts" categories contain many images that are *not* clean receipt photos (e.g. a person at a till, a vending-machine ticket, waybills, bills of lading, spikes of receipts). They are kept as "real photo" negatives, which is fine for a real-vs-AI detector but means the false-alarm rate on Commons is **not** the false-alarm rate on a clean receipt stream. The share of true receipts among the 389 was **not measured**. Where the repo is public, per-file attribution is in `ATTRIBUTION_commons.csv` (title, artist, licence, Commons page link). CC BY / BY-SA require attribution; BY-SA ShareAlike may apply to redistributed derivatives, so we do not redistribute the images, only the list.

## Splits (`data/splits.csv`, built by `src/prepare.py`)
80/20 train/val **per source**, grouped so near-duplicates stay on the same side: GPT4o-Receipt grouped by store name parsed from the file name (`receipt_<store>_<n>.png`; files without a store name are their own group), Commons grouped by artist, CORD random (no template field). Real-class numbers: CORD 800 train / 200 val, Commons 311 train / 78 val. GPT4o-Receipt contributed 619 training and 157 validation images.

## The country / style confound (read this before trusting any number)
* CORD is **Indonesian** (IDR, thousands separators, Bahasa); the GPT-4o fakes are **US-style** ($, 8.25% tax, North-American chains). A model can reach a high validation score by detecting "US layout vs Indonesian layout", which says nothing about AI-ness.
* The Commons reals (Poland, Japan, Germany, China, ...) diversify the real side but are different again (many scans/photos of non-Latin or non-US receipts).
* Other confounds: CORD/GPT4o files are PNG, Commons files are JPEG (earlier lossy compression survives our q85 re-encode as a double-compression trace; `leak_check.py` reports a blockiness-only AUROC for this); capture style (flat scan vs phone photo).
* This is why the real test is the held-out *twin* set (same country, shop type and style on both sides), and why per-source results are reported separately.


## Linked sources and licence clarification (November 15)

See [CREDITS.md](CREDITS.md) for complete linked credits. The original [WildReceipt dataset metafile](https://github.com/open-mmlab/mmocr/blob/main/dataset_zoo/wildreceipt/metafile.yml) lists dataset licence type/link as N/A. Its receipt-image licence is **unspecified**, not verified Apache-2.0; the MMOCR code licence must not be extended to the dataset by assumption. No WildReceipt images are redistributed here.

The [GPT4o-Receipt card](https://huggingface.co/datasets/Scam-AI/gpt4o-receipt/blob/main/README.md) identifies CC BY-NC-SA 4.0; the [CORD v2 card](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/main/README.md) identifies CC BY 4.0. The local timm `efficientnet_b0` default resolves to the [efficientnet_b0.ra_in1k model card](https://huggingface.co/timm/efficientnet_b0.ra_in1k), which identifies ImageNet-1k pretraining and Apache-2.0 for that checkpoint.

The existing 389-row ATTRIBUTION_commons.csv covers retained training/validation sources only; [CREDITS.md](CREDITS.md#selected-commons-receipt-source-credits) separately lists the selected 50 test and 10 spare source pages and metadata, without receipt-content personal details. The individual linked Commons file pages are authoritative for reuse terms.
