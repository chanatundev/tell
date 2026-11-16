"""Build the list of Commons images allowed as TRAINING reals.

Reads from heldout_sealed/real_selected.json ONLY the `id` and `artist` fields (never any image).
Excludes (1) every held-out id and (2) every image whose artist/uploader matches a held-out artist.
Output: data/commons_train_ok.json  (list of dicts: id, path, license, artist, title)

Without heldout_sealed/ (it is private), the published result of this filter, ATTRIBUTION_commons.csv,
defines the kept ids, so a public rebuild uses exactly the same 389 Commons training/validation reals.
"""
import csv, json, re, html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm_artist(a: str) -> str:
    a = html.unescape(re.sub(r"<[^>]+>", " ", a or ""))
    a = re.sub(r"\b(user|talk|contributions?)\b", " ", a, flags=re.I)
    a = re.sub(r"[^a-z0-9]+", " ", a.lower()).strip()
    return a


def same_artist(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    # containment for names of reasonable length (e.g. "Hippietrail" vs "Hippietrail (talk)")
    return (len(a) >= 5 and a in b) or (len(b) >= 5 and b in a)


def main():
    items = json.load(open(ROOT / "commons_filtered.json"))
    sealed_file = ROOT / "heldout_sealed" / "real_selected.json"
    if sealed_file.exists():
        sealed = json.load(open(sealed_file))
        held_ids = {r["id"] for r in sealed}                       # only id ...
        held_artists = {norm_artist(r["artist"]) for r in sealed}  # ... and artist
        held_artists.discard("")
    else:
        with open(ROOT / "ATTRIBUTION_commons.csv", newline="", encoding="utf-8") as f:
            published = {r["id"] for r in csv.DictReader(f)}
        print("heldout_sealed/ not found: keeping the ids published in ATTRIBUTION_commons.csv")
        held_ids = {it["id"] for it in items} - published         # everything the original filter excluded
        held_artists = set()
    keep, drop_id, drop_artist, drop_missing, drop_unknown_artist = [], [], [], [], []
    for it in items:
        p = ROOT / "data" / "commons_raw" / f"{it['id']}.jpg"
        if not p.exists():
            drop_missing.append(it["id"]); continue
        if it["id"] in held_ids:
            drop_id.append(it["id"]); continue
        a = norm_artist(it.get("artist", ""))
        if any(same_artist(a, h) for h in held_artists):
            drop_artist.append(it["id"]); continue
        if not a:
            # unknown uploader: cannot prove it differs from a held-out artist -> keep but record
            drop_unknown_artist.append(it["id"])
        keep.append({"id": it["id"], "path": str(p.relative_to(ROOT)), "license": it.get("license", ""),
                     "artist": re.sub(r"<[^>]+>", "", html.unescape(it.get("artist", "")))[:120],
                     "title": it.get("title", ""), "cat": it.get("cat", ""), "artist_norm": a})
    json.dump(keep, open(ROOT / "data" / "commons_train_ok.json", "w"), indent=1)
    stats = dict(total=len(items), kept=len(keep), dropped_heldout_id=len(drop_id),
                 dropped_same_artist=len(drop_artist), missing_file=len(drop_missing),
                 kept_with_unknown_artist=len(drop_unknown_artist), heldout_artists=len(held_artists))
    from collections import Counter
    stats["licenses_kept"] = Counter(k["license"] for k in keep).most_common()
    stats["cats_kept"] = Counter(k["cat"] for k in keep).most_common()
    json.dump(stats, open(ROOT / "data" / "commons_filter_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
