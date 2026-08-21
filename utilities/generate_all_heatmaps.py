"""
generate_all_heatmaps.py

Generate foreground (_0002) and background (_0003) scribble heatmaps
for ALL THREE strategies (centerline, random, boundary).

Heatmaps are stored in three separate directories under the dataset root.
Then, one strategy is deterministically assigned to each case (round-robin
on sorted case names) and copied into imagesTr as _0002 / _0003 for
nnU-Net preprocessing.

Usage:
    python generate_all_heatmaps.py --data_dir data/PSMA-FDG-PET-CT-Lesions_v2
    python generate_all_heatmaps.py --data_dir data/PSMA-FDG-PET-CT-Lesions_v2 --dry_run
"""

import os
import sys
import shutil
import argparse
import time
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib

try:
    from tqdm import tqdm
except ImportError:
    print("[WARN] pip install tqdm")
    class tqdm:
        def __init__(self, iterable=None, **kw):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable or [])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            pass
        def set_postfix_str(self, s, refresh=True):
            pass
        def close(self):
            pass
        @staticmethod
        def write(s):
            print(s)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "interactive"))

from simulate_scribbles import (
    get_random_k_components,
    generate_scribbles_for_components,
    generate_heatmap_from_scribbles,
    save_heatmap_nifti,
)
from skimage.morphology import binary_dilation, ball

STRATEGIES = ["centerline", "random", "boundary"]


def process_one(task):
    """Generate FG + BG heatmaps for one (case, strategy) pair."""
    label_path = task["label_path"]
    out_dir = task["out_dir"]
    strategy = task["strategy"]
    seed = task["seed"]
    case_name = task["case_name"]

    fg_out = os.path.join(out_dir, "{}_0002.nii.gz".format(case_name))
    bg_out = os.path.join(out_dir, "{}_0003.nii.gz".format(case_name))

    if (os.path.isfile(fg_out) and os.path.getsize(fg_out) > 100
            and os.path.isfile(bg_out) and os.path.getsize(bg_out) > 100):
        return {"case": case_name, "strategy": strategy, "status": "skip"}

    t0 = time.time()
    try:
        img = nib.load(label_path)
        data = img.get_fdata().astype(np.uint8)

        if np.sum(data) == 0:
            empty = np.zeros_like(data, dtype=np.float32)
            save_heatmap_nifti(empty, label_path, fg_out)
            save_heatmap_nifti(empty, label_path, bg_out)
            return {"case": case_name, "strategy": strategy, "status": "empty",
                    "time": time.time() - t0}

        # Foreground
        labels_fg, ids_fg = get_random_k_components(data, k=5, seed=seed)
        scr_fg = generate_scribbles_for_components(labels_fg, ids_fg, strategy, seed)
        hm_fg = generate_heatmap_from_scribbles(scr_fg, sigma=0)

        # Background
        dilated = binary_dilation(data, ball(1))
        dilated = binary_dilation(dilated, ball(1))
        bg_region = (dilated.astype(np.uint8) - data.astype(np.uint8)) > 0
        bg_region = bg_region.astype(np.uint8)
        labels_bg, ids_bg = get_random_k_components(bg_region, k=5, seed=seed)
        scr_bg = generate_scribbles_for_components(labels_bg, ids_bg, strategy, seed)
        hm_bg = generate_heatmap_from_scribbles(scr_bg, sigma=0)

        save_heatmap_nifti(hm_fg, label_path, fg_out)
        save_heatmap_nifti(hm_bg, label_path, bg_out)

        return {"case": case_name, "strategy": strategy, "status": "ok",
                "time": time.time() - t0}
    except Exception as e:
        return {"case": case_name, "strategy": strategy,
                "status": "error: {}".format(e), "time": time.time() - t0}


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmaps for 3 scribble strategies and assign to imagesTr."
    )
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "PSMA-FDG-PET-CT-Lesions_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")

    for d, name in [(images_dir, "imagesTr"), (labels_dir, "labelsTr")]:
        if not os.path.isdir(d):
            print("[ERROR] {} not found: {}".format(name, d))
            sys.exit(1)

    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")])
    cases = [f.replace(".nii.gz", "") for f in label_files]
    print("Found {} cases".format(len(cases)))

    # ---- Strategy assignment (round-robin on sorted names) ----
    assignment = {}
    for i, c in enumerate(cases):
        assignment[c] = STRATEGIES[i % len(STRATEGIES)]

    counts = {}
    for s in STRATEGIES:
        counts[s] = sum(1 for v in assignment.values() if v == s)
    print("Strategy assignment: {}".format(counts))

    # Save assignment for reproducibility
    assign_path = os.path.join(args.data_dir, "strategy_assignment.json")
    with open(assign_path, "w") as f:
        json.dump(assignment, f, indent=2)
    print("Saved strategy assignment to {}".format(assign_path))

    if args.dry_run:
        # Check status
        for s in STRATEGIES:
            hm_dir = os.path.join(args.data_dir, "heatmaps_{}".format(s))
            if os.path.isdir(hm_dir):
                n = len([f for f in os.listdir(hm_dir) if f.endswith("_0002.nii.gz")])
            else:
                n = 0
            print("  heatmaps_{}: {} files".format(s, n))
        # Check imagesTr
        n_0002 = len([f for f in os.listdir(images_dir) if f.endswith("_0002.nii.gz")])
        print("  imagesTr _0002: {} files".format(n_0002))
        return

    # ---- Phase 1: Generate heatmaps for all 3 strategies ----
    print("")
    print("=" * 60)
    print("Phase 1: Generating heatmaps for all 3 strategies")
    print("=" * 60)

    tasks = []
    for s in STRATEGIES:
        hm_dir = os.path.join(args.data_dir, "heatmaps_{}".format(s))
        os.makedirs(hm_dir, exist_ok=True)
        for lf in label_files:
            case_name = lf.replace(".nii.gz", "")
            tasks.append({
                "label_path": os.path.join(labels_dir, lf),
                "out_dir": hm_dir,
                "strategy": s,
                "seed": args.seed,
                "case_name": case_name,
            })

    ok = 0
    empty = 0
    skip = 0
    err = 0
    t0 = time.time()

    pbar = tqdm(total=len(tasks), desc="Generating", unit="file")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            st = r["status"]
            if st == "ok":
                ok += 1
            elif st == "empty":
                empty += 1
            elif st == "skip":
                skip += 1
            else:
                err += 1
                tqdm.write("  [ERR] {} / {}: {}".format(r["case"], r["strategy"], st))
            pbar.update(1)
            pbar.set_postfix_str("ok={} empty={} skip={} err={}".format(ok, empty, skip, err))
    pbar.close()

    print("Phase 1 done in {:.1f} min  (ok={} empty={} skip={} err={})".format(
        (time.time() - t0) / 60, ok, empty, skip, err))

    # ---- Phase 2: Copy assigned heatmaps into imagesTr ----
    print("")
    print("=" * 60)
    print("Phase 2: Copying assigned heatmaps to imagesTr")
    print("=" * 60)

    copied = 0
    for case_name, strategy in tqdm(assignment.items(), desc="Copying", unit="case"):
        hm_dir = os.path.join(args.data_dir, "heatmaps_{}".format(strategy))
        for suffix in ("_0002.nii.gz", "_0003.nii.gz"):
            src = os.path.join(hm_dir, "{}{}".format(case_name, suffix))
            dst = os.path.join(images_dir, "{}{}".format(case_name, suffix))
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                tqdm.write("  [WARN] missing: {}".format(src))
        copied += 1

    print("Copied heatmaps for {} cases into imagesTr".format(copied))
    print("Done.")


if __name__ == "__main__":
    main()