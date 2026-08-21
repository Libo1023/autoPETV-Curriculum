"""
convert_and_merge_deeppsma.py

Convert DeepPSMA raw data (train_0001 ... train_0100) to nnU-Net format
and merge into the main PSMA-FDG dataset.

For each of the 100 patients, both PSMA and FDG scans are converted,
giving up to 200 additional training samples.

Steps:
  1. Resample CT -> PET space for each case
  2. Save CT (_0000), PET (_0001), label (TTB) into main imagesTr/labelsTr
  3. Create modified splits_final.json with DeepPSMA in all folds' train sets
  4. Update dataset.json numTraining count

Usage:
    python convert_and_merge_deeppsma.py
    python convert_and_merge_deeppsma.py --dry_run
"""

import os
import sys
import json
import shutil
import argparse
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

try:
    from tqdm import tqdm
except ImportError:
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

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def convert_one_case(task):
    """Convert one (patient, tracer) pair to nnUNet format."""
    case_dir = task["case_dir"]
    tracer = task["tracer"]
    case_id = task["case_id"]
    images_out = task["images_out"]
    labels_out = task["labels_out"]

    tracer_dir = os.path.join(case_dir, tracer)
    ct_path = os.path.join(tracer_dir, "CT.nii.gz")
    pet_path = os.path.join(tracer_dir, "PET.nii.gz")
    ttb_path = os.path.join(tracer_dir, "TTB.nii.gz")

    base_name = "deep{}_{:04d}".format(tracer, case_id)

    ct_out = os.path.join(images_out, "{}_0000.nii.gz".format(base_name))
    pet_out = os.path.join(images_out, "{}_0001.nii.gz".format(base_name))
    label_out = os.path.join(labels_out, "{}.nii.gz".format(base_name))

    # Skip if already done
    if (os.path.isfile(ct_out) and os.path.getsize(ct_out) > 1000
            and os.path.isfile(pet_out) and os.path.getsize(pet_out) > 1000
            and os.path.isfile(label_out) and os.path.getsize(label_out) > 100):
        return {"case": base_name, "status": "skip"}

    # Check source files exist
    for p, name in [(ct_path, "CT"), (pet_path, "PET"), (ttb_path, "TTB")]:
        if not os.path.isfile(p):
            return {"case": base_name, "status": "missing {}".format(name)}

    try:
        t0 = time.time()

        ct_img = nib.load(ct_path)
        pet_img = nib.load(pet_path)

        # Resample CT to PET space (linear interpolation)
        ct_resampled = resample_from_to(ct_img, pet_img, order=1)

        # Save
        nib.save(ct_resampled, ct_out)
        shutil.copy2(pet_path, pet_out)
        shutil.copy2(ttb_path, label_out)

        return {"case": base_name, "status": "ok", "time": time.time() - t0}

    except Exception as e:
        return {"case": base_name, "status": "error: {}".format(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepPSMA to nnUNet format and merge with main dataset."
    )
    parser.add_argument(
        "--deeppsma_dir", type=str,
        default=os.path.join(REPO_ROOT, "data"),
        help="Directory containing train_0001 ... train_0100 folders.",
    )
    parser.add_argument(
        "--main_dataset", type=str,
        default=os.path.join(REPO_ROOT, "data", "PSMA-FDG-PET-CT-Lesions_v2"),
        help="Main dataset directory with imagesTr, labelsTr, splits_final.json.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    images_out = os.path.join(args.main_dataset, "imagesTr")
    labels_out = os.path.join(args.main_dataset, "labelsTr")
    splits_path = os.path.join(args.main_dataset, "splits_final.json")
    dataset_json_path = os.path.join(args.main_dataset, "dataset.json")

    for p, name in [(images_out, "imagesTr"), (labels_out, "labelsTr"),
                     (splits_path, "splits_final.json")]:
        if not os.path.exists(p):
            print("[ERROR] {} not found: {}".format(name, p))
            sys.exit(1)

    # Find all DeepPSMA case directories
    case_dirs = sorted([
        d for d in Path(args.deeppsma_dir).glob("train_*")
        if d.is_dir()
    ])
    print("Found {} DeepPSMA case directories".format(len(case_dirs)))

    if len(case_dirs) == 0:
        print("[ERROR] No train_XXXX directories found in {}".format(args.deeppsma_dir))
        sys.exit(1)

    # Build task list
    tasks = []
    tracers = ["PSMA", "FDG"]
    for case_dir in case_dirs:
        case_str = case_dir.name.split("_")[-1]
        case_id = int(case_str)
        for tracer in tracers:
            tracer_dir = os.path.join(str(case_dir), tracer)
            if os.path.isdir(tracer_dir):
                tasks.append({
                    "case_dir": str(case_dir),
                    "tracer": tracer,
                    "case_id": case_id,
                    "images_out": images_out,
                    "labels_out": labels_out,
                })

    print("Total conversion tasks: {} ({} patients x {} tracers)".format(
        len(tasks), len(case_dirs), len(tracers)))

    if args.dry_run:
        # Check existing
        done = 0
        for t in tasks:
            bn = "deep{}_{:04d}".format(t["tracer"], t["case_id"])
            ct = os.path.join(images_out, "{}_0000.nii.gz".format(bn))
            if os.path.isfile(ct) and os.path.getsize(ct) > 1000:
                done += 1
        print("  Already converted: {}".format(done))
        print("  Remaining: {}".format(len(tasks) - done))
        return

    # ---- Phase 1: Convert ----
    print("")
    print("=" * 60)
    print("Phase 1: Converting DeepPSMA to nnUNet format")
    print("=" * 60)

    ok = 0
    skip = 0
    err = 0
    converted_names = []

    pbar = tqdm(total=len(tasks), desc="Converting", unit="case")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one_case, t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            if r["status"] == "ok":
                ok += 1
                converted_names.append(r["case"])
            elif r["status"] == "skip":
                skip += 1
                converted_names.append(r["case"])
            else:
                err += 1
                tqdm.write("  [WARN] {}: {}".format(r["case"], r["status"]))
            pbar.update(1)
            pbar.set_postfix_str("ok={} skip={} err={}".format(ok, skip, err))
    pbar.close()

    # Collect ALL deep* case names from labelsTr (in case of partial runs)
    all_deep_cases = sorted([
        f.replace(".nii.gz", "")
        for f in os.listdir(labels_out)
        if f.startswith("deep") and f.endswith(".nii.gz")
    ])

    print("")
    print("Conversion done: ok={} skip={} err={}".format(ok, skip, err))
    print("Total DeepPSMA cases in labelsTr: {}".format(len(all_deep_cases)))

    # ---- Phase 2: Modify splits ----
    print("")
    print("=" * 60)
    print("Phase 2: Updating splits_final.json")
    print("=" * 60)

    # Backup original splits
    splits_backup = splits_path + ".original"
    if not os.path.isfile(splits_backup):
        shutil.copy2(splits_path, splits_backup)
        print("Original splits backed up to: {}".format(splits_backup))

    with open(splits_path, "r") as f:
        splits = json.load(f)

    for fold_idx, fold in enumerate(splits):
        # Remove any existing deep* cases from train (idempotent)
        fold["train"] = [c for c in fold["train"] if not c.startswith("deep")]
        # Add all DeepPSMA cases to train
        fold["train"] = sorted(fold["train"] + all_deep_cases)
        # Val stays unchanged
        print("  Fold {}: train={} (added {}), val={}".format(
            fold_idx, len(fold["train"]), len(all_deep_cases), len(fold["val"])))

    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print("[OK] splits_final.json updated")

    # ---- Phase 3: Update dataset.json ----
    print("")
    print("=" * 60)
    print("Phase 3: Updating dataset.json")
    print("=" * 60)

    # Count total cases in labelsTr
    total_labels = len([
        f for f in os.listdir(labels_out) if f.endswith(".nii.gz")
    ])

    dataset_json = {
        "channel_names": {
            "0": "CT",
            "1": "PET",
            "2": "FG",
            "3": "BG"
        },
        "labels": {
            "background": 0,
            "tumor": 1
        },
        "numTraining": total_labels,
        "file_ending": ".nii.gz"
    }

    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print("[OK] dataset.json updated: numTraining = {}".format(total_labels))

    print("")
    print("=" * 60)
    print("Done. {} DeepPSMA samples merged into main dataset.".format(
        len(all_deep_cases)))
    print("=" * 60)


if __name__ == "__main__":
    main()