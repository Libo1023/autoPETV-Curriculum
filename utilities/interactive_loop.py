from pathlib import Path 
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import argparse
import shutil
import subprocess
import traceback
import logging
from typing import Dict, List

import numpy as np
import nibabel as nib
import SimpleITK as sitk

from simulate_scribbles import (
    simulate_scribble_from_label,
    scribbles_to_gc_format,
    gc_to_swfastedit_format,
    heatmap_from_coords,
    save_heatmap_nifti,
)

from metrics import MetricEvaluator

# Disable SimpleITK warnings
sitk.ProcessObject_SetGlobalWarningDisplay(False)


# =============================================================================
# Logging setup
# =============================================================================
def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("interactive_segmentation")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# =============================================================================
# Metrics
# =============================================================================
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice score between prediction and ground truth."""
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)

    if denom == 0:
        return 1.0

    return 2.0 * intersection / denom


def detection_matching_metric(pred: np.ndarray, gt: np.ndarray) -> float:
    """Placeholder for detection matching metric."""
    return 0.0


# =============================================================================
# Utilities
# =============================================================================
def convert_mha_to_nii(mha_input_path: str, nii_out_path: str) -> None:
    """Convert .mha image to .nii.gz."""
    img = sitk.ReadImage(mha_input_path)
    sitk.WriteImage(img, nii_out_path, True)


def safe_remove(path: str) -> None:
    """Remove file if it exists (silently ignore errors)."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def clean_directory(directory: str) -> None:
    """Remove all files in a directory."""
    if not os.path.exists(directory):
        return

    for f in os.listdir(directory):
        safe_remove(os.path.join(directory, f))


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_cases", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--input_interface", type=str, required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["centerline", "random", "boundary"],
    )
    parser.add_argument("--max_iters", type=int, default=5)

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    log_file = os.path.join(args.result_dir, "run.log")
    logger = setup_logger(log_file)

    # Check model weights before starting
    logger.info("Checking model weights...")
    weight_check_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "nnunet-baseline",
        "check_weights.sh"
    )
    try:
        result = subprocess.run(
            ["bash", weight_check_script],
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )
        logger.info("Weight check passed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Weight check failed: {e.stderr}")
        raise RuntimeError("Model weights are not properly downloaded. Cannot proceed.")
    except subprocess.TimeoutExpired:
        logger.error("Weight check timed out")
        raise RuntimeError("Weight check took too long. Please check manually.")

    logger.info("Starting interactive segmentation pipeline")

    metrics_evaluator = MetricEvaluator()

    # -------------------------------------------------------------------------
    # Collect data
    # -------------------------------------------------------------------------
    image_dir = os.path.join(args.input_cases, "images")
    label_dir = os.path.join(args.input_cases, "labels")

    cts = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if "_0000" in f])
    pets = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if "_0001" in f])
    labels = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)])

    output_dice_file = os.path.join(args.result_dir, "metric_scores.json")
    case_dict: Dict[str, List[Dict]] = {}

    # Interface paths
    ct_dir = os.path.join(args.input_interface, "input", "images", "ct")
    pet_dir = os.path.join(args.input_interface, "input", "images", "pet")
    seg_dir = os.path.join(args.input_interface, "output", "images", "tumor-lesion-segmentation")

    # -------------------------------------------------------------------------
    # Case loop
    # -------------------------------------------------------------------------
    for ct, pet, label in zip(cts, pets, labels):

        if "fdg" in ct or '198' in ct:
            continue

        tag = os.path.basename(ct).replace(".nii.gz", "")
        logger.info(f"Processing case: {tag}")

        case_result_dir = os.path.join(args.result_dir, tag)
        os.makedirs(case_result_dir, exist_ok=True)

        case_dict[tag] = []

        # Clean interface
        for d in [ct_dir, pet_dir, seg_dir]:
            clean_directory(d)

        try:
            # -----------------------------------------------------------------
            # Load data
            # -----------------------------------------------------------------
            ct_img = sitk.ReadImage(ct)
            pet_img = sitk.ReadImage(pet)

            ct_out = os.path.join(ct_dir, f"case_{tag}.mha")
            pet_out = os.path.join(pet_dir, f"case_{tag}.mha")

            sitk.WriteImage(ct_img, ct_out)
            sitk.WriteImage(pet_img, pet_out)

            label_img = nib.load(label)
            gt = label_img.get_fdata()

            case_shape = gt.shape
            empty_gt = np.sum(gt) == 0

            output_json = os.path.join(args.input_interface, "input", "lesion-clicks.json")
            prev_dice = None
            prev_dmm = None  # FIX: was `prec_dmm` (typo)

            # -----------------------------------------------------------------
            # Iteration loop
            # -----------------------------------------------------------------
            for it in range(args.max_iters):
                logger.info(f"[{tag}] Iteration {it}")

                try:
                    if it == 0:
                        data = {"tumor": [], "background": []}

                    else:
                        if empty_gt:
                            logger.info("Empty GT → reusing previous Dice")
                            dice = prev_dice if prev_dice is not None else 0.0
                            dmm = prev_dmm if prev_dmm is not None else 0.0

                        else:
                            seg_path = os.path.join(seg_dir, f"case_{tag}.mha")
                            seg_nii = seg_path.replace(".mha", ".nii.gz")

                            if not os.path.exists(seg_path):
                                raise FileNotFoundError("Missing segmentation")

                            convert_mha_to_nii(seg_path, seg_nii)

                            pred = nib.load(seg_nii).get_fdata()
                            os.remove(seg_nii)

                            with open(output_json, "r") as f:
                                data = gc_to_swfastedit_format(json.load(f))

                            if pred.shape != gt.shape:
                                raise ValueError("Shape mismatch")

                            overseg = (pred == 1) & (gt == 0)
                            underseg = (pred == 0) & (gt == 1)

                            scribbles_bg, _, fp = simulate_scribble_from_label(overseg, args.strategy)
                            scribbles_fg, _, fn = simulate_scribble_from_label(underseg, args.strategy)

                            if fp <= fn:
                                data["tumor"] += scribbles_fg
                            else:
                                data["background"] += scribbles_bg

                    # Save scribbles
                    with open(output_json, "w") as f:
                        json.dump(scribbles_to_gc_format(data), f)

                    try:
                        click_save_path = os.path.join(case_result_dir, f"iter_{it}_scribbles.json")
                        with open(click_save_path, "w") as f:
                            json.dump(data, f, indent=4)
                        logger.info(f"[{tag}] Saved scribbles: {click_save_path}")
                    except Exception as e:
                        logger.warning(f"[{tag}] Failed to save scribbles at iter {it}: {e}")

                    # Run inference
                    subprocess.run(
                        "bash nnunet-baseline/test.sh",
                        shell=True,
                        timeout=600,
                        check=True,
                    )

                    seg_path = os.path.join(seg_dir, f"case_{tag}.mha")
                    seg_nii = seg_path.replace(".mha", ".nii.gz")

                    convert_mha_to_nii(seg_path, seg_nii)

                    pred = nib.load(seg_nii).get_fdata()
                    os.remove(seg_nii)

                    dice = dice_score(pred, gt)

                    dmm = metrics_evaluator(
                        prediction=pred.astype(np.uint8),
                        ground_truth=gt.astype(np.uint8),
                        case_name="case1"
                    )['f1']

                    # Save intermediate prediction
                    try:
                        save_path = os.path.join(case_result_dir, f"iter_{it}.nii.gz")
                        convert_mha_to_nii(seg_path, save_path)
                        logger.info(f"[{tag}] Saved prediction: {save_path}")
                    except Exception as e:
                        logger.warning(f"[{tag}] Failed to save prediction at iter {it}: {e}")

                except Exception as e:
                    logger.warning(f"[{tag}] Iteration {it} failed: {e}")
                    logger.debug(traceback.format_exc())

                    dice, dmm = 0.0, 0.0

                prev_dice = float(dice)
                prev_dmm = float(dmm)

                case_dict[tag].append(
                    {"iteration": it, "dice": float(dice), "dmm": float(dmm)}
                )

                logger.info(f"[{tag}] Dice@{it}: {dice:.4f}")

        except Exception as e:
            logger.error(f"[{tag}] Case failed completely: {e}")
            logger.debug(traceback.format_exc())

            case_dict[tag] = [
                {"iteration": i, "dice": 0.0, "dmm": 0.0}
                for i in range(args.max_iters)
            ]

        # Save progress
        with open(output_dice_file, "w") as f:
            json.dump(case_dict, f, indent=4)

    logger.info("All cases processed")

    # -------------------------------------------------------------------------
    # Compute AUC
    # -------------------------------------------------------------------------
    with open(output_dice_file, "r") as f:
        data = json.load(f)

    auc_results = {}

    for case_id, records in data.items():
        records = sorted(records, key=lambda x: x["iteration"])

        iterations = np.array([r["iteration"] for r in records], dtype=float)
        dice = np.array([r["dice"] for r in records], dtype=float)
        dmm = np.array([r["dmm"] for r in records], dtype=float)

        auc_dice = np.trapz(dice, iterations)
        auc_dmm = np.trapz(dmm, iterations)

        auc_results[case_id] = {"auc_dice": float(auc_dice), "auc_dmm": float(auc_dmm)}

    auc_output_file = output_dice_file.replace(".json", "_AUC.json")

    with open(auc_output_file, "w") as f:
        json.dump(auc_results, f, indent=4)

    logger.info(f"AUC results saved to: {auc_output_file}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()