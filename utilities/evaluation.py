"""
baseline_evaluation.py

Local evaluation for autoPET V nnU-Net models.
Supports the downloaded baseline, ResEnc-M, and custom-trainer models.
Multi-GPU parallel evaluation with round-robin scribble strategies.

Usage examples:

  # Evaluate our custom model (ResEnc-M + nnUNetTrainerAutoPETV)
  python baseline_evaluation.py \
      --model_results_dir nnUNet_results \
      --plans_name nnUNetResEncUNetMPlans_40G \
      --trainer_name nnUNetTrainerAutoPETV \
      --result_dir results/custom_eval

  # Evaluate with best checkpoint
  python baseline_evaluation.py \
      --model_results_dir nnUNet_results \
      --plans_name nnUNetResEncUNetMPlans_40G \
      --trainer_name nnUNetTrainerAutoPETV \
      --chk checkpoint_best.pth \
      --result_dir results/custom_eval_best

  # Resume an interrupted run
  python baseline_evaluation.py \
      --model_results_dir nnUNet_results \
      --plans_name nnUNetResEncUNetMPlans_40G \
      --trainer_name nnUNetTrainerAutoPETV \
      --result_dir results/custom_eval \
      --resume
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import time
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
            self.n = 0
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def update(self, n=1):
            self.n += n
        def set_postfix_str(self, s, refresh=True):
            pass
        def close(self):
            pass
        @staticmethod
        def write(s):
            print(s)


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "interactive"))

from simulate_scribbles import simulate_scribble_from_label, heatmap_from_coords
from metrics import MetricEvaluator

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

STRATEGIES = ["centerline", "random", "boundary"]


# ===============================================================
# Utilities
# ===============================================================

def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def fmt_float(x, width=8):
    if x is None or (isinstance(x, float) and x != x):
        return "{:>{w}s}".format("NaN", w=width)
    return "{:>{w}.4f}".format(x, w=width)


def to_python_float(x):
    if x is None:
        return float("nan")
    return float(x)


def nanmean_safe(values):
    valid = [v for v in values if v is not None and v == v]
    return float(np.mean(valid)) if valid else float("nan")


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if v != v else v
    return obj


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(sanitize_for_json(obj), f, indent=2)


def safe_simulate_scribble(mask, strategy, seed=42):
    if np.sum(mask) == 0:
        return [], 0
    try:
        result = simulate_scribble_from_label(mask, strategy, seed)
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            return result[0], int(result[2])
        return [], 0
    except Exception:
        return [], 0


def make_heatmap(coords, shape, sigma=0.0):
    if not coords:
        return np.zeros(shape, dtype=np.float32)
    return heatmap_from_coords(coords, shape, sigma=sigma).astype(np.float32)


def compute_auc(values, steps):
    values = np.asarray(values, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    if len(values) < 2 or np.any(np.isnan(values)):
        return float("nan")
    return float(np.trapz(values, steps))


def load_val_cases(splits_path, fold):
    with open(splits_path, "r") as f:
        splits = json.load(f)
    if fold < 0 or fold >= len(splits):
        raise ValueError("Fold {} out of range ({} folds).".format(fold, len(splits)))
    return sorted(splits[fold]["val"])


def assign_strategies(cases):
    return {c: STRATEGIES[i % len(STRATEGIES)] for i, c in enumerate(cases)}


def build_predict_cmd(temp_in, temp_out, dataset_id, fold,
                      plans_name, trainer_name, chk):
    cmd = [
        "nnUNetv2_predict",
        "-i", temp_in,
        "-o", temp_out,
        "-d", str(dataset_id),
        "-c", "3d_fullres",
        "-f", str(fold),
        "--disable_tta",
    ]
    if plans_name is not None:
        cmd.extend(["-p", plans_name])
    if trainer_name is not None:
        cmd.extend(["-tr", trainer_name])
    if chk is not None:
        cmd.extend(["-chk", chk])
    return cmd


# ===============================================================
# Worker
# ===============================================================

def worker_evaluate_case(task):
    case_name = task["case_name"]
    strategy = task["strategy"]
    gpu_id = task["gpu_id"]
    task_id = task["task_id"]
    images_dir = task["images_dir"]
    labels_dir = task["labels_dir"]
    result_dir = task["result_dir"]
    max_iters = task["max_iters"]
    repo_root = task["repo_root"]
    model_results_dir = task["model_results_dir"]
    plans_name = task["plans_name"]
    trainer_name = task["trainer_name"]
    dataset_id = task["dataset_id"]
    fold = task["fold"]
    chk = task["chk"]
    ext_trainer_dir = task["ext_trainer_dir"]

    case_dir = os.path.join(result_dir, case_name)
    temp_in = os.path.join(result_dir, "_tmp_{:04d}_in".format(task_id))
    temp_out = os.path.join(result_dir, "_tmp_{:04d}_out".format(task_id))
    log_lines = []

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["nnUNet_raw"] = repo_root
    env["nnUNet_preprocessed"] = repo_root
    env["nnUNet_results"] = model_results_dir
    env["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::DeprecationWarning"
    if ext_trainer_dir and os.path.isdir(ext_trainer_dir):
        env["nnUNet_extTrainer"] = ext_trainer_dir

    t_start = time.time()

    def cleanup():
        for d in [temp_in, temp_out]:
            if os.path.exists(d):
                shutil.rmtree(d)

    try:
        os.makedirs(case_dir, exist_ok=True)

        gt_obj = nib.load(os.path.join(labels_dir, "{}.nii.gz".format(case_name)))
        gt = gt_obj.get_fdata().astype(np.uint8)
        spacing = tuple(float(s) for s in gt_obj.header.get_zooms())
        gt_voxels = int(np.sum(gt))
        empty_gt = gt_voxels == 0

        pet_obj = nib.load(os.path.join(images_dir, "{}_0001.nii.gz".format(case_name)))
        pet_shape = pet_obj.shape
        pet_affine = pet_obj.affine

        log_lines.append("  gt voxels: {}{}".format(
            gt_voxels, "  (EMPTY -- no lesions)" if empty_gt else ""))
        log_lines.append("  volume shape: {}".format(pet_shape))

        clean_dir(temp_in)
        for ch in ("0000", "0001"):
            shutil.copy2(
                os.path.join(images_dir, "{}_{}.nii.gz".format(case_name, ch)),
                os.path.join(temp_in, "{}_{}.nii.gz".format(case_name, ch)))

        scribble_data = {"tumor": [], "background": []}
        case_metrics = []

        for it in range(max_iters):
            t_iter = time.time()

            fg_hm = make_heatmap(scribble_data["tumor"], pet_shape)
            bg_hm = make_heatmap(scribble_data["background"], pet_shape)
            nib.save(nib.Nifti1Image(fg_hm, pet_affine),
                     os.path.join(temp_in, "{}_0002.nii.gz".format(case_name)))
            nib.save(nib.Nifti1Image(bg_hm, pet_affine),
                     os.path.join(temp_in, "{}_0003.nii.gz".format(case_name)))

            with open(os.path.join(case_dir, "iter_{}_scribbles.json".format(it)), "w") as fh:
                json.dump(scribble_data, fh, indent=2)

            clean_dir(temp_out)
            cmd = build_predict_cmd(temp_in, temp_out, dataset_id, fold,
                                    plans_name, trainer_name, chk)
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
            if proc.returncode != 0:
                snippet = (proc.stderr or "")[-500:]
                raise RuntimeError("nnUNet exit code {}: {}".format(proc.returncode, snippet))

            pred_path = os.path.join(temp_out, "{}.nii.gz".format(case_name))
            if not os.path.isfile(pred_path):
                raise FileNotFoundError("Prediction not found: {}".format(pred_path))
            pred = nib.load(pred_path).get_fdata().astype(np.uint8)

            if pred.shape != gt.shape:
                raise ValueError("Shape mismatch: pred {} vs gt {}".format(pred.shape, gt.shape))

            evaluator = MetricEvaluator(overlap_threshold=0.1, connectivity=18)
            m = evaluator(pred, gt, case_name, spacing=spacing)

            dice_val = to_python_float(m.get("dsc"))
            dmm_val = to_python_float(m.get("f1"))
            tp = int(m.get("tp", 0))
            fp = int(m.get("fp", 0))
            fn = int(m.get("fn", 0))

            case_metrics.append({"iteration": it, "dice": dice_val, "dmm": dmm_val,
                                 "tp": tp, "fp": fp, "fn": fn})

            dt = time.time() - t_iter
            log_lines.append(
                "  step {}: Dice={}  DMM={}  (TP={} FP={} FN={})  "
                "scribbles: fg={} bg={}  ({:.1f}s)".format(
                    it, fmt_float(dice_val), fmt_float(dmm_val),
                    tp, fp, fn,
                    len(scribble_data["tumor"]), len(scribble_data["background"]), dt))

            if it < max_iters - 1:
                if not empty_gt:
                    overseg = ((pred == 1) & (gt == 0)).astype(np.uint8)
                    underseg = ((pred == 0) & (gt == 1)).astype(np.uint8)
                    scr_bg, fp_sz = safe_simulate_scribble(overseg, strategy)
                    scr_fg, fn_sz = safe_simulate_scribble(underseg, strategy)
                    if fp_sz <= fn_sz and scr_fg:
                        scribble_data["tumor"] += scr_fg
                    elif scr_bg:
                        scribble_data["background"] += scr_bg
                else:
                    if np.sum(pred) > 0:
                        scr_bg, _ = safe_simulate_scribble(pred.astype(np.uint8), strategy)
                        if scr_bg:
                            scribble_data["background"] += scr_bg

        iters_l = [r["iteration"] for r in case_metrics]
        dices_l = [r["dice"] for r in case_metrics]
        dmms_l = [r["dmm"] for r in case_metrics]
        auc_d = compute_auc(dices_l, iters_l)
        auc_m = compute_auc(dmms_l, iters_l)

        elapsed = time.time() - t_start
        log_lines.append("  >> AUC-Dice={}  AUC-DMM={}  ({:.1f}s, {:.1f} min)".format(
            fmt_float(auc_d), fmt_float(auc_m), elapsed, elapsed / 60.0))

        cleanup()
        return {"case_name": case_name, "strategy": strategy, "gpu_id": gpu_id,
                "metrics": case_metrics, "auc_dice": auc_d, "auc_dmm": auc_m,
                "log_lines": log_lines, "elapsed": elapsed, "success": True}

    except Exception as exc:
        elapsed = time.time() - t_start
        log_lines.append("  FAILED: {}".format(exc))
        cleanup()
        return {"case_name": case_name, "strategy": strategy, "gpu_id": gpu_id,
                "metrics": [{"iteration": i, "dice": 0.0, "dmm": 0.0,
                             "tp": 0, "fp": 0, "fn": 0} for i in range(max_iters)],
                "auc_dice": 0.0, "auc_dmm": 0.0,
                "log_lines": log_lines, "elapsed": elapsed, "success": False}


# ===============================================================
# Summary
# ===============================================================

def compute_summary(all_results):
    auc_records = []
    for cname, data in all_results.items():
        recs = sorted(data["metrics"], key=lambda r: r["iteration"])
        iters = [r["iteration"] for r in recs]
        dices = [r["dice"] if r["dice"] is not None else float("nan") for r in recs]
        dmms = [r["dmm"] if r["dmm"] is not None else float("nan") for r in recs]
        tracer = "fdg" if cname.startswith("fdg") else "psma"
        auc_records.append({"case": cname, "tracer": tracer, "strategy": data["strategy"],
                            "auc_dice": compute_auc(dices, iters),
                            "auc_dmm": compute_auc(dmms, iters)})

    mean_d = nanmean_safe([r["auc_dice"] for r in auc_records])
    mean_m = nanmean_safe([r["auc_dmm"] for r in auc_records])

    tracers = sorted(set(r["tracer"] for r in auc_records))
    per_tracer = {}
    for t in tracers:
        td = [r["auc_dice"] for r in auc_records if r["tracer"] == t]
        tm = [r["auc_dmm"] for r in auc_records if r["tracer"] == t]
        per_tracer[t] = {"count": len(td), "mean_auc_dice": nanmean_safe(td),
                         "mean_auc_dmm": nanmean_safe(tm)}

    if len(tracers) >= 2:
        w_d = float(np.mean([per_tracer[t]["mean_auc_dice"] for t in tracers]))
        w_m = float(np.mean([per_tracer[t]["mean_auc_dmm"] for t in tracers]))
    else:
        w_d, w_m = mean_d, mean_m

    per_strategy = {}
    for s in STRATEGIES:
        sd = [r["auc_dice"] for r in auc_records if r["strategy"] == s]
        sm = [r["auc_dmm"] for r in auc_records if r["strategy"] == s]
        if sd:
            per_strategy[s] = {"count": len(sd), "mean_auc_dice": nanmean_safe(sd),
                               "mean_auc_dmm": nanmean_safe(sm)}

    return {"auc_records": auc_records, "mean_auc_dice": mean_d, "mean_auc_dmm": mean_m,
            "weighted_auc_dice": w_d, "weighted_auc_dmm": w_m,
            "per_tracer": per_tracer, "per_strategy": per_strategy}


def print_summary(log_fn, summary, num_cases, elapsed, model_label):
    log_fn("")
    log_fn("=" * 68)
    log_fn("EVALUATION COMPLETE -- {}".format(model_label))
    log_fn("=" * 68)
    log_fn("  Cases evaluated   : {}".format(num_cases))
    log_fn("  Total time        : {:.0f}s ({:.1f} min, {:.1f} h)".format(
        elapsed, elapsed / 60.0, elapsed / 3600.0))
    log_fn("")
    log_fn("  --- Overall (simple average) ---")
    log_fn("  Mean AUC-Dice     : {}".format(fmt_float(summary["mean_auc_dice"])))
    log_fn("  Mean AUC-DMM      : {}".format(fmt_float(summary["mean_auc_dmm"])))
    log_fn("")
    log_fn("  --- Dataset-weighted ---")
    log_fn("  Weighted AUC-Dice : {}".format(fmt_float(summary["weighted_auc_dice"])))
    log_fn("  Weighted AUC-DMM  : {}".format(fmt_float(summary["weighted_auc_dmm"])))
    log_fn("")
    log_fn("  --- Per tracer ---")
    for t in sorted(summary["per_tracer"].keys()):
        info = summary["per_tracer"][t]
        log_fn("  {:6s} (n={:4d}):  AUC-Dice={}   AUC-DMM={}".format(
            t, info["count"], fmt_float(info["mean_auc_dice"]),
            fmt_float(info["mean_auc_dmm"])))
    log_fn("")
    log_fn("  --- Per strategy ---")
    for s in STRATEGIES:
        if s in summary["per_strategy"]:
            info = summary["per_strategy"][s]
            log_fn("  {:12s} (n={:4d}):  AUC-Dice={}   AUC-DMM={}".format(
                s, info["count"], fmt_float(info["mean_auc_dice"]),
                fmt_float(info["mean_auc_dmm"])))
    log_fn("=" * 68)


# ===============================================================
# Main
# ===============================================================

def main():
    parser = argparse.ArgumentParser(
        description="autoPET V evaluation for baseline and custom models.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(REPO_ROOT, "data", "PSMA-FDG-PET-CT-Lesions_v2"))
    parser.add_argument("--model_results_dir", type=str,
                        default=os.path.join(REPO_ROOT, "nnUNet_results"),
                        help="Path to nnUNet_results with trained weights.")
    parser.add_argument("--plans_name", type=str, default=None,
                        help="Plans identifier (-p flag). None = default nnUNetPlans.")
    parser.add_argument("--trainer_name", type=str, default=None,
                        help="Trainer class name (-tr flag). None = default nnUNetTrainer.")
    parser.add_argument("--ext_trainer_dir", type=str,
                        default=os.path.join(REPO_ROOT, "custom_trainers"),
                        help="Directory with custom trainer .py files.")
    parser.add_argument("--dataset_id", type=int, default=998)
    parser.add_argument("--chk", type=str, default=None,
                        help="Checkpoint file (e.g. checkpoint_best.pth).")
    parser.add_argument("--result_dir", type=str,
                        default=os.path.join(REPO_ROOT, "results", "custom_eval"))
    parser.add_argument("--max_iters", type=int, default=6)
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")
    parser.add_argument("--workers_per_gpu", type=int, default=4)

    args = parser.parse_args()

    if not os.path.isabs(args.model_results_dir):
        args.model_results_dir = os.path.join(REPO_ROOT, args.model_results_dir)
    if not os.path.isabs(args.ext_trainer_dir):
        args.ext_trainer_dir = os.path.join(REPO_ROOT, args.ext_trainer_dir)

    if args.trainer_name:
        model_label = "{} ({})".format(args.trainer_name, args.plans_name or "nnUNetPlans")
    else:
        model_label = "Baseline (nnUNetPlans)"

    gpu_ids = [g.strip() for g in args.gpu_ids.split(",")]
    total_workers = len(gpu_ids) * args.workers_per_gpu

    images_dir = os.path.join(args.data_dir, "imagesTr")
    labels_dir = os.path.join(args.data_dir, "labelsTr")
    splits_path = os.path.join(args.data_dir, "splits_final.json")
    output_json = os.path.join(args.result_dir, "all_results.json")
    log_path = os.path.join(args.result_dir, "evaluation.log")

    for p, desc in [(images_dir, "imagesTr"), (labels_dir, "labelsTr"),
                     (splits_path, "splits_final.json"),
                     (args.model_results_dir, "model_results_dir")]:
        if not os.path.exists(p):
            print("[ERROR] {} not found: {}".format(desc, p))
            sys.exit(1)

    os.makedirs(args.result_dir, exist_ok=True)
    log_fh = open(log_path, "a")

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        tqdm.write(msg)
        log_fh.write("[{}] {}\n".format(ts, msg))
        log_fh.flush()

    all_val = load_val_cases(splits_path, args.fold)
    valid_cases = []
    skipped = 0
    for c in all_val:
        ct = os.path.join(images_dir, "{}_0000.nii.gz".format(c))
        pet = os.path.join(images_dir, "{}_0001.nii.gz".format(c))
        lbl = os.path.join(labels_dir, "{}.nii.gz".format(c))
        if os.path.isfile(ct) and os.path.isfile(pet) and os.path.isfile(lbl):
            valid_cases.append(c)
        else:
            skipped += 1

    if skipped > 0:
        log("[WARN] {} val cases skipped (missing files).".format(skipped))

    num_valid_total = len(valid_cases)
    strategy_map = assign_strategies(valid_cases)

    if args.max_cases is not None and args.max_cases < len(valid_cases):
        valid_cases = valid_cases[:args.max_cases]

    if args.resume and os.path.isfile(output_json):
        with open(output_json, "r") as f:
            all_results = json.load(f)
        log("[RESUME] Loaded {} existing results.".format(len(all_results)))
    else:
        all_results = {}

    remaining = [c for c in valid_cases if c not in all_results]

    strat_counts = {}
    for c in valid_cases:
        s = strategy_map[c]
        strat_counts[s] = strat_counts.get(s, 0) + 1

    log("=" * 68)
    log("autoPET V Evaluation ({})".format(model_label))
    log("=" * 68)
    log("  Model            : {}".format(model_label))
    log("  Model weights    : {}".format(args.model_results_dir))
    if args.plans_name:
        log("  Plans            : {}".format(args.plans_name))
    if args.trainer_name:
        log("  Trainer          : {}".format(args.trainer_name))
    log("  Checkpoint       : {}".format(args.chk or "checkpoint_final.pth (default)"))
    log("  Dataset          : {}".format(args.data_dir))
    log("  Fold             : {}".format(args.fold))
    log("  Total val cases  : {}".format(len(all_val)))
    log("  Valid (on disk)  : {}".format(num_valid_total))
    log("  Evaluating       : {}".format(len(valid_cases)))
    log("  Already done     : {}".format(len(valid_cases) - len(remaining)))
    log("  To evaluate      : {}".format(len(remaining)))
    log("  Interaction steps: {}".format(args.max_iters))
    log("  Strategy dist.   : {}".format(strat_counts))
    log("  GPUs             : {} ({})".format(len(gpu_ids), ", ".join(gpu_ids)))
    log("  Workers per GPU  : {}".format(args.workers_per_gpu))
    log("  Total workers    : {}".format(total_workers))
    log("=" * 68)

    if len(remaining) == 0:
        log("[INFO] All cases done. Computing summary.")
        elapsed_eval = 0.0
    else:
        t_global = time.time()

        tasks = []
        for i, cname in enumerate(remaining):
            tasks.append({
                "task_id": i, "case_name": cname,
                "strategy": strategy_map[cname],
                "gpu_id": gpu_ids[i % len(gpu_ids)],
                "images_dir": images_dir, "labels_dir": labels_dir,
                "result_dir": args.result_dir, "max_iters": args.max_iters,
                "repo_root": REPO_ROOT,
                "model_results_dir": args.model_results_dir,
                "plans_name": args.plans_name,
                "trainer_name": args.trainer_name,
                "ext_trainer_dir": args.ext_trainer_dir,
                "dataset_id": args.dataset_id,
                "fold": args.fold, "chk": args.chk,
            })

        running_auc_d = []
        running_auc_m = []
        completed = 0
        failed = 0

        pbar = tqdm(total=len(tasks), desc="Evaluating", unit="case", position=0,
                    dynamic_ncols=True,
                    bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"))

        with ProcessPoolExecutor(max_workers=total_workers) as executor:
            future_map = {executor.submit(worker_evaluate_case, t): t for t in tasks}
            for future in as_completed(future_map):
                task_info = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "case_name": task_info["case_name"],
                        "strategy": task_info["strategy"],
                        "gpu_id": task_info["gpu_id"],
                        "metrics": [{"iteration": i, "dice": 0.0, "dmm": 0.0,
                                     "tp": 0, "fp": 0, "fn": 0} for i in range(args.max_iters)],
                        "auc_dice": 0.0, "auc_dmm": 0.0,
                        "log_lines": ["  WORKER ERROR: {}".format(exc)],
                        "elapsed": 0.0, "success": False}

                completed += 1
                if not result["success"]:
                    failed += 1

                log("")
                log("-" * 68)
                log("[{:03d}/{:03d}] [GPU {}] {}  [{}]  ({:.1f}s)".format(
                    completed, len(tasks), result["gpu_id"],
                    result["case_name"], result["strategy"], result["elapsed"]))
                for line in result["log_lines"]:
                    log(line)

                all_results[result["case_name"]] = {
                    "strategy": result["strategy"], "metrics": result["metrics"]}
                save_json(all_results, output_json)

                running_auc_d.append(result["auc_dice"])
                running_auc_m.append(result["auc_dmm"])
                status = "D={:.3f} M={:.3f}".format(
                    nanmean_safe(running_auc_d), nanmean_safe(running_auc_m))
                if failed > 0:
                    status += " fail={}".format(failed)
                pbar.update(1)
                pbar.set_postfix_str(status, refresh=True)

        pbar.close()
        elapsed_eval = time.time() - t_global

    for entry in os.listdir(args.result_dir):
        if entry.startswith("_tmp_"):
            p = os.path.join(args.result_dir, entry)
            if os.path.isdir(p):
                shutil.rmtree(p)

    summary = compute_summary(all_results)
    save_json(summary["auc_records"], os.path.join(args.result_dir, "auc_per_case.json"))
    summary_out = {
        "model": model_label, "model_results_dir": args.model_results_dir,
        "plans_name": args.plans_name, "trainer_name": args.trainer_name,
        "checkpoint": args.chk or "checkpoint_final.pth",
        "fold": args.fold, "num_cases": len(all_results), "max_iters": args.max_iters,
        "mean_auc_dice": summary["mean_auc_dice"], "mean_auc_dmm": summary["mean_auc_dmm"],
        "weighted_auc_dice": summary["weighted_auc_dice"],
        "weighted_auc_dmm": summary["weighted_auc_dmm"],
        "per_tracer": summary["per_tracer"], "per_strategy": summary["per_strategy"],
        "total_workers": total_workers, "gpu_ids": gpu_ids,
        "total_time_seconds": round(elapsed_eval, 1)}
    save_json(summary_out, os.path.join(args.result_dir, "summary.json"))
    print_summary(log, summary, len(all_results), elapsed_eval, model_label)
    log_fh.close()


if __name__ == "__main__":
    main()