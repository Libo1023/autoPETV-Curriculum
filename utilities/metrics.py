import os
import cc3d
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import json


def load_nii(path, return_spacing=False):
    nii = nib.load(path)
    data = nii.get_fdata()
    if return_spacing:
        return data, nii.header.get_zooms()
    return data


def calc_iou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    # Calculate intersection over union (IoU)
    if ground_truth.sum() == 0:
        return np.nan
    intersection = (ground_truth * prediction).sum()
    union = ground_truth.sum() + prediction.sum() - intersection
    if union == 0:
        return 0.0
    iou_score = intersection / union
    return iou_score


def calc_dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    # Calculate dice score
    if ground_truth.sum() == 0:
        return np.nan
    intersection = (ground_truth * prediction).sum()
    union = ground_truth.sum() + prediction.sum()
    dice_score = 2 * intersection / union
    return dice_score


def calc_tp_fp_fn(overlap_list, threshold, num_gt_instances, num_pred_instances):
    """
    From the overlap pairs + IoU threshold, derive TP/FP/FN.
    Multi-assignment is not punished: a GT lesion is TP if *any* pred
    lesion matches it above threshold (and vice versa for FP).
    """
    matched_gt = set()
    matched_pred = set()

    for ref_label, pred_label, overlap in overlap_list:
        if overlap >= threshold:
            matched_gt.add(ref_label)
            matched_pred.add(pred_label)

    tp = len(matched_gt)
    fn = num_gt_instances - tp
    fp = num_pred_instances - len(matched_pred)
    return tp, fp, fn


def calc_f1(tp, fp, fn):
    if tp + fn == 0:  # empty gt
        return np.nan
    if tp == 0:
        return 0.0
    return (2 * tp) / (2 * tp + fp + fn)


def calc_fpv_fnv(overlap_list, threshold, gt_volumes, pred_volumes):
    """
    FPV: sum of volumes of pred components with no match >= threshold.
    FNV: sum of volumes of GT components with no match >= threshold.
    Volumes are lists indexed by label-1.
    """
    matched_gt = set()
    matched_pred = set()
    for ref_label, pred_label, overlap in overlap_list:
        if overlap >= threshold:
            matched_gt.add(ref_label)
            matched_pred.add(pred_label)

    fpv = sum(
        pred_volumes[j - 1]
        for j in range(1, len(pred_volumes) + 1)
        if j not in matched_pred
    )
    fnv = (
        sum(
            gt_volumes[i - 1]
            for i in range(1, len(gt_volumes) + 1)
            if i not in matched_gt
        )
        if len(gt_volumes) > 0
        else np.nan
    )
    return fpv, fnv


def _get_bbox_nd(
    img: np.ndarray,
    px_dist: int | tuple[int, ...] = 0,
) -> tuple[slice, ...]:
    """calculates a bounding box in n dimensions given a image (factor ~2 times faster than compute_crop_slice)

    Args:
        img: input array
        px_dist: int | tuple[int]: dist (int): The amount of padding to be added to the cropped image.
        If int, will apply the same padding to each dim. Default value is 0.

    Returns:
        list of boundary coordinates [x_min, x_max, y_min, y_max, z_min, z_max]

    This part of the algorithm is based on panoptica (https://github.com/BrainLesion/panoptica)
    Licensed under the Apache License, Version 2.0
    Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
    """
    assert img is not None, "bbox_nd: received None as image"
    assert np.count_nonzero(img) > 0, "bbox_nd: img is empty, cannot calculate a bbox"
    N = img.ndim
    shp = img.shape
    if isinstance(px_dist, int):
        px_dist = np.ones(N, dtype=np.uint8) * px_dist
    assert (
        len(px_dist) == N
    ), f"dimension mismatch, got img shape {shp} and px_dist {px_dist}"

    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(a=img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    out = tuple(
        slice(
            max(out[i] - px_dist[i // 2], 0),
            min(out[i + 1] + px_dist[i // 2], shp[i // 2]) + 1,
        )
        for i in range(0, len(out), 2)
    )
    return out


def _get_paired_crop(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    px_pad: int = 2,
):
    """
    Calculates a bounding box based on paired prediction and reference arrays.

    Args:
        prediction_arr: The predicted segmentation array
        reference_arr: The ground truth segmentation array
        px_pad: Padding to apply around the bounding box

    Returns:
        np.ndarray: The bounding box coordinates around the combined non-zero regions

    This part of the algorithm is based on panoptica (https://github.com/BrainLesion/panoptica)
    Licensed under the Apache License, Version 2.0
    Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
    """
    assert prediction_arr.shape == reference_arr.shape

    combined = prediction_arr + reference_arr
    if combined.sum() == 0:
        combined += 1
    return _get_bbox_nd(combined, px_dist=px_pad)


def _calc_overlapping_labels(
    prediction_arr: np.ndarray,
    reference_arr: np.ndarray,
    ref_labels: tuple[int, ...],
) -> list[tuple[int, int]]:
    """
    Calculates the pairs of labels that overlap in at least one voxel.

    Args:
        prediction_arr: Array containing prediction labels
        reference_arr: Array containing reference labels
        ref_labels: List of unique reference labels

    Returns:
        list: Pairs of (ref_label, pred_label) that overlap

    This part of the algorithm is based on panoptica (https://github.com/BrainLesion/panoptica)
    Licensed under the Apache License, Version 2.0
    Copyright [2019] [Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany]
    """
    overlap_arr = prediction_arr.astype(np.uint32)
    max_ref = max(ref_labels) + 1
    overlap_arr = (overlap_arr * max_ref) + reference_arr
    overlap_arr[reference_arr == 0] = 0

    return [
        (int(i % (max_ref)), int(i // (max_ref)))
        for i in np.unique(overlap_arr)
        if i > max_ref
    ]


class MetricEvaluator:
    """
    Computes metrics for autoPETV challenge 2026.

    Licensed under the Apache License, Version 2.0
    Copyright [2026] [Jakob Dexl, Clinical Data Science, Munich, Germany]
    """

    def __init__(self, overlap_threshold=0.1, connectivity=18, save_dir=None):
        self.overlap_threshold = overlap_threshold
        self.connectivity = connectivity
        self.save_dir = save_dir
        self.metrics = {}

    def __call__(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        case_name: str,
        spacing=None,
        suv=None,
        return_meta=False,
    ):
        """
        prediction: binary prediction volume (numpy array)
        ground_truth: can be either a single class ground truth or a precomputed multiclass ground truth with unique
        lesions, this way you can track lesions across experiments
        case_name: file name for saving results, should follow center_tracer_XYZ convention
        spacing: needed for calculating size based metrics
        suv: needed to track lesions statistics
        """
        self.spacing = spacing
        self.original_shape = ground_truth.shape
        self.crop_shape = _get_paired_crop(prediction, np.clip(ground_truth, 0, 1), 2)
        self.suv = None
        self.metrics[case_name] = {}

        # Connected components of ground truth, thats the compute heavy part
        if ground_truth.max() > 1:  # Still calculates for zero and one lesion cases
            self.gt_multiclass = ground_truth.astype(int)[self.crop_shape]
            self.num_gt_instances = int(self.gt_multiclass.max())
        else:
            self.gt_multiclass, self.num_gt_instances = cc3d.connected_components(
                ground_truth.astype(int)[self.crop_shape],
                connectivity=self.connectivity,
                return_N=True,
            )

        # Connected components of prediction
        self.pred_multiclass, self.num_pred_instances = cc3d.connected_components(
            prediction.astype(int)[self.crop_shape],
            connectivity=self.connectivity,
            return_N=True,
        )

        # SUV values
        if suv is not None:
            self.suv = suv[self.crop_shape]

        # Match all lesions
        self.iou_list, self.dsc_list = self.compute_pairwise_lists()

        # Some metadata
        gt_vol, pred_vol = self.get_volumes()
        gt_suv, pred_suv = self.get_suv_max()

        self.meta = {
            "num_gt_instances": self.num_gt_instances,
            "num_pred_instances": self.num_pred_instances,
            "iou": self.iou_list,
            "dsc": self.dsc_list,
            "gt_volume": gt_vol,
            "pred_volume": pred_vol,
            "spacing": [float(x) for x in self.spacing]
            if self.spacing is not None
            else None,
            "gt_suv_max": gt_suv,
            "pred_suv_max": pred_suv,
        }

        # Calculate global metrics
        self.metrics[case_name]["dsc"] = calc_dice(prediction, ground_truth)

        # Calculate lesion based metrics
        tp, fp, fn = calc_tp_fp_fn(
            self.iou_list,
            self.overlap_threshold,
            self.num_gt_instances,
            self.num_pred_instances,
        )
        self.metrics[case_name]["tp"] = tp
        self.metrics[case_name]["fp"] = fp
        self.metrics[case_name]["fn"] = fn

        self.metrics[case_name]["f1"] = calc_f1(tp, fp, fn)

        # Extra FNV and FPV
        if spacing is not None:
            fpv, fnv = calc_fpv_fnv(self.iou_list, 0, gt_vol, pred_vol)
            self.metrics[case_name]["fpv"] = fpv * np.prod(spacing) / 1000
            self.metrics[case_name]["fnv"] = fnv * np.prod(spacing) / 1000

        if self.save_dir is not None:
            self._save_meta(case_name, self.save_dir)

        if return_meta:
            return {**self.metrics[case_name], **self.meta}

        return self.metrics[case_name]

    def reset(self):
        self.metrics = {}

    def aggregate(self, weighted=False):
        cases = list(self.metrics.keys())

        if weighted:
            ds_of = lambda c: "_".join(c.split("_")[:2])
            datasets = sorted(set(ds_of(c) for c in cases))
            if len(datasets) < 2:
                raise ValueError(
                    f"dataset_weighted requires ≥2 datasets, found: {datasets}"
                )

            ds_dice = [
                np.nanmean([self.metrics[c]["dsc"] for c in cases if ds_of(c) == ds])
                for ds in datasets
            ]
            ds_f1 = []
            for ds in datasets:
                dc = [c for c in cases if ds_of(c) == ds]
                ds_f1.append(
                    calc_f1(
                        sum(self.metrics[c]["tp"] for c in dc),
                        sum(self.metrics[c]["fp"] for c in dc),
                        sum(self.metrics[c]["fn"] for c in dc),
                    )
                )
            return {
                "dsc_weighted": float(np.nanmean(ds_dice)),
                "f1_aggregated_weighted": float(np.nanmean(ds_f1)),
            }

        dices = np.array([self.metrics[c]["dsc"] for c in cases])
        total_tp = sum(self.metrics[c]["tp"] for c in cases)
        total_fp = sum(self.metrics[c]["fp"] for c in cases)
        total_fn = sum(self.metrics[c]["fn"] for c in cases)
        return {
            "dsc": float(np.nanmean(dices)),
            "f1_aggregated": calc_f1(total_tp, total_fp, total_fn),
        }

    def compute_pairwise_lists(self):
        iou_list = []
        dsc_list = []

        gt_unique = np.unique(self.gt_multiclass)
        indices = _calc_overlapping_labels(
            self.pred_multiclass, self.gt_multiclass, gt_unique
        )

        for i, j in indices:
            mask_gt = self.gt_multiclass == i
            mask_pred = self.pred_multiclass == j
            iou = calc_iou(mask_pred, mask_gt)
            dsc = calc_dice(mask_pred, mask_gt)
            iou_list.append((i, j, iou))
            dsc_list.append((i, j, dsc))
        return iou_list, dsc_list

    def get_volumes(self):
        gts = []
        preds = []
        for i in range(1, self.gt_multiclass.max() + 1):
            gts.append(int(np.sum(self.gt_multiclass == i)))

        for i in range(1, self.pred_multiclass.max() + 1):
            preds.append(int(np.sum(self.pred_multiclass == i)))
        return gts, preds

    def get_suv_max(self):
        if self.suv is None:
            return None, None

        gts = []
        preds = []
        for i in range(1, self.gt_multiclass.max() + 1):
            gts.append(np.max(self.suv * (self.gt_multiclass == i)))

        for i in range(1, self.pred_multiclass.max() + 1):
            preds.append(np.max(self.suv * (self.pred_multiclass == i)))
        return gts, preds

    def _save_meta(self, case_name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        json_path = os.path.join(save_dir, f"{case_name.split('.')[0]}_meta.json")
        data = {"case_name": case_name, **self.metrics[case_name], **self.meta}
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_iou_matrix(self):
        overlap_matrix = np.zeros((self.num_gt_instances, self.num_pred_instances))
        for i, j, iou in self.iou_list:
            overlap_matrix[i - 1, j - 1] = iou
        return overlap_matrix


# Helper functions for tutorial
def make_circle(shape, center, radius):
    """Create a binary 2D circle mask."""
    Y, X = np.ogrid[: shape[0], : shape[1]]
    return ((X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius**2).astype(np.uint8)


def to_volume(mask_2d):
    """Expand 2D to 3D (single slice) for cc3d compatibility."""
    return mask_2d[:, :, np.newaxis]


SHAPE = (128, 128)


def case_low_iou():
    """3 GT lesions, 3 pred. One pred barely touches GT lesion 3 -> IoU < 0.1."""
    gt = np.zeros(SHAPE, dtype=np.uint8)
    gt += make_circle(SHAPE, (30, 30), 12) * 1
    gt += make_circle(SHAPE, (80, 30), 10) * 2
    gt += make_circle(SHAPE, (55, 90), 14) * 3

    pred = np.zeros(SHAPE, dtype=np.uint8)
    pred += make_circle(SHAPE, (30, 30), 11)
    pred += make_circle(SHAPE, (80, 30), 9)
    pred += make_circle(SHAPE, (55, 70), 8)  # shifted far from GT 3

    return to_volume(gt), to_volume(np.clip(pred, 0, 1)), "fdg_ukt_001"


def case_pure_fp():
    """3 GT lesions, 2 matched pred + 1 pure FP with no GT overlap, one pure FN."""
    gt = np.zeros(SHAPE, dtype=np.uint8)
    gt += make_circle(SHAPE, (40, 40), 12) * 1
    gt += make_circle(SHAPE, (90, 90), 10) * 2
    gt += make_circle(SHAPE, (55, 90), 1) * 3

    pred = np.zeros(SHAPE, dtype=np.uint8)
    pred += make_circle(SHAPE, (40, 40), 11)
    pred += make_circle(SHAPE, (90, 90), 9)
    pred += make_circle(SHAPE, (20, 100), 8)  # pure FP

    return to_volume(gt), to_volume(np.clip(pred, 0, 1)), "fdg_ukt_002"


def case_empty_gt():
    """Empty GT, one FP prediction."""
    gt = np.zeros(SHAPE, dtype=np.uint8)
    pred = np.zeros(SHAPE, dtype=np.uint8)
    pred += make_circle(SHAPE, (64, 64), 10)

    return to_volume(gt), to_volume(pred), "psma_lmu_001"


def case_two_preds_one_gt():
    """2 GT lesions, 3 pred. Two pred circles both overlap GT lesion 1 with IoU > 0.1."""
    gt = np.zeros(SHAPE, dtype=np.uint8)
    gt += make_circle(SHAPE, (50, 50), 18) * 1  # large GT lesion
    gt += make_circle(SHAPE, (100, 90), 10) * 2

    pred = np.zeros(SHAPE, dtype=np.uint8)
    pred += make_circle(SHAPE, (35, 50), 10)  # overlaps GT 1 left side
    pred += make_circle(SHAPE, (65, 50), 10)  # overlaps GT 1 right side
    pred += make_circle(SHAPE, (100, 90), 9)  # matches GT 2

    return to_volume(gt), to_volume(np.clip(pred, 0, 1)), "psma_lmu_002"


def all_cases():
    """Returns list of (gt, pred, case_name) tuples."""
    return [case_low_iou(), case_pure_fp(), case_empty_gt(), case_two_preds_one_gt()]


def plot_cases(cases=None):
    """Plot all test cases in a 2x2 grid, each cell has GT and pred side by side."""
    if cases is None:
        cases = all_cases()
    cmap = ListedColormap(
        ["black", "#e74c3c", "#2ecc71", "#3498db", "#f1c40f", "#9b59b6"]
    )
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (gt, pred, name) in enumerate(cases):
        row = idx // 2
        col = (idx % 2) * 2
        gt_2d = gt[:, :, 0]
        pred_2d = pred[:, :, 0]
        axes[row, col].imshow(gt_2d, cmap=cmap, vmin=0, vmax=5, interpolation="nearest")
        axes[row, col].set_title(f"{name} — GT")
        axes[row, col].axis("off")
        axes[row, col + 1].imshow(
            pred_2d * 4, cmap=cmap, vmin=0, vmax=5, interpolation="nearest"
        )
        axes[row, col + 1].set_title(f"{name} — Pred")
        axes[row, col + 1].axis("off")
    plt.tight_layout()


def print_matrix(matrix, row_prefix="G", col_prefix="P"):
    """Print a labeled overlap matrix."""
    num_rows, num_cols = matrix.shape
    if num_rows == 0 and num_cols == 0:
        print("[]")
        return
    col_labels = [f"{col_prefix}{i + 1}" for i in range(num_cols)]
    row_w = max(len(f"{row_prefix}{i + 1}") for i in range(num_rows))
    col_w = 7
    print(" " * (row_w + 2) + "   ".join(f"{l:<{col_w}}" for l in col_labels))
    for i in range(num_rows):
        label = f"{row_prefix}{i + 1}"
        vals = "   ".join(f"{x:.3f}  " for x in matrix[i])
        print(f"{label:<{row_w}}  {vals}")
