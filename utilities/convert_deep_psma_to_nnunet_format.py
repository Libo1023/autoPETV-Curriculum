import os
import argparse
from pathlib import Path
import shutil

import nibabel as nib
from nibabel.processing import resample_from_to


def resample_ct_to_pet(ct_img, pet_img):
    """
    Resample CT image to PET space using linear interpolation.
    """
    return resample_from_to(ct_img, pet_img, order=1)  # linear


def copy_vol(src, dst):
    shutil.copy2(src, dst)



def process_case(case_dir, tracer, case_id, imagesTr, labelsTr):
    tracer_dir = case_dir / tracer

    ct_path = tracer_dir / "CT.nii.gz"
    pet_path = tracer_dir / "PET.nii.gz"
    label_path = tracer_dir / "TTB.nii.gz"

    if not (ct_path.exists() and pet_path.exists() and label_path.exists()):
        print(f"[WARN] Missing files in {tracer_dir}")
        return

    # Load images
    ct_img = nib.load(ct_path)
    pet_img = nib.load(pet_path)
    label_img = nib.load(label_path)

    # Resample CT -> PET space
    ct_resampled = resample_ct_to_pet(ct_img, pet_img)

    # Naming
    base_name = f"{tracer}_{case_id:04d}"

    ct_out = imagesTr / f"{base_name}_0000.nii.gz"
    pet_out = imagesTr / f"{base_name}_0001.nii.gz"
    label_out = labelsTr / f"{base_name}.nii.gz"

    # Save CT (resampled)
    nib.save(ct_resampled, ct_out)

    # PET can be copied directly
    copy_vol(pet_path, pet_out)

    # Label (already aligned to PET, no resampling)
    copy_vol(label_path, label_out)

    print(f"[OK] {base_name}")


def main():
    parser = argparse.ArgumentParser(description="Convert DEEP-PSMA dataset to nnU-Net format")

    parser.add_argument("--root", type=str, required=True, help="Path to DEEP-PSMA root")
    parser.add_argument("--dest", type=str, required=True, help="Output nnU-Net dataset folder")

    args = parser.parse_args()

    root = Path(args.root)
    dest = Path(args.dest)

    imagesTr = dest / "imagesTr"
    labelsTr = dest / "labelsTr"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([d for d in root.glob("train_*") if d.is_dir()])

    print(f"Found {len(case_dirs)} cases")

    for case_dir in case_dirs:
        case_str = case_dir.name.split("_")[-1]
        case_id = int(case_str)

        for tracer in ["PSMA", "FDG"]:
            process_case(
                case_dir=case_dir,
                tracer=tracer,
                case_id=case_id,
                imagesTr=imagesTr,
                labelsTr=labelsTr,
            )


if __name__ == "__main__":
    main()
