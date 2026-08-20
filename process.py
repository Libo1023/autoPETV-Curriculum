import json
import os
import shutil
import subprocess

import numpy as np
import nibabel as nib
import SimpleITK
from scipy.ndimage import gaussian_filter


class AutoPETVAlgorithm:

    def __init__(self):
        self.input_path = "/input/"
        self.output_path = "/output/images/tumor-lesion-segmentation/"

        self.nii_dir = "/opt/algorithm/_tmp_nii"
        self.result_dir = "/opt/algorithm/_tmp_result"

        self.dataset_id = 998
        self.configuration = "3d_fullres"
        # self.folds = "0 1 2 3 4" 
        self.folds = "0 1 2 3 4 5 6 7 8 9" 
        self.trainer = "nnUNetTrainerAutoPETV"
        self.plans = "nnUNetResEncUNetMPlans_40G"

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def convert_mha_to_nii(mha_path, nii_path):
        img = SimpleITK.ReadImage(mha_path)
        SimpleITK.WriteImage(img, nii_path, True)

    @staticmethod
    def convert_nii_to_mha(nii_path, mha_path):
        img = SimpleITK.ReadImage(nii_path)
        SimpleITK.WriteImage(img, mha_path, True)

    # ------------------------------------------------------------------
    # Clicks JSON -> heatmap NIfTI
    # ------------------------------------------------------------------

    @staticmethod
    def gc_json_to_clicks(gc_json_path):
        """Parse Grand Challenge lesion-clicks.json into simple dict."""
        with open(gc_json_path, "r") as f:
            gc_data = json.load(f)
        clicks = {"tumor": [], "background": []}
        for point in gc_data.get("points", []):
            name = point.get("name", "")
            coord = point.get("point", [])
            if name in clicks and len(coord) == 3:
                clicks[name].append(coord)
        return clicks

    @staticmethod
    def make_heatmap(coords, shape, sigma=0.0):
        """Create a float32 heatmap volume from coordinate list."""
        hm = np.zeros(shape, dtype=np.float32)
        for c in coords:
            x, y, z = int(c[0]), int(c[1]), int(c[2])
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                hm[x, y, z] = 1.0
        if sigma > 0:
            hm = gaussian_filter(hm, sigma=sigma)
        return hm

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def load_inputs(self):
        """Read CT, PET, and clicks; convert to nnUNet 4-channel NIfTI."""

        # Find the .mha files (filenames are random UUIDs)
        ct_dir = os.path.join(self.input_path, "images", "ct")
        pet_dir = os.path.join(self.input_path, "images", "pet")

        ct_mha = os.listdir(ct_dir)[0]
        pet_mha = os.listdir(pet_dir)[0]

        # UUID is the stem of the CT filename (used for output naming)
        uuid = os.path.splitext(ct_mha)[0]
        if uuid.endswith(".mha"):
            uuid = uuid[:-4]

        # Prepare temp directories
        for d in [self.nii_dir, self.result_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        # Convert CT and PET to NIfTI
        case_id = "CASE_001"
        self.convert_mha_to_nii(
            os.path.join(ct_dir, ct_mha),
            os.path.join(self.nii_dir, "{}_0000.nii.gz".format(case_id)),
        )
        self.convert_mha_to_nii(
            os.path.join(pet_dir, pet_mha),
            os.path.join(self.nii_dir, "{}_0001.nii.gz".format(case_id)),
        )

        # Read clicks and generate heatmaps
        clicks_file = os.path.join(self.input_path, "lesion-clicks.json")
        clicks = self.gc_json_to_clicks(clicks_file)

        pet_nib = nib.load(
            os.path.join(self.nii_dir, "{}_0001.nii.gz".format(case_id))
        )
        ref_shape = pet_nib.shape
        ref_affine = pet_nib.affine

        fg_hm = self.make_heatmap(clicks["tumor"], ref_shape, sigma=0)
        bg_hm = self.make_heatmap(clicks["background"], ref_shape, sigma=0)

        nib.save(
            nib.Nifti1Image(fg_hm, ref_affine),
            os.path.join(self.nii_dir, "{}_0002.nii.gz".format(case_id)),
        )
        nib.save(
            nib.Nifti1Image(bg_hm, ref_affine),
            os.path.join(self.nii_dir, "{}_0003.nii.gz".format(case_id)),
        )

        print("[load_inputs] 4-channel input ready: {}".format(
            os.listdir(self.nii_dir)))

        return uuid

    def predict(self):
        """Run nnUNetv2_predict with 3-fold ensemble."""
        cmd = (
            "nnUNetv2_predict"
            " -i {nii_dir}"
            " -o {result_dir}"
            " -d {dataset_id}"
            " -c {config}"
            " -f {folds}"
            " -tr {trainer}"
            " -p {plans}"
            " --disable_tta"
            " --verbose"
            # " --disable_progress_bar"
        ).format(
            nii_dir=self.nii_dir,
            result_dir=self.result_dir,
            dataset_id=self.dataset_id,
            config=self.configuration,
            folds=self.folds,
            trainer=self.trainer,
            plans=self.plans,
        )

        print("[predict] Running: {}".format(cmd))
        subprocess.run(cmd, shell=True, check=True)
        print("[predict] Done")

    def write_outputs(self, uuid):
        """Convert prediction NIfTI to MHA and write to output path."""
        os.makedirs(self.output_path, exist_ok=True)

        pred_nii = os.path.join(self.result_dir, "CASE_001.nii.gz")
        out_mha = os.path.join(self.output_path, uuid + ".mha")

        self.convert_nii_to_mha(pred_nii, out_mha)
        print("[write_outputs] Output written: {}".format(out_mha))

    def process(self):
        """Main entry point: load -> predict -> write."""
        print("[process] Starting inference")
        uuid = self.load_inputs()
        self.predict()
        self.write_outputs(uuid)
        print("[process] Complete")


if __name__ == "__main__":
    AutoPETVAlgorithm().process()