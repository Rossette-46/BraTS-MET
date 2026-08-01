"""
Region-specific hard-override ensemble: every voxel from MedNeXt except RC,
which is taken entirely from nnU-Net. Not probability averaging -- MedNeXt's
own RC calls are discarded, nnU-Net's are stamped on top. Then optional hole
filling and lesion-size filtering via postprocess.py.

Usage: python ensemble.py --mednext-dir <dir> --nnunet-dir <dir> --out-dir <dir>
       [--wt-mm3 N] [--rc-mm3 N] [--hole-mm3 N] [--no-postproc]
"""

import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk

MEDNEXT_DIR = "pathto/mednextpredictdata"
NNUNET_DIR  = "pathto/nnunetpredictdata"
OUT_DIR     = "output/path"

RC_LABEL = 4

APPLY_POSTHOC_FILTER = True
WT_MM3 = 5.0
RC_MM3 = 26.0
HOLE_MM3 = 15.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Region-specific hard-override ensemble: RC from nnU-Net, "
                    "all other labels from MedNeXt, followed by optional "
                    "hole filling and lesion-size filtering.")
    p.add_argument("--mednext-dir", default=MEDNEXT_DIR,
                   help="MedNeXt predictions (source of NETC/SNFH/ET)")
    p.add_argument("--nnunet-dir", default=NNUNET_DIR,
                   help="nnU-Net predictions (source of RC only)")
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="output directory for merged predictions")
    p.add_argument("--wt-mm3", type=float, default=WT_MM3,
                   help="lesion-size threshold for the tumour union {1,2,3}")
    p.add_argument("--rc-mm3", type=float, default=RC_MM3,
                   help="lesion-size threshold for RC (label 4)")
    p.add_argument("--hole-mm3", type=float, default=HOLE_MM3,
                   help="enclosed-background hole-filling threshold")
    p.add_argument("--no-postproc", action="store_true",
                   help="write the raw merge with no hole filling or filtering")
    return p.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocess import fill_small_holes_mm3, remove_small_lesions_mm3


def merge_case(mednext_path, nnunet_path):
    """Returns (merged_sitk_image, status_string)."""
    mednext_img = sitk.ReadImage(mednext_path)
    nnunet_img  = sitk.ReadImage(nnunet_path)

    mednext_arr = sitk.GetArrayFromImage(mednext_img)
    nnunet_arr  = sitk.GetArrayFromImage(nnunet_img)

    if mednext_arr.shape != nnunet_arr.shape:
        return mednext_img, f"GEOMETRY MISMATCH {mednext_arr.shape} vs {nnunet_arr.shape} -> used MedNeXt only, RC NOT overridden"

    result = mednext_arr.copy()
    result[result == RC_LABEL] = 0

    rc_from_nnunet = nnunet_arr == RC_LABEL
    n_rc_voxels = int(rc_from_nnunet.sum())
    result[rc_from_nnunet] = RC_LABEL

    out_img = sitk.GetImageFromArray(result.astype(np.uint8))
    out_img.CopyInformation(mednext_img)

    status = f"OK (RC voxels from nnU-Net: {n_rc_voxels})" if n_rc_voxels else "OK (nnU-Net predicted NO RC for this case)"
    return out_img, status


def main():
    args = parse_args()
    apply_filter = not args.no_postproc

    os.makedirs(args.out_dir, exist_ok=True)

    mednext_files = sorted(f for f in os.listdir(args.mednext_dir) if f.endswith(".nii.gz"))
    print(f"MedNeXt : {args.mednext_dir}  ({len(mednext_files)} predictions)")
    print(f"nnU-Net : {args.nnunet_dir}")
    print(f"output  : {args.out_dir}")
    if apply_filter:
        print(f"filter  : WT={args.wt_mm3}mm^3  RC={args.rc_mm3}mm^3  hole={args.hole_mm3}mm^3")
    else:
        print("filter  : DISABLED (raw merge)")
    print()

    n_ok, n_missing, n_mismatch = 0, 0, 0

    for fname in mednext_files:
        subject = fname[:-len(".nii.gz")]
        mednext_path = os.path.join(args.mednext_dir, fname)
        nnunet_path  = os.path.join(args.nnunet_dir, fname)

        if not os.path.exists(nnunet_path):
            print(f"[SKIP] {subject}  <- no matching nnU-Net prediction, cannot override RC")
            n_missing += 1
            continue

        merged_img, status = merge_case(mednext_path, nnunet_path)
        if "MISMATCH" in status:
            n_mismatch += 1
        else:
            n_ok += 1

        if apply_filter:
            merged_img = fill_small_holes_mm3(merged_img, args.hole_mm3)
            merged_img = remove_small_lesions_mm3(merged_img, args.wt_mm3, args.rc_mm3)

        sitk.WriteImage(merged_img, os.path.join(args.out_dir, fname))
        print(f"[{status}] {subject}")

    print("\n" + "=" * 60)
    print(f"  Combined predictions written: {args.out_dir}")
    print(f"  OK (RC overridden normally)                        : {n_ok}")
    print(f"  Geometry mismatch (RC NOT overridden, MedNeXt-only): {n_mismatch}")
    print(f"  Missing nnU-Net prediction (skipped entirely)      : {n_missing}")
    if apply_filter:
        print(f"  Post-hoc filter applied                            : "
              f"WT={args.wt_mm3}mm^3, RC={args.rc_mm3}mm^3, hole={args.hole_mm3}mm^3")
    else:
        print("  Post-hoc filter applied                            : no (raw merge)")
    print("=" * 60)


if __name__ == "__main__":
    main()
