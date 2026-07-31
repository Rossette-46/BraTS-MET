"""
Guarantee exactly one <case>.nii.gz per input case folder in a flat /output.
A case with no prediction gets an all-background mask matching its own
geometry, so the submission stays structurally valid instead of missing a
file. Also strips any non-prediction files (bookkeeping JSON, subfolders)
that would invalidate a flat-output submission.

Usage: python ensure_outputs.py --in-dir /input --out-dir /output
"""

import os
import sys
import shutil
import argparse

import numpy as np
import SimpleITK as sitk

SEQUENCES = ["t1c", "t1n", "t2f", "t2w"]


def reference_image(case_dir, case):
    for m in SEQUENCES:
        path = os.path.join(case_dir, f"{case}-{m}.nii.gz")
        if os.path.exists(path):
            try:
                return sitk.ReadImage(path)
            except Exception:
                continue
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--keep-extras", action="store_true",
                   help="do not remove non-prediction files from the output")
    args = p.parse_args()

    cases = sorted(d for d in os.listdir(args.in_dir)
                   if os.path.isdir(os.path.join(args.in_dir, d)))
    if not cases:
        sys.exit(f"ERROR: no case folders in {args.in_dir}")

    expected = {f"{c}.nii.gz" for c in cases}

    removed = []
    if not args.keep_extras:
        for entry in sorted(os.listdir(args.out_dir)):
            path = os.path.join(args.out_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
                removed.append(entry + "/")
            elif entry not in expected:
                os.remove(path)
                removed.append(entry)

    filled, unfixable = [], []
    for case in cases:
        out_path = os.path.join(args.out_dir, f"{case}.nii.gz")
        if os.path.exists(out_path):
            continue

        ref = reference_image(os.path.join(args.in_dir, case), case)
        if ref is None:
            unfixable.append(case)
            print(f"[UNFIXABLE] {case}: no readable sequence to copy geometry from")
            continue

        empty = sitk.GetImageFromArray(
            np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8))
        empty.CopyInformation(ref)
        sitk.WriteImage(empty, out_path)
        filled.append(case)
        print(f"[FILLED] {case}: no prediction was produced -- wrote an "
              f"all-background mask so the case is not missing")

    n_out = len([f for f in os.listdir(args.out_dir) if f.endswith(".nii.gz")])
    print()
    print(f"input cases        : {len(cases)}")
    print(f"predictions present: {n_out}")
    print(f"gaps filled        : {len(filled)}")
    print(f"extras removed     : {len(removed)}"
          + (f" -> {removed}" if removed else ""))

    if unfixable:
        print(f"\nERROR: {len(unfixable)} cases have no prediction and no readable "
              f"reference: {unfixable}")
        sys.exit(1)
    if n_out != len(cases):
        print(f"\nERROR: {n_out} predictions for {len(cases)} cases")
        sys.exit(1)
    print("\noutput is flat and complete")


if __name__ == "__main__":
    main()
