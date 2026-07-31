"""
Subject-wise (voxel-level) Dice per evaluation region (WT/TC/ET/RC) over any
number of prediction folders. Not the challenge's lesion-wise Dice -- this is
volume-based, insensitive to lesion count, and reads higher on multi-focal
disease. Label it as subject-wise in any write-up.

A region absent from the reference scores 1.0 if nothing is predicted
("vacuous") or 0.0 if something is ("fp_only"); means are reported both over
all cases and over only cases where the reference contains the region --
report the latter for RC, since ~7 in 8 cases have none.

Usage:
    python evaluate_dice.py --gt-dir <labelsTr> --pred nnunet=<dir> --pred mednext=<dir> --out-csv out.csv
"""

import os
import csv
import sys
import json
import argparse

import numpy as np
import SimpleITK as sitk

REGIONS = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
    "RC": (4,),
}


def dice(pred_mask, gt_mask):
    """Returns (dice, status); status is 'scored', 'fp_only', or 'vacuous'."""
    p, g = pred_mask.sum(), gt_mask.sum()
    if g == 0:
        if p == 0:
            return 1.0, "vacuous"
        return 0.0, "fp_only"
    inter = np.logical_and(pred_mask, gt_mask).sum()
    return (2.0 * inter) / (p + g), "scored"


def evaluate_dir(name, pred_dir, gt_dir, verbose=True):
    cases = sorted(f[:-len(".nii.gz")] for f in os.listdir(pred_dir)
                   if f.endswith(".nii.gz"))
    if not cases:
        sys.exit(f"ERROR: no predictions in {pred_dir}")

    rows, missing = [], 0
    for i, case in enumerate(cases, 1):
        gt_path = os.path.join(gt_dir, case + ".nii.gz")
        if not os.path.exists(gt_path):
            missing += 1
            continue

        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        pr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_dir, case + ".nii.gz")))

        if gt.shape != pr.shape:
            print(f"  {case}: SHAPE MISMATCH {pr.shape} vs {gt.shape} -- skipped")
            continue

        row = {"config": name, "case": case}
        for region, labels in REGIONS.items():
            gm = np.isin(gt, labels)
            pm = np.isin(pr, labels)
            d, status = dice(pm, gm)
            row[f"{region}_dice"] = d
            row[f"{region}_status"] = status
            row[f"{region}_gt_voxels"] = int(gm.sum())
            row[f"{region}_pred_voxels"] = int(pm.sum())
        rows.append(row)

        if verbose and (i % 25 == 0 or i == len(cases)):
            print(f"  [{name}] {i}/{len(cases)}")

    if missing:
        print(f"  [{name}] {missing} cases had no reference annotation and were skipped")
    return rows


def summarise(rows, name):
    """Returns {region: (mean_all, n_all, mean_present, n_present)}."""
    out = {}
    for region in REGIONS:
        alls, present = [], []
        for r in rows:
            if r["config"] != name:
                continue
            d = r[f"{region}_dice"]
            alls.append(d)
            if r[f"{region}_status"] == "scored":
                present.append(d)
        out[region] = (
            sum(alls) / len(alls) if alls else None, len(alls),
            sum(present) / len(present) if present else None, len(present),
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--pred", action="append", required=True, metavar="NAME=PATH",
                   help="prediction folder to evaluate; repeatable")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--splits-json", default=None,
                   help="verify each prediction folder against a fold's val list")
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()

    configs = []
    for spec in args.pred:
        if "=" not in spec:
            sys.exit(f"ERROR: --pred needs NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        if not os.path.isdir(path):
            sys.exit(f"ERROR: not a directory: {path}")
        configs.append((name, path))

    expected = None
    if args.splits_json:
        with open(args.splits_json, encoding="utf-8") as f:
            expected = set(json.load(f)[args.fold]["val"])
        print(f"split check against fold {args.fold}: {len(expected)} cases\n")

    all_rows = []
    for name, path in configs:
        print(f"evaluating {name}: {path}")
        if expected is not None:
            got = set(f[:-len(".nii.gz")] for f in os.listdir(path)
                      if f.endswith(".nii.gz"))
            if got != expected:
                print(f"  WARNING: case list differs from fold {args.fold} "
                      f"(+{len(got - expected)} / -{len(expected - got)})")
            else:
                print(f"  split check OK")
        all_rows.extend(evaluate_dir(name, path, args.gt_dir))
        print()

    if not all_rows:
        sys.exit("no results")

    fieldnames = ["config", "case"]
    for region in REGIONS:
        fieldnames += [f"{region}_dice", f"{region}_status",
                       f"{region}_gt_voxels", f"{region}_pred_voxels"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("=" * 78)
    print(" SUBJECT-WISE DICE  (not the challenge's lesion-wise metric)")
    print("=" * 78)
    print(f"\n{'Configuration':<22}" + "".join(f"{r:>9}" for r in REGIONS) + f"{'Mean':>9}")
    print("-" * 78)
    for name, _ in configs:
        s = summarise(all_rows, name)
        vals = [s[r][0] for r in REGIONS]
        line = f"{name:<22}" + "".join(f"{v:>9.4f}" if v is not None else f"{'--':>9}"
                                       for v in vals)
        got = [v for v in vals if v is not None]
        line += f"{sum(got) / len(got):>9.4f}" if len(got) == len(REGIONS) else f"{'--':>9}"
        print(line)

    print(f"\n{'':22}over cases whose reference CONTAINS the region:")
    print("-" * 78)
    for name, _ in configs:
        s = summarise(all_rows, name)
        line = f"{name:<22}"
        for r in REGIONS:
            _, _, mp, npres = s[r]
            line += f"{mp:>9.4f}" if mp is not None else f"{'--':>9}"
        print(line)
    print(f"\n{'':22}(n per region, first config): " +
          ", ".join(f"{r}={summarise(all_rows, configs[0][0])[r][3]}" for r in REGIONS))

    print(f"\nPer-case results: {args.out_csv}")
    print("\nRC over ALL cases is inflated by pre-operative subjects scoring 1.0")
    print("for correctly predicting no cavity. Report the CONTAINS-region row.")


if __name__ == "__main__":
    main()
