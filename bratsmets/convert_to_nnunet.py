"""
Convert BraTS-MET per-subject folders into nnU-Net v2's Dataset001_BraTSMET.

Source: {subject}-t1n/t1c/t2w/t2f/seg.nii.gz per folder.
Target: imagesTr/{subject}_0000..0003.nii.gz, labelsTr/{subject}.nii.gz, dataset.json.

Usage: python convert_to_nnunet.py
"""

import os
import json
import shutil

BASE_DIR      = "/flash/cbme/phd/bmz258430/BraTS/MICCAI-LH-BraTS2025-MET-Challenge-TrainingData_batch1/MICCAI-LH-BraTS2025-MET-Challenge-Training/"
UCSD_DIR      = os.path.join(BASE_DIR, "UCSD - Training")
CORRECTED_DIR = "/flash/cbme/phd/bmz258430/BraTS/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels_batch1/MICCAI-LH-BraTS2025-MET-Challenge-corrected-labels"

NNUNET_RAW    = "/flash/cbme/phd/bmz258430/BraTS/nnUNet_raw"
DATASET_NAME  = "Dataset001_BraTSMET"

LINK_MODE = "symlink"  # or "copy"

CHANNEL_ORDER = ["t1n", "t1c", "t2w", "t2f"]  # must match convert_validation.py

EXCLUDE_HARD = {"BraTS-MET-01094-002", "BraTS-MET-01094-003"}  # no corrected label
CORRECTED_SUBJECTS = {"BraTS-MET-01184-002"}  # use CORRECTED_DIR instead

LABELS = {
    "background": 0,
    "NETC": 1,
    "SNFH": 2,
    "ET": 3,
    "RC": 4,
}


def _place(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    if LINK_MODE == "symlink":
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError as e:
            print(f"  [warn] symlink failed ({e}); falling back to copy for {os.path.basename(dst)}")
    shutil.copy2(src, dst)


def _walk_subjects(base):
    out = []
    if not os.path.exists(base):
        print(f"[WARN] not found: {base}")
        return out
    for name in sorted(os.listdir(base)):
        if not name.startswith("BraTS-MET"):
            continue
        path = os.path.join(base, name)
        if os.path.isdir(path):
            out.append((name, path))
    return out


def main():
    out_root   = os.path.join(NNUNET_RAW, DATASET_NAME)
    images_dir = os.path.join(out_root, "imagesTr")
    labels_dir = os.path.join(out_root, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    all_dirs = _walk_subjects(BASE_DIR) + _walk_subjects(UCSD_DIR)
    print(f"Raw subject folders found : {len(all_dirs)}")

    converted, skipped_excluded, skipped_missing, recovered = 0, 0, 0, 0

    for subject, path in all_dirs:
        if subject in EXCLUDE_HARD:
            print(f"[SKIP] {subject}  <- corrupted label, no correction available")
            skipped_excluded += 1
            continue

        modality_paths = {m: os.path.join(path, f"{subject}-{m}.nii.gz") for m in CHANNEL_ORDER}

        if subject in CORRECTED_SUBJECTS:
            seg = os.path.join(CORRECTED_DIR, f"{subject}-seg.nii.gz")
            if os.path.exists(seg):
                recovered += 1
                print(f"[INFO] {subject}: using corrected seg")
            else:
                print(f"[SKIP] {subject}  <- corrected seg expected but not found at {seg}")
                skipped_excluded += 1
                continue
        else:
            seg = os.path.join(path, f"{subject}-seg.nii.gz")

        needed = list(modality_paths.values()) + [seg]
        if not all(os.path.exists(f) for f in needed):
            missing = [os.path.basename(f) for f in needed if not os.path.exists(f)]
            print(f"[SKIP] {subject}  <- missing files: {missing}")
            skipped_missing += 1
            continue

        for idx, m in enumerate(CHANNEL_ORDER):
            _place(modality_paths[m], os.path.join(images_dir, f"{subject}_{idx:04d}.nii.gz"))
        _place(seg, os.path.join(labels_dir, f"{subject}.nii.gz"))
        converted += 1

    dataset_json = {
        "channel_names": {str(i): m for i, m in enumerate(CHANNEL_ORDER)},
        "labels": LABELS,
        "numTraining": converted,
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(out_root, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("\n" + "=" * 60)
    print(f"  nnU-Net dataset written: {out_root}")
    print(f"  Converted subjects    : {converted}")
    print(f"    (of which recovered via corrected label: {recovered})")
    print(f"  Skipped (excluded)    : {skipped_excluded}")
    print(f"  Skipped (missing files): {skipped_missing}")
    print(f"  Link mode             : {LINK_MODE}")
    print(f"  dataset.json numTraining = {converted}")
    print("=" * 60)
    print("\nNext: run  nnUNetv2_plan_and_preprocess  (see nnunet_commands.md).")


if __name__ == "__main__":
    main()
