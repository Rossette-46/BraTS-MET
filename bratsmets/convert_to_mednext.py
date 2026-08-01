"""
Convert BraTS-MET per-subject folders into nnU-Net v1's raw-data format for
MedNeXt. Parallel and isolated from the v2 conversion -- different env vars,
directory layout, and dataset.json schema; same subject selection.

Target: nnUNet_raw_data/{TASK_NAME}/imagesTr/{subject}_0000..0003.nii.gz,
labelsTr/{subject}.nii.gz, dataset.json (v1 schema).

Usage: python convert_to_mednext.py
"""

import os
import json
import shutil

BASE_DIR      = "pathtotraining/dataset"
UCSD_DIR      = os.path.join(BASE_DIR, "UCSD - Training")
CORRECTED_DIR = "pathtocorrected/dataset"

NNUNET_V1_RAW_DATA_BASE = "pathto/nnunetrawfolder"
TASK_ID   = 501
TASK_NAME = f"Task{TASK_ID:03d}_BraTSMET"

LINK_MODE = "symlink"   # or "copy"

CHANNEL_ORDER = ["t1n", "t1c", "t2w", "t2f"]

EXCLUDE_HARD       = {"BraTS-MET-01094-002", "BraTS-MET-01094-003"}
CORRECTED_SUBJECTS = {"BraTS-MET-01184-002"}

LABELS_V1   = {"0": "background", "1": "NETC", "2": "SNFH", "3": "ET", "4": "RC"}
MODALITY_V1 = {str(i): m for i, m in enumerate(CHANNEL_ORDER)}


def _place(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    if LINK_MODE == "symlink":
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError as e:
            print(f"  [warn] symlink failed ({e}); copying instead: {os.path.basename(dst)}")
    shutil.copy2(src, dst)


def _walk_subjects(base):
    out = []
    if not os.path.exists(base):
        print(f"[WARN] not found: {base}")
        return out
    for name in sorted(os.listdir(base)):
        if name.startswith("BraTS-MET") and os.path.isdir(os.path.join(base, name)):
            out.append((name, os.path.join(base, name)))
    return out


def main():
    out_root   = os.path.join(NNUNET_V1_RAW_DATA_BASE, "nnUNet_raw_data", TASK_NAME)
    images_dir = os.path.join(out_root, "imagesTr")
    labels_dir = os.path.join(out_root, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    all_dirs = _walk_subjects(BASE_DIR) + _walk_subjects(UCSD_DIR)
    print(f"Raw subject folders found : {len(all_dirs)}")

    converted, skipped_excluded, skipped_missing, recovered = 0, 0, 0, 0
    training_list = []

    for subject, path in all_dirs:
        if subject in EXCLUDE_HARD:
            print(f"[SKIP] {subject}  <- corrupted label, no correction")
            skipped_excluded += 1
            continue

        modality_paths = {m: os.path.join(path, f"{subject}-{m}.nii.gz") for m in CHANNEL_ORDER}

        if subject in CORRECTED_SUBJECTS:
            seg = os.path.join(CORRECTED_DIR, f"{subject}-seg.nii.gz")
            if os.path.exists(seg):
                recovered += 1
                print(f"[INFO] {subject}: using corrected seg")
            else:
                print(f"[SKIP] {subject}  <- corrected seg missing at {seg}")
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

        training_list.append({
            "image": f"./imagesTr/{subject}.nii.gz",
            "label": f"./labelsTr/{subject}.nii.gz",
        })
        converted += 1

    dataset_json = {
        "name": "BraTSMET",
        "description": "BraTS-MET 2025, 4 modalities, labels BG/NETC/SNFH/ET/RC",
        "tensorImageSize": "4D",
        "reference": "",
        "licence": "",
        "release": "1.0",
        "modality": MODALITY_V1,
        "labels": LABELS_V1,
        "numTraining": converted,
        "numTest": 0,
        "training": training_list,
        "test": [],
    }
    with open(os.path.join(out_root, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("\n" + "=" * 60)
    print(f"  nnU-Net v1 (MedNeXt) dataset written: {out_root}")
    print(f"  Converted subjects       : {converted}")
    print(f"    (recovered via corrected label: {recovered})")
    print(f"  Skipped (excluded)       : {skipped_excluded}")
    print(f"  Skipped (missing files)  : {skipped_missing}")
    print(f"  Task                     : {TASK_NAME}")
    print("=" * 60)
    print("\nNext: set v1 env vars + run MedNeXt plan_and_preprocess (see the")
    print("workflow -- VERIFY the exact CLI command name against the repo README).")


if __name__ == "__main__":
    main()
