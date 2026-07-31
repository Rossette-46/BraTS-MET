"""
Convert a BraTS-MET inference set (per-subject folders, no labels) into the
flat _0000..0003 layout both nnUNetv2_predict and mednextv1_predict expect.
Channel order [t1n, t1c, t2w, t2f] must match training; do not reorder.

Cases missing a sequence get it substituted from the closest available
contrast by default (T1n<->T1c, T2w<->T2-FLAIR); pass --no-substitute-missing
to skip such cases instead.

Usage: python convert_validation.py --in-dir <per-subject dirs> --out-dir <flat>
"""

import os
import sys
import shutil
import argparse

CHANNEL_ORDER = ["t1n", "t1c", "t2w", "t2f"]

SUBSTITUTES = {
    "t1n": ["t1c", "t2f", "t2w"],
    "t1c": ["t1n", "t2f", "t2w"],
    "t2w": ["t2f", "t1n", "t1c"],
    "t2f": ["t2w", "t1n", "t1c"],
}


def scan(in_dir):
    """Return [(name, path, {modality: source_path_or_None})]."""
    out = []
    for entry in sorted(os.listdir(in_dir)):
        path = os.path.join(in_dir, entry)
        if not os.path.isdir(path):
            continue
        present = {}
        for m in CHANNEL_ORDER:
            f = os.path.join(path, f"{entry}-{m}.nii.gz")
            present[m] = f if os.path.exists(f) else None
        if any(present.values()):
            out.append((entry, path, present))
    return out


def place(src, dst, copy):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    if not copy:
        try:
            os.symlink(os.path.abspath(src), dst)
            return
        except OSError as e:
            print(f"  [warn] symlink failed ({e}); copying instead: "
                  f"{os.path.basename(dst)}")
    shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True,
                   help="directory of per-subject folders (read-only safe)")
    p.add_argument("--out-dir", required=True,
                   help="destination for the flat _0000.._0003 layout")
    p.add_argument("--copy", action="store_true",
                   help="copy rather than symlink")
    p.add_argument("--no-substitute-missing", dest="substitute",
                   action="store_false", default=True,
                   help="skip cases missing a sequence instead of substituting")
    p.add_argument("--manifest", default=None,
                   help="write the list of prepared case identifiers here, one "
                        "per line, for downstream completeness checking")
    args = p.parse_args()

    if not os.path.isdir(args.in_dir):
        sys.exit(f"ERROR: not a directory: {args.in_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    subjects = scan(args.in_dir)
    if not subjects:
        subdirs = [d for d in os.listdir(args.in_dir)
                   if os.path.isdir(os.path.join(args.in_dir, d))]
        sys.exit(f"ERROR: no subject folders with any expected sequence under "
                 f"{args.in_dir}\n"
                 f"  subdirectories present: {len(subdirs)}\n"
                 f"  expected <name>-t1n/-t1c/-t2w/-t2f.nii.gz inside each")

    print(f"subject folders found: {len(subjects)}")

    prepared, skipped, n_substituted = [], [], 0
    for name, path, present in subjects:
        missing = [m for m in CHANNEL_ORDER if present[m] is None]

        if missing and not args.substitute:
            print(f"[SKIP] {name}  missing {missing}")
            skipped.append(name)
            continue

        resolved = dict(present)
        for m in missing:
            pick = next((present[alt] for alt in SUBSTITUTES[m] if present[alt]), None)
            if pick is None:
                break
            resolved[m] = pick
            print(f"[SUBSTITUTE] {name}: {m} absent, using "
                  f"{os.path.basename(pick).rsplit('-', 1)[-1].replace('.nii.gz','')} "
                  f"-- prediction for this case is compromised")
            n_substituted += 1

        if any(v is None for v in resolved.values()):
            print(f"[SKIP] {name}  no usable sequence to substitute from")
            skipped.append(name)
            continue

        for idx, m in enumerate(CHANNEL_ORDER):
            place(resolved[m],
                  os.path.join(args.out_dir, f"{name}_{idx:04d}.nii.gz"),
                  args.copy)
        prepared.append(name)

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            f.write("\n".join(prepared) + "\n")

    print()
    print(f"prepared            : {len(prepared)}")
    print(f"channels substituted: {n_substituted}")
    print(f"skipped             : {len(skipped)}"
          + (f" -> {skipped}" if skipped else ""))
    print(f"flat input          : {args.out_dir}")


if __name__ == "__main__":
    main()
