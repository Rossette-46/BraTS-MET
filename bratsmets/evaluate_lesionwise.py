"""
Lesion-wise evaluation of held-out fold-0 predictions against reference
annotations, via panoptica through Astarakee/brats_evaluator. Used because
the portal's validation labels aren't released, so this evaluates on the
259 training-set cases held out by fold 0 instead, using the same lesion-wise
definition. No inference needed -- both frameworks write their own fold-0
validation predictions at training time; use MedNeXt's validation_raw, not
validation_raw_postprocessed.

analyze_exam()'s return is flattened generically, not assumed, so a
structure change degrades gracefully. Run with --limit 1 first and check the
printed columns; use --dice-key if auto-detection picks the wrong one.

Setup (separate env):
    conda create -n bratseval python=3.10 -y && conda activate bratseval
    pip install panoptica pandas
    git clone https://github.com/Astarakee/brats_evaluator.git

Usage:
    python evaluate_lesionwise.py --pred-dir <preds> --gt-dir <labelsTr>
        --evaluator <brats_evaluator repo> --out-csv out.csv [--limit 1]

Resumable: cases already in --out-csv are skipped on re-run.
"""

import os
import sys
import csv
import json
import argparse
import traceback
from numbers import Number

REGIONS = ["wt", "tc", "et", "rc", "netc"]
DICE_HINTS = ["sq_dsc", "lesion_wise_dice", "lesionwise_dice", "sq_dice", "dsc", "dice"]


def flatten(obj, prefix=""):
    """Flatten an arbitrary nested result into {column_name: scalar},
    tolerant of dicts, lists, pandas objects, and attribute-carrying objects."""
    out = {}
    key = prefix.rstrip(".")

    if obj is None or isinstance(obj, (str, bool)):
        if key:
            out[key] = obj
        return out

    if isinstance(obj, Number):
        if key:
            out[key] = obj
        return out

    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return flatten(to_dict(), prefix)
        except Exception:
            pass

    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
        return out

    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
        return out

    d = getattr(obj, "__dict__", None)
    if d:
        for k, v in d.items():
            if not k.startswith("_"):
                out.update(flatten(v, f"{prefix}{k}."))
        return out

    if key:
        out[key] = str(obj)
    return out


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def add_derived(flat):
    """Add the challenge's lesion-wise Dice, reconstructed from panoptica's
    parts. panoptica's sq_dsc scores matched instances only and ignores
    false positives/negatives; the challenge scores every unmatched lesion
    zero. Since sum(TP dice) == sq_dsc * tp, the challenge metric is exactly
    sq_dsc * tp / (tp + fp + fn). Do not report sq_dsc as lesion-wise Dice."""
    for region in REGIONS:
        tp = _num(flat.get(f"{region}.tp"))
        fp = _num(flat.get(f"{region}.fp"))
        fn = _num(flat.get(f"{region}.fn"))
        if tp is None or fp is None or fn is None:
            continue
        denom = tp + fp + fn
        if denom <= 0:
            flat[f"{region}.lesionwise_dsc"] = float("nan")
            continue
        sq = _num(flat.get(f"{region}.sq_dsc"))
        if sq is None or sq != sq:
            sq = 0.0
        flat[f"{region}.lesionwise_dsc"] = sq * tp / denom
    return flat


def region_present(row, region):
    """True when the reference actually contains this region."""
    n = _num(row.get(f"{region}.num_ref_instances"))
    return n is not None and n > 0


def pick_dice_columns(columns):
    """Map each region to its most likely lesion-wise Dice column."""
    chosen = {}
    for region in REGIONS:
        candidates = [c for c in columns if region in c.lower()]
        if not candidates:
            continue
        for hint in DICE_HINTS:
            hits = [c for c in candidates if hint in c.lower()]
            if hits:
                chosen[region] = sorted(hits, key=len)[0]
                break
    return chosen


def load_case_ids(args):
    """Case list from the prediction folder, cross-checked against the
    split file if given."""
    preds = sorted(f[:-len(".nii.gz")] for f in os.listdir(args.pred_dir)
                   if f.endswith(".nii.gz"))
    if not preds:
        sys.exit(f"ERROR: no .nii.gz files in {args.pred_dir}")

    if args.splits_json:
        with open(args.splits_json, encoding="utf-8") as f:
            folds = json.load(f)
        expected = set(folds[args.fold]["val"])
        got = set(preds)
        if got != expected:
            print(f"  WARNING: prediction folder does not match fold {args.fold} "
                  f"of {os.path.basename(args.splits_json)}")
            print(f"    in predictions only : {len(got - expected)}")
            print(f"    in split only       : {len(expected - got)}")
            if not args.allow_split_mismatch:
                sys.exit("Refusing to continue. Pass --allow-split-mismatch to override.")
        else:
            print(f"  split check: OK -- all {len(preds)} cases match fold "
                  f"{args.fold} validation set")
    return preds


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, help="directory of prediction .nii.gz")
    p.add_argument("--gt-dir", required=True, help="directory of reference .nii.gz")
    p.add_argument("--evaluator", required=True,
                   help="path to the cloned brats_evaluator repository")
    p.add_argument("--config", default=None,
                   help="panoptica config (default: <evaluator>/brats-configs/config_mets.yaml)")
    p.add_argument("--out-csv", required=True, help="per-case results CSV")
    p.add_argument("--splits-json", default=None,
                   help="split file to cross-check the case list against")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--allow-split-mismatch", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N cases (use 1 to inspect output structure)")
    p.add_argument("--dice-key", default=None,
                   help="exact column holding lesion-wise Dice, e.g. 'wt.sq_dsc'. "
                        "Use {region} as a placeholder, e.g. '{region}.sq_dsc'. "
                        "Overrides auto-detection.")
    args = p.parse_args()

    config = args.config or os.path.join(args.evaluator, "brats-configs", "config_mets.yaml")
    for path, what in ((args.pred_dir, "prediction dir"), (args.gt_dir, "ground-truth dir"),
                       (args.evaluator, "evaluator repo"), (config, "panoptica config")):
        if not os.path.exists(path):
            sys.exit(f"ERROR: {what} not found: {path}")

    sys.path.insert(0, args.evaluator)
    try:
        from analyze_brats import analyze_exam
    except ImportError as e:
        missing = getattr(e, "name", None)
        sys.exit(f"ERROR: cannot import analyze_brats from {args.evaluator}\n"
                 f"  {e}\n\n"
                 f"  brats_evaluator needs more than panoptica alone"
                 + (f" (missing: {missing})" if missing else "") + ".\n"
                 f"  Install its full dependency set:\n"
                 f"      pip install -r {os.path.join(args.evaluator, 'requirements.txt')}")

    print(f"predictions : {args.pred_dir}")
    print(f"ground truth: {args.gt_dir}")
    print(f"config      : {config}")
    cases = load_case_ids(args)
    if args.limit:
        cases = cases[:args.limit]
    print(f"cases to evaluate: {len(cases)}\n")

    done, rows = set(), []
    if os.path.exists(args.out_csv):
        with open(args.out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done.add(row.get("case"))
        if done:
            print(f"resuming: {len(done)} cases already in {args.out_csv}\n")

    fieldnames, n_fail = None, 0
    if rows:
        fieldnames = list(rows[0].keys())

    for i, case in enumerate(cases, 1):
        if case in done:
            continue
        gt = os.path.join(args.gt_dir, case + ".nii.gz")
        pred = os.path.join(args.pred_dir, case + ".nii.gz")
        if not os.path.exists(gt):
            print(f"[{i}/{len(cases)}] {case}  SKIP: no reference annotation")
            continue

        try:
            result = analyze_exam(prediction_path=pred, label_path=gt,
                                  identifier=case, panoptica_config_path=config)
            flat = add_derived(flatten(result))
            flat["case"] = case
        except Exception:
            n_fail += 1
            print(f"[{i}/{len(cases)}] {case}  FAILED")
            traceback.print_exc()
            continue

        if fieldnames is None:
            fieldnames = ["case"] + sorted(k for k in flat if k != "case")
            print("\n" + "=" * 70)
            print("DISCOVERED COLUMNS (verify the lesion-wise Dice column is present):")
            for k in fieldnames:
                if k != "case":
                    print(f"    {k} = {flat.get(k)}")
            print("=" * 70 + "\n")

        rows.append(flat)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"[{i}/{len(cases)}] {case}  ok")

    if not rows:
        sys.exit("\nNo results produced.")

    cols = [c for c in (fieldnames or []) if c != "case"]

    print("\n" + "=" * 70)
    print(f"  {len(rows)} cases evaluated, {n_fail} failed   ->  {args.out_csv}")
    print("=" * 70)

    def mean_of(col, only_present=None):
        vals = []
        for r in rows:
            if only_present and not region_present(r, only_present):
                continue
            v = _num(r.get(col))
            if v is not None and v == v and v not in (float("inf"), float("-inf")):
                vals.append(v)
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    metrics = [
        ("lesionwise_dsc", "lesion-wise DSC  (challenge definition: FP/FN scored 0)"),
        ("sq_dsc",         "SQ  segmentation quality (matched lesions only)"),
        ("rq",             "RQ  detection F1"),
        ("pq_dsc",         "PQ  = SQ x RQ"),
    ]
    if args.dice_key:
        metrics.insert(0, (args.dice_key.replace("{region}.", ""), "requested metric"))

    for suffix, description in metrics:
        available = {r: f"{r}.{suffix}" for r in REGIONS if f"{r}.{suffix}" in cols}
        if not available:
            continue
        print(f"\n  {description}")
        means = {}
        for region in REGIONS:
            col = available.get(region)
            if not col:
                continue
            m, n = mean_of(col)
            if m is None:
                continue
            means[region] = m
            line = f"    {region.upper():5s}  {m:.4f}   (n={n}"
            mp, npres = mean_of(col, only_present=region)
            if npres and npres != n:
                line += f"; {mp:.4f} over the {npres} cases with {region.upper()} present"
            print(line + ")")
        scored = [means[r] for r in ("wt", "tc", "et", "rc") if r in means]
        if len(scored) == 4:
            print(f"    MEAN   {sum(scored) / 4:.4f}   (WT/TC/ET/RC)")

    print("\n  NOTE: 'lesion-wise DSC' is the row comparable to the validation")
    print("  portal. SQ ignores false positives and will read substantially")
    print("  higher -- do not report it as lesion-wise Dice.")
    print()


if __name__ == "__main__":
    main()
