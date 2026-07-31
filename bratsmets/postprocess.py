"""
Post-hoc hole filling and spacing-aware lesion-size filtering on saved
native-space predictions. Two things matter: thresholds are mm^3, converted
to voxel counts per file's own header spacing (native spacing varies up to
4x across subjects); and lesion filtering runs on the UNION of tumor labels
{1,2,3}, not per label, so a NETC core enclosed by an ET rim isn't mistaken
for an isolated speck and deleted. RC (4) is filtered independently. Hole
filling runs first so the size filter sees each lesion's true extent.

Usage: python postprocess.py <input_dir> <output_dir> <wt_mm3> [<rc_mm3>] [<hole_mm3>]
    rc_mm3 defaults to wt_mm3; hole_mm3 defaults to 15.0; pass hole_mm3=0 to disable.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as _ndi


def fill_small_holes_mm3(pred_img, max_hole_mm3, report_filled=False):
    """Fill small, fully-enclosed background pockets with the majority label
    of their surrounding tissue. A pocket qualifies if it is below
    max_hole_mm3 and does not touch the array's outer boundary -- a
    genuinely enclosed pocket cannot reach the exterior background."""
    spacing = pred_img.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    pred_np = sitk.GetArrayFromImage(pred_img)  # (Z,Y,X)
    result = pred_np.copy()
    filled = []

    if max_hole_mm3 <= 0:
        out = sitk.GetImageFromArray(result.astype(np.uint8))
        out.CopyInformation(pred_img)
        return (out, filled) if report_filled else out

    max_hole_threshold_vox = max_hole_mm3 / voxel_vol

    try:
        import cc3d
        def _cc(mask):
            labeled = cc3d.connected_components(mask, connectivity=26)
            return labeled, labeled.max()
    except ImportError:
        struct26_lbl = np.ones((3, 3, 3), dtype=int)
        def _cc(mask):
            labeled, n = _ndi.label(mask, structure=struct26_lbl)
            return labeled, n

    bg_mask = pred_np == 0
    if not bg_mask.any():
        out = sitk.GetImageFromArray(result.astype(np.uint8))
        out.CopyInformation(pred_img)
        return (out, filled) if report_filled else out

    labeled, n = _cc(bg_mask)
    struct26 = np.ones((3, 3, 3), dtype=int)

    for cid in range(1, n + 1):
        comp = labeled == cid
        size = int(comp.sum())
        if size >= max_hole_threshold_vox:
            continue

        zs, ys, xs = np.where(comp)
        if (zs.min() == 0 or zs.max() == comp.shape[0] - 1 or
                ys.min() == 0 or ys.max() == comp.shape[1] - 1 or
                xs.min() == 0 or xs.max() == comp.shape[2] - 1):
            continue

        # pooled majority over the whole hole, not a per-voxel walk, to
        # avoid an order-dependent result near multi-label interfaces
        dilated = _ndi.binary_dilation(comp, structure=struct26)
        border_labels = pred_np[dilated & ~comp]
        border_labels = border_labels[border_labels != 0]
        if border_labels.size == 0:
            continue

        majority_label = int(np.argmax(np.bincount(border_labels)))
        result[comp] = majority_label

        if report_filled:
            centroid_zyx = np.argwhere(comp).mean(axis=0)
            idx_xyz = (float(centroid_zyx[2]), float(centroid_zyx[1]), float(centroid_zyx[0]))
            phys = pred_img.TransformContinuousIndexToPhysicalPoint(idx_xyz)
            filled.append({
                "assigned_label": majority_label,
                "voxel_count": size,
                "physical_mm3": round(size * voxel_vol, 2),
                "index_centroid_xyz": tuple(round(v, 1) for v in idx_xyz),
                "physical_centroid_xyz_mm": tuple(round(v, 1) for v in phys),
            })

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(pred_img)

    if report_filled:
        return out, filled
    return out


def remove_small_lesions_mm3(pred_img, wt_mm3, rc_mm3, report_removed=False):
    """Remove connected components below threshold. Tumor labels {1,2,3}
    are filtered as one union so nested anatomy survives; RC (4) is
    filtered independently."""
    spacing = pred_img.GetSpacing()
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    wt_threshold_vox = wt_mm3 / voxel_vol
    rc_threshold_vox = rc_mm3 / voxel_vol

    pred_np = sitk.GetArrayFromImage(pred_img)  # (Z,Y,X)

    try:
        import cc3d
        def _cc(mask):
            labeled = cc3d.connected_components(mask, connectivity=26)
            return labeled, labeled.max()
    except ImportError:
        from scipy import ndimage
        struct26 = np.ones((3, 3, 3), dtype=int)
        def _cc(mask):
            labeled, n = ndimage.label(mask, structure=struct26)
            return labeled, n

    result = np.zeros_like(pred_np)
    removed = []

    def _record_removed(comp, region_name, threshold_vox):
        zyx = np.argwhere(comp)
        centroid_zyx = zyx.mean(axis=0)
        idx_xyz = (float(centroid_zyx[2]), float(centroid_zyx[1]), float(centroid_zyx[0]))
        phys = pred_img.TransformContinuousIndexToPhysicalPoint(idx_xyz)
        removed.append({
            "region": region_name,
            "voxel_count": int(comp.sum()),
            "physical_mm3": round(comp.sum() * voxel_vol, 2),
            "threshold_mm3_equivalent": round(threshold_vox * voxel_vol, 2),
            "index_centroid_xyz": tuple(round(v, 1) for v in idx_xyz),
            "physical_centroid_xyz_mm": tuple(round(v, 1) for v in phys),
        })

    tumor_mask = np.isin(pred_np, [1, 2, 3])
    if tumor_mask.any():
        labeled, n = _cc(tumor_mask)
        for cid in range(1, n + 1):
            comp = labeled == cid
            if comp.sum() >= wt_threshold_vox:
                result[comp] = pred_np[comp]
            elif report_removed:
                _record_removed(comp, "WT(1,2,3)", wt_threshold_vox)

    rc_mask = pred_np == 4
    if rc_mask.any():
        labeled, n = _cc(rc_mask)
        for cid in range(1, n + 1):
            comp = labeled == cid
            if comp.sum() >= rc_threshold_vox:
                result[comp] = 4
            elif report_removed:
                _record_removed(comp, "RC(4)", rc_threshold_vox)

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(pred_img)

    if report_removed:
        return out, removed
    return out


def main():
    if len(sys.argv) < 4:
        print("Usage: python postprocess.py <input_dir> <output_dir> <wt_mm3> [<rc_mm3>] [<hole_mm3>]")
        sys.exit(1)

    in_dir, out_dir = sys.argv[1], sys.argv[2]
    wt_mm3 = float(sys.argv[3])
    rc_mm3 = float(sys.argv[4]) if len(sys.argv) > 4 else wt_mm3
    hole_mm3 = float(sys.argv[5]) if len(sys.argv) > 5 else 15.0
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".nii.gz"))
    print(f"Filtering {len(files)} files: hole-fill threshold={hole_mm3}mm^3, "
          f"WT threshold={wt_mm3}mm^3, RC threshold={rc_mm3}mm^3  -> {out_dir}")

    for fname in files:
        img = sitk.ReadImage(os.path.join(in_dir, fname))

        holefilled, filled = fill_small_holes_mm3(img, hole_mm3, report_filled=True)
        filtered, removed = remove_small_lesions_mm3(holefilled, wt_mm3, rc_mm3, report_removed=True)

        orig_arr = sitk.GetArrayFromImage(img)
        new_arr = sitk.GetArrayFromImage(filtered)
        changed_voxels = int((orig_arr != new_arr).sum())

        sitk.WriteImage(filtered, os.path.join(out_dir, fname))
        print(f"[OK] {fname}  (spacing={img.GetSpacing()}, voxels changed: {changed_voxels}, "
              f"holes filled: {len(filled)}, components removed: {len(removed)})")
        for f in filled:
            print(f"      filled hole: {f['voxel_count']} voxels ({f['physical_mm3']} mm^3) "
                  f"-> label {f['assigned_label']} "
                  f"at voxel-index (x,y,z)={f['index_centroid_xyz']}  "
                  f"physical (x,y,z) mm={f['physical_centroid_xyz_mm']}")
        for r in removed:
            print(f"      removed {r['region']}: {r['voxel_count']} voxels ({r['physical_mm3']} mm^3, "
                  f"threshold was {r['threshold_mm3_equivalent']} mm^3) "
                  f"at voxel-index (x,y,z)={r['index_centroid_xyz']}  "
                  f"physical (x,y,z) mm={r['physical_centroid_xyz_mm']}")

    print("Done.")


if __name__ == "__main__":
    main()
