"""
Region-specific ensemble of nnU-Net v2 (ResEncL) and MedNeXt for BraTS-MET.
MedNeXt supplies NETC/SNFH/ET, nnU-Net supplies RC; hole filling and
lesion-size filtering run after the merge.

Modules: convert_to_nnunet, convert_to_mednext, convert_validation,
ensemble, postprocess, evaluate_dice, evaluate_lesionwise, ensure_outputs.
"""

__version__ = "1.0.0"
