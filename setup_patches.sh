#!/usr/bin/env bash
# setup_patches.sh -- applies the fixes MedNeXt needs before it will run
# (live inside the installed package, not this repo, so a fresh
# `pip install -e .` doesn't include them). nnU-Net v2 is not touched.
# Idempotent; all paths derived from the live interpreter.
#
# USAGE (inside the activated MedNeXt environment): bash setup_patches.sh

set -euo pipefail

echo "=============================================================="
echo " MedNeXt environment patches"
echo "=============================================================="

if ! python -c "import nnunet_mednext" 2>/dev/null; then
    echo "ERROR: nnunet_mednext is not importable."
    echo "       Activate the MedNeXt environment first."
    exit 1
fi

SITE_PACKAGES="$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
PKG_DIR="$(python -c "import nnunet_mednext, os; print(os.path.dirname(nnunet_mednext.__file__))")"
echo "  site-packages : ${SITE_PACKAGES}"
echo "  package dir   : ${PKG_DIR}"
echo

echo "[1/3] module-alias shim ('nnunet' -> 'nnunet_mednext')"
ALIAS_PTH="${SITE_PACKAGES}/nnunet_alias.pth"
if [ -f "${ALIAS_PTH}" ]; then
    echo "      already present: ${ALIAS_PTH}"
else
    echo "import sys, nnunet_mednext; sys.modules.setdefault('nnunet', nnunet_mednext)" > "${ALIAS_PTH}"
    echo "      created: ${ALIAS_PTH}"
fi

if python -c "import nnunet.preprocessing" 2>/dev/null; then
    echo "      verified: 'import nnunet' resolves"
else
    echo "      ERROR: shim written but 'import nnunet' still fails."
    exit 1
fi
echo

echo "[2/3] torch.load(weights_only=False) fixes"

patch_torch_load () {
    local file="$1" ; local from="$2" ; local to="$3"
    if [ ! -f "${file}" ]; then
        echo "      WARNING: not found, skipping: ${file}"
        return 0
    fi
    if grep -qF "${to}" "${file}"; then
        echo "      already patched: $(basename "${file}")"
        return 0
    fi
    if ! grep -qF "${from}" "${file}"; then
        echo "      WARNING: expected pattern absent in $(basename "${file}")."
        echo "               Upstream code may have changed -- verify manually."
        return 0
    fi
    python - "${file}" "${from}" "${to}" <<'PYEOF'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()
with open(path, 'w', encoding='utf-8') as f:
    f.write(src.replace(frm, to))
PYEOF
    echo "      patched: $(basename "${file}")"
}

patch_torch_load \
    "${PKG_DIR}/training/model_restore.py" \
    "torch.load(i, map_location=torch.device('cpu'))" \
    "torch.load(i, map_location=torch.device('cpu'), weights_only=False)"

patch_torch_load \
    "${PKG_DIR}/training/network_training/network_trainer.py" \
    "saved_model = torch.load(fname, map_location=torch.device('cpu'))" \
    "saved_model = torch.load(fname, map_location=torch.device('cpu'), weights_only=False)"

echo

echo "[3/3] trainer class"

TRAINER_FILE="${PKG_DIR}/training/network_training/MedNeXt/nnUNetTrainerV2_MedNeXt.py"
CLS="nnUNetTrainerV2_MedNeXt_M_kernel3_500epochs"

if [ ! -f "${TRAINER_FILE}" ]; then
    echo "      ERROR: trainer file not found: ${TRAINER_FILE}"
    exit 1
fi

if grep -q "class ${CLS}" "${TRAINER_FILE}"; then
    echo "      already present: ${CLS}"
else
    cat >> "${TRAINER_FILE}" <<EOF


class ${CLS}(nnUNetTrainerV2_MedNeXt_M_kernel3):
    """Added by setup_patches.sh. Identical to the base trainer except the
    epoch budget, which affects only the LR schedule and training loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 500
EOF
    echo "      appended: ${CLS}"
fi

python - <<PYEOF
import importlib, sys
m = importlib.import_module(
    "nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt")
if not hasattr(m, "${CLS}"):
    print("      ERROR: ${CLS} not importable after patching")
    sys.exit(1)
print("      verified: ${CLS} imports cleanly")
PYEOF

echo
echo "=============================================================="
echo " Done."
echo "=============================================================="
echo
echo "Reminder -- these must be exported before any MedNeXt command;"
echo "they do not persist across shells:"
echo "    export nnUNet_raw_data_base=/path/to/nnUNet_raw_data_base"
echo "    export nnUNet_preprocessed=/path/to/nnUNet_preprocessed"
echo "    export RESULTS_FOLDER=/path/to/RESULTS_FOLDER"
