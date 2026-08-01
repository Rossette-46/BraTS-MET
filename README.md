# Segmentation of Pre- and Post-treatment BraTS-METS using an Ensemble Model

Automated segmentation of brain metastases on multi-parametric MRI, covering
both pre-operative and post-operative studies. Four classes are labelled:
non-enhancing tumour core, surrounding non-enhancing FLAIR hyperintensity,
enhancing tumour, and — in post-operative studies — the resection cavity.

Input is four co-registered, skull-stripped sequences per study (T1n, T1c, T2w,
T2-FLAIR). Output is a label file in the input's native geometry.

---

## Method

### Why ensemble

Two networks are trained independently on the same data:

- **nnU-Net v2 with a Residual Encoder (ResEncL)** — a self-configuring
  framework that derives patch size, batch size, resampling and network depth
  from the dataset's own properties, with residual blocks and a larger
  parameter budget in the encoder.
- **MedNeXt (M, kernel 3)** — a ConvNeXt-style 3D network keeping the U-Net
  skeleton but using depthwise-separable blocks with large kernels and inverted
  bottlenecks, giving a wide receptive field without attention's memory cost.

Neither dominates. Their overall accuracy is close, but they are stronger on
different regions, and they fail at the resection cavity in opposite ways: one
over-predicts small spurious cavities on pre-operative studies, the other
under-detects real ones. That asymmetry is what the ensemble exploits — small
false positives can be removed afterwards by a size filter, whereas a cavity
the network never found cannot be recovered by any post-hoc step.

### Region-specific ensembling

Rather than averaging probabilities, each region is assigned wholesale to the
model that segments it better. MedNeXt supplies the three tumour classes;
nnU-Net supplies the resection cavity:

```
result = mednext_prediction
result[result == RC] = background          # discard MedNeXt's cavity entirely
result[nnunet_prediction == RC] = RC       # stamp nnU-Net's cavity on top
```

No voxel is contested and no probability calibration between two differently
trained networks is needed. The cost is that a hard override cannot express
uncertainty — where the two disagree at a cavity boundary, nnU-Net wins
unconditionally, which slightly perturbs tumour labels there.

### Post-processing

Applied to the merged map, in this order.

**Hole filling.** Small fully-enclosed background pockets are relabelled with
the majority label of the surrounding tissue. A pocket qualifies only if it is
both below a volume threshold and does not touch the volume's outer boundary —
a genuinely enclosed hole cannot be connected to the exterior background, so
those two conditions together are sufficient. This runs first so the size
filter sees each lesion's true extent rather than one fragmented by a few
missed voxels.

**Lesion-size filtering.** Connected components below a volume threshold are
removed. Thresholds are given in mm³ and converted to voxel counts using each
study's own header spacing, because native voxel volumes vary by up to 4×
across contributing sites. Components are computed on the **union** of the
tumour labels, not per label: a small necrotic core enclosed by an enhancing
rim has no neighbouring core voxels, so per-label connectivity would see an
isolated speck and delete the centre of a large, correctly segmented lesion.
The cavity is filtered independently, since it is not nested inside any other
label.

The threshold is a real trade-off, not a tuning detail. Raising it raises
overlap scores and lowers small-lesion detection, because filtering removes
correctly-detected small lesions along with spurious ones.

---

## Setup

Two **separate** environments are required. nnU-Net v2 and MedNeXt (built on
nnU-Net v1) share module names and use different names for the same
environment variables. Do not install them together.

### nnU-Net v2

```bash
conda create -n nnunet python=3.10 -y
conda activate nnunet
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install nnunetv2
```

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

### MedNeXt

```bash
conda create -n mednext python=3.10 -y
conda activate mednext
pip install torch --index-url https://download.pytorch.org/whl/cu130
git clone https://github.com/MIC-DKFZ/MedNeXt.git
cd MedNeXt && pip install -e .
```

```bash
export nnUNet_raw_data_base=/path/to/mednext_v1/nnUNet_raw_data_base
export nnUNet_preprocessed=/path/to/mednext_v1/nnUNet_preprocessed
export RESULTS_FOLDER=/path/to/mednext_v1/RESULTS_FOLDER
```

Then, **with the `mednext` environment active**:

```bash
bash setup_patches.sh
```

This one is required — MedNeXt will not run without it. The fork renamed its
package to `nnunet_mednext`, but several call sites still build import paths
from the literal string `nnunet`, so imports fail; and PyTorch ≥ 2.6 changed
`torch.load` to refuse pickled checkpoints by default, which blocks loading any
nnU-Net v1 checkpoint. The script installs a module-alias shim, patches the
two affected `torch.load` calls, and appends the trainer class used below.
It is idempotent and derives all paths from the live interpreter.

Environment variables do not persist across shells or compute nodes. Re-export
them in every new session.

---

## Training

Trained weights are not distributed; train both networks before running
inference.

Edit paths in the code before using it. 
```bash
conda activate nnunet
python bratsmets/convert_to_nnunet.py
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL --verify_dataset_integrity
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans
```

```bash
conda activate mednext
python bratsmets/convert_to_mednext.py
mednextv1_plan_and_preprocess -t 501 --verify_dataset_integrity
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3_500epochs 501 0 --npz
```

Both frameworks generate their own 5-fold cross-validation split during
preprocessing and train fold 0 above; the held-out fifth is used for
evaluation. Both derive their results directory from the **trainer class
name**, so two experiments sharing a class name silently overwrite each
other's checkpoints — give any new run its own class.

Jobs are subject to wall-clock limits on most clusters. Pass `-c` to
`nnUNetv2_train` to resume, or it silently restarts from epoch 0.

---

## Inference

### 1. Prepare the input

Both frameworks read a flat directory of `<case>_0000.nii.gz` …
`<case>_0003.nii.gz`, in channel order T1n, T1c, T2w, T2-FLAIR:

```bash
python bratsmets/convert_validation.py
```

### 2. Predict with each network

```bash
conda activate nnunet
nnUNetv2_predict -i <flat_input_dir> -o predictions_nnunet \
  -d Dataset001_BraTSMET -c 3d_fullres -f 0 \
  -tr nnUNetTrainer -p nnUNetResEncUNetLPlans
```

```bash
conda activate mednext
mednextv1_predict -i <flat_input_dir> -o predictions_mednext \
  -t 501 -m 3d_fullres -tr nnUNetTrainerV2_MedNeXt_M_kernel3_500epochs -f 0
```

Both resample their output back to each study's native geometry, so the two
folders are voxel-aligned and share filenames.

`mednextv1_predict` warns that it cannot find a postprocessing file. **This is
expected and desired** — it means the framework's built-in largest-component
postprocessing is skipped. That heuristic assumes a single lesion and discards
correct lesions on multi-focal metastatic disease.

### 3. Ensemble and post-process

```bash
python bratsmets/ensemble.py \
  --mednext-dir predictions_mednext \
  --nnunet-dir  predictions_nnunet \
  --out-dir     predictions_final \
  --wt-mm3 5 --rc-mm3 5 --hole-mm3 15
```

This performs the region-specific merge and both post-processing steps in one
pass. `--no-postproc` writes the raw merge instead.

To sweep thresholds without recomputing the merge, run the filter directly:

```bash
python bratsmets/postprocess.py <input_dir> <output_dir> 5 5 15
```

Arguments are the tumour threshold, the cavity threshold and the hole-filling
threshold, all in mm³.

---

## Evaluation

`bratsmets/evaluate_dice.py` computes subject-wise Dice per region,
needing only SimpleITK and numpy:

```bash
python bratsmets/evaluate_dice.py \
  --gt-dir  <nnUNet_raw>/Dataset001_BraTSMET/labelsTr \
  --pred    nnunet=<.../fold_0/validation> \
  --pred    mednext=<.../fold_0/validation_raw> \
  --out-csv internal_dice.csv
```

Both frameworks write predictions for their own validation fold at the end of
training, already resampled to native geometry, so no extra inference run is
needed to evaluate the held-out fold. Use MedNeXt's `validation_raw`, not
`validation_raw_postprocessed`.

Regions absent from the reference are handled explicitly rather than averaged
blindly, and means are reported both over all studies and over only those whose
reference contains the region. This matters for the resection cavity: most
studies are pre-operative, so a model that never predicts a cavity scores near
1.0 on the naive average while one that finds most real cavities scores far
lower.

Subject-wise Dice is **not** the lesion-wise metric the challenge portal
reports — it compares whole label volumes per study and is indifferent to how
many distinct lesions were found, so it reads substantially higher on
multi-focal disease. For lesion-wise metrics,
`bratsmets/evaluate_lesionwise.py` wraps
[panoptica](https://github.com/BrainLesion/panoptica) via
[brats_evaluator](https://github.com/Astarakee/brats_evaluator):

```bash
conda create -n bratseval python=3.10 -y && conda activate bratseval
pip install "panoptica==1.4.0" "auxiliary==0.1.3" pandas tqdm
git clone https://github.com/Astarakee/brats_evaluator.git
```

**Pin `auxiliary`.** brats_evaluator's own requirements pin panoptica but leave
`auxiliary` unpinned; versions from 0.2.0 onward drop an argument its I/O path
relies on, so a fresh install fails with `TypeError: read_image() got an
unexpected keyword argument 'maintain_dtype'`. Report the script's
`lesionwise_dsc`, not panoptica's `sq_dsc` — the latter is Segmentation
Quality, which ignores false positives and reads much higher.

---

## Known issues

- **Trainer class names determine checkpoint paths** in both frameworks, so
  they cannot be renamed after training without breaking the path.
- **Optimisers are not matched** between the two networks (SGD for nnU-Net,
  AdamW for MedNeXt — each framework's own default). The comparison is between
  two complete, independently tuned systems.
- **The hard override is not uncertainty-aware.** At cavity boundaries
  nnU-Net's decision overwrites MedNeXt's tumour labels unconditionally.
- **Environment variables are not persistent.** Re-export on every new shell
  and compute node.
