# Downstream segmentation training

Run commands from this directory. `train.sh` launches `train.py` through SLURM
and forwards additional arguments.

## Select the segmentation task

```bash
python train.py --classes nodule
python train.py --classes lung+nodule

# Through the existing SLURM launcher:
sbatch train.sh --classes lung+nodule

# One training and validation batch, without a checkpoint/report:
python train.py --classes lung+nodule --fast-dev-run
```

`--classes` overrides `classes` in `configs/training.yaml`; the default is `nodule`.
The model output channels are derived automatically. Both tasks reuse
`configs/splits/real_split.json` and the same real-data subset selection.

| Setting | nodule | lung+nodule |
| --- | --- | --- |
| Required masks | nod_masks | nod_masks and ts_masks |
| Target IDs | 0 background, 1 nodule | 0 background, 1 lung, 2 nodule |
| Output channels | 1 | 3 |
| Prediction | sigmoid(logit) > 0.5 | argmax over class logits |
| Training loss | sigmoid Dice + BCE | foreground softmax Dice + multiclass CE |

Dataset folder keys are `images`, `nod_masks`, and `ts_masks`. Each folder contains
one NIfTI per case directly inside it, with matching filename stems. Patient IDs
are filename stems such as `LIDC-IDRI-0001` and `LNDb-0001`, without a folder prefix.
Nodule mode does not inspect or require the TS folder.

`data/transforms.py` checks CT/mask shapes and affines, then constructs the target
at loading time. In joint mode, TS labels 28, 29, 30, 31, and 32 become lung;
positive nodule instance IDs become class 2 last, overriding lung. Nodules outside
the TS lung region are retained. All other TS labels become background. The
source mask tensors are removed after composing `label`. Aligned masks are not
resampled. The constants and class semantics live in `tasks.py`.

Inspect a batch using:

```bash
python -m data.check_data --classes lung+nodule
python -m data.dataset_index --classes lung+nodule
```

## Real + synthetic training

Validation and test always contain the complete, fixed **real-only** splits.
Only the training cohort changes. Available synthetic sources are `lungddpm`,
`land`, `maisi_ct`, and `maisi_maskct`, with paths in `configs/datasets.yaml`.
The former placeholder names `maisi_real_masks` and `maisi_synthetic_masks`
have been replaced by `maisi_ct` and `maisi_maskct` respectively.

Both `real_percent` and `synthetic_percent` refer to the full real training
pool **after validation is removed**, not to the resulting mixed dataset or
the available synthetic pool. Counts round to the nearest integer, with .5
rounded upward. For the saved 932-case training pool, 50/50 gives 466 real
and 466 synthetic CTs; 100/50 gives 932 real and 466 synthetic CTs.

For real-conditioned generators (`lungddpm`, `land`, `maisi_ct`):

- Partial mixes summing to 100% select the real subset first, then use synthetic
  versions of exactly the remaining training patients. No patient occurs as
  both real and synthetic. For odd cohort sizes the synthetic count is the
  exact complement of the rounded real count.
- When either percentage is 100%, real and synthetic subsets are selected
  independently from the real training patient pool, allowing both versions
  of the same patient. Synthetic-only training is also supported.
- Other mixes with both percentages strictly between 0 and 100 are rejected
  until their overlap policy is defined.
- Selection uses the dataset split seed (42), preserving LIDC/LNDb proportions.
  The same percentages choose the same patients across generators. Existing
  real-only subset selection is preserved.
- Generated validation/test patients are excluded, even at 100% synthetic.
  Unknown conditioning IDs and missing selected CTs/masks produce errors;
  the loader does not substitute different patients.

`maisi_maskct` has independent synthetic identities. Its cases are sampled
without replacement from sorted IDs using the same seed, with the requested
count based on the real training pool. Insufficient supply produces an error.
Its binary nodule masks come from `masks`, and lung labels from `masks_132`.
As with real TS masks, lung labels are 28--32 and nodules override lung labels.

LungDDPM uses the original real conditioning masks; LAND and MAISI CT use the
generator-specific mask paths supplied in the dataset config. Mask folder
settings accept either one path or a list, rejecting duplicate case IDs across
folders. Nodule mode needs only `nod_masks`; joint mode also needs `ts_masks`.
All selected masks must match the CT shape and affine at loading time; this
pipeline checks alignment but does not resample masks.

Example commands, run from `downstream`:

```bash
# Inspect the selected counts and export exact paths/IDs without training.
python -m data.experiment_data \
  --experiment configs/experiments/real_50_lungddpm_50.yaml \
  --classes lung+nodule --manifest selected_cases.json

# Check one training and validation batch on the training machine.
python train.py --experiment configs/experiments/real_50_lungddpm_50.yaml \
  --classes lung+nodule --fast-dev-run

# Submit the full run; the launcher forwards the experiment override.
sbatch --job-name=train_real50_lungddpm50 train.sh \
  --experiment configs/experiments/real_50_lungddpm_50.yaml --classes lung+nodule
```

Equivalent 50/50 configs are included for all four generators. To change the
mix, copy one and edit its `name`, `real_percent`, and `synthetic_percent`.
For real-only runs, use `synthetic_percent: 0` and `synthetic_dataset: null`.
The cohort stays fixed across epochs; the DataLoader shuffles all selected
samples together, without enforcing per-batch proportions or oversampling.
Each run saves `configs/selected_cases.json` with the exact train/val/test
paths, unique `sample_id`, real/synthetic `source`, generator/dataset, and
conditioning patient/dataset (empty for independently generated cases).

Run selection and loading regressions with:

```bash
python -m pytest tests/test_mixed_data.py
```

## Losses and model selection

The training objective is `dice_weight * Dice + entropy_weight * entropy`, where
entropy is BCE (`bce_weight`) for nodule mode and multiclass CE (`ce_weight`) for
joint mode. Multiclass Dice averages lung and nodule Dice losses, excluding
background. CE includes all three classes.

The configured checkpoint monitor is **val_dice_nodule**, maximized. The early
stopping monitor is **val_loss_nodule**, minimized. Early stopping retains the
existing patience of 10 and min_delta of 0.001. These monitors can select a best
epoch different from the epoch when training stops.

In binary mode, `val_loss_nodule` equals the global validation loss. In joint mode,
it is a separate nodule-vs-rest diagnostic: Dice plus BCE evaluated using the
softmax probability of class 2 and the binary target `label == 2`. The stable
binary log-odds are:

```text
z_nodule - logsumexp(z_background, z_lung)
```

The sigmoid of these log-odds equals the softmax nodule probability. This loss
uses all voxels, penalizing false positives and false negatives, including on
nodule-negative scans. It uses `dice_weight` and `ce_weight`; it is not an additive
component of multiclass CE and is not added to the training loss. Its purpose is
to stop when nodule performance stops improving, even if lung learning continues.

Training logs global loss and its Dice/entropy components. Validation/test logs
global loss, nodule loss, and separate hard Dice for each foreground class.
Hard Dice is computed per volume, then averaged; both-empty masks score 1 and
one-empty masks score 0. These loop metrics include all cases.

The selected task and resolved model configuration are saved in checkpoints.
Each run also saves effective settings in `configs/training.yaml` and the original
input file in `configs/training_source.yaml`. W&B run names include the task.

## Post-training report

A successful full run reloads the selected checkpoint and writes artifacts to
`<output_dir>_<classes>/<experiment>/<timestamp>/report/`:

- `report.png`: global training/validation loss curves, nodule validation loss,
  class metric table, and the best/worst nodule examples. This is the only PNG
  report in either mode. Joint examples overlay both lung and nodule masks on
  the same nodule-selected axial slice.
- `validation_metrics.csv`: per-case Dice, precision, and recall. Joint mode uses
  one row per eligible case/class, with `class` and `class_id` columns.
- `summary.json`: checkpoint, task, per-class mean metrics, eligible/excluded case
  counts, and example IDs. Top-level metrics remain nodule metrics; `per_class`
  contains all reported classes.

Joint reports include individual **background (0), lung (1), and nodule (2)**
metrics. Lung means class 1, excluding nodule voxels. Predictions use argmax, so
no classes are pooled together. Binary reports retain nodule-only metrics.

Reporting makes one validation pass with gradients disabled, zero DataLoader
workers, and pinned memory disabled. It does not evaluate train/test data.
`--fast-dev-run` skips reporting because no checkpoint is saved.

For each class independently, cases with no ground-truth voxels of that class
are excluded from report metrics and example ranking. A nodule-negative case
still contributes lung/background metrics. False positives in class-absent cases
are therefore not represented in the report's positive-case mean; the validation
loss and loop Dice above include those cases.

Metrics use whole-volume counts:

- Dice = 2 TP / (predicted class voxels + ground-truth class voxels)
- Precision = TP / predicted class voxels; zero when nothing is predicted
- Recall = TP / ground-truth class voxels

Tables average eligible cases arithmetically, without pooling voxel counts.
Displayed examples are ranked by nodule Dice, with patient ID breaking ties.
Up to two best and two worst distinct cases are shown. Slices maximize the nodule
ground-truth area after RAS reorientation (axis flips/permutations, not oblique
resampling). Nodule ground truth is green and predictions are magenta. In joint
mode, the corresponding ground-truth/predicted lung masks are overlaid in faint
cyan on the same slices. The table includes separate lung metrics.

Reporting requires Matplotlib and NiBabel in addition to training dependencies.
Run regression tests with:

```bash
python -m pytest tests/test_segmentation_modes.py tests/test_reporting.py
```


## DataLoader multiprocessing

Worker processes use `data.sharing_strategy: file_system` to avoid the Linux
file-descriptor transfer path that can fail with `received 0 items of ancdata`
when descriptors are exhausted. The strategy is set inside each worker, so it
also applies with the `spawn` start method. MONAI transform seeding is preserved.
Set this option to `null` to retain the platform's default strategy.

`data.prefetch_factor: 1` limits each worker to one prefetched batch of full CT
volumes. Both worker options are ignored with `data.num_workers: 0`, including
`--fast-dev-run`. Train, validation, and test loaders use the same settings.

If loading still fails, set `data.num_workers: 0` for a diagnostic run. On the
compute node, inspect `ulimit -n`, `df -h /dev/shm`, and the Slurm job's memory/OOM
logs. The `file_system` strategy still uses shared memory; it does not remove
RAM or shared-memory capacity limits. Reduce worker count if memory is tight.

Run the worker regression tests from `downstream` in the training environment:

```bash
python -m pytest tests/test_dataloader_workers.py
```
