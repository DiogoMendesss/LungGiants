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

## Input intensity range

The training/debug configs and DataModule default to preprocessed volumes:

```yaml
data:
  in_min: 0.0
  in_max: 1.0
```

With exactly these bounds, loading validates that every image voxel is finite
and in the inclusive range [0, 1], without clipping or rescaling. Out-of-range
values, NaN, and infinity raise an error identifying the case. Shape and affine
checks and segmentation-label construction still run.

For inputs that need normalization, set `in_min` and `in_max` to their intensity
window (for example, -1000 and 1000 for raw HU). Values are clipped to that
window and linearly mapped to [0, 1]. Both bounds must be finite and
`in_min < in_max`. Rename `hu_min`/`hu_max` in custom configs to
`in_min`/`in_max`; use 0 and 1 for already preprocessed volumes.

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


## Standalone validation and test evaluation

From `downstream`, evaluate one saved checkpoint with:

```bash
python evaluate.py --model-path /path/to/run/checkpoints/best.ckpt
```

The shell launcher supports direct execution and Slurm submission. From
`downstream`, activate your training environment and run either:

```bash
bash evaluate.sh \
  --model /path/to/run/checkpoints/best.ckpt \
  --set both \
  --output-path /path/to/new-report \
  --device cuda --precision 16-mixed --save-masks

sbatch evaluate.sh \
  --model /path/to/run/checkpoints/best.ckpt \
  --set test \
  --output-path /path/to/new-test-report
```

`--set` accepts `val`, `test`, or `both` (default). Only selected splits have
their data loaded and reports generated. `--model` and `--output-path` are
aliases for `--model-path` and `--output-dir`. An explicit output directory must
not exist. Remove the output argument from the script to use an automatic
timestamped directory.
Use `bash evaluate.sh --help` to see all evaluator options.

Like `train.sh`, the launcher contains a direct `python -u evaluate.py` command
with editable arguments and a final `"$@"` for command-line overrides. Edit the
model and output paths in `evaluate.sh`, or provide them when launching it.
Run or submit from `downstream/` with your training environment activated.

Slurm defaults match `train.sh` (`gpu_min32gb` partition/QoS); override these
with `sbatch --partition=... --qos=... evaluate.sh ...` if needed. Logs go to
`slurm_out/evaluate/slurm_evaluate_nodules.<job-id>.out/.err`. Create the log
directory before submission with `mkdir -p slurm_out/evaluate`.

Alternatively, set `MODEL_PATH` in `evaluate.py` or as an environment variable,
then run `python evaluate.py`. The evaluator loads the saved network parameters
and discovers the run's `configs/training.yaml`, `datasets.yaml`, and
`real_split.json`. It reuses the saved validation/test patients and checks them
against `selected_cases.json` when available. It does not load synthetic data,
recreate a split, train the model, or initialize W&B. Inference uses the same
transforms and hard-label decoding as training, one full volume at a time.

Optional arguments:

```bash
python evaluate.py --model-path /path/to/best.ckpt \
  --set both \
  --config-dir /path/to/run/configs \
  --data-root /nas-ctm01/datasets/public/LungNodule-nifti \
  --annotations-root /nas-ctm01/datasets/public/Lung_Nodule \
  --output-dir /path/to/new-evaluation-directory \
  --device cuda --precision 16-mixed --save-masks
```

Without overrides, annotations are read from the sibling `Lung_Nodule` directory.
Validation annotations come from its `train` folders; test annotations come from
its `test` folders. On Windows, saved `/nas-...` dataset roots are resolved through
`Z:/nas-...` if available. GPU/CPU selection is automatic; GPU precision defaults
to the saved training setting and CPU uses float32. `--save-masks` additionally
writes semantic predictions and assigned nodule-instance predictions as NIfTI.

Reports default to a new timestamped directory under the run's `evaluation/`.
Existing output directories are not overwritten. Separate `validation/` and
`test/` folders each contain:

| File | Contents |
| --- | --- |
| `report.html` | Readable mean ± sample standard deviation tables and counts |
| `case_metrics.csv` | Lung and nodule DSC, precision, recall for every case |
| `nodule_metrics.csv` | Individual nodule DSC, attributes, voxel counts, and status |
| `attribute_summary.csv` | All 3 size, 5 sphericity, and 5 spiculation classes |
| `component_warnings.csv` | Every excluded predicted component, its size, and affected labels |
| `summary.json` | Summaries, evaluation rules, resolved case paths, and checkpoint provenance |

Per-case metrics use the complete predictions. Lung means class 1 excluding
nodule voxels. Lung metrics are unavailable for a nodule-only checkpoint.
Cases with empty GT for a class remain in the CSV with unavailable metrics and
their predicted voxel counts, but are excluded from that class's means. For
nonempty GT, an empty prediction scores zero for all three metrics.

Per-nodule matching uses 26-connected predicted components. A component
overlapping exactly one GT instance is assigned **in full**, including voxels
outside GT; multiple fragments assigned to the same instance are unioned.
Components with no GT overlap are excluded only from the attribute analysis.
Components overlapping multiple GT instances are discarded with a warning;
affected nodule rows and the split's attribute summaries are provisional.
Every present GT nodule remains in the analysis, including misses with DSC 0.

The instance mask is the preprocessed mask on the inference grid. Its geometry
is checked against the CT. Filename `noduleX` maps to label `X+1` for LIDC and
label `X` for LNDb. Annotation labels absent after preprocessing are reported as
`missing_from_evaluation_grid`, counted but not scored. GT labels lacking an
annotation are still scored, with unavailable attributes and a diagnostic.
Do not substitute binary masks for the original instance-labelled NIfTI masks.

Size uses the stored `equiv_diameter_mm` class (1–3). Shape classes use
`ceil(median(valid reader ratings))`, accepting values in [1,5], including
fractional LNDb values such as 1.5 and 2.5; missing or
invalid values are recorded explicitly. Each attribute mean weights scored
nodules equally. `n_nodules` counts all identified nodules in the class;
`n_scored` excludes those missing from the evaluation grid. Reports also count
misses, ambiguous assignments, missing attributes, and contributing cases.
Standard deviation uses `ddof=1`; empty groups have unavailable means/std, and
single-observation groups have unavailable std.

Dependencies are the training environment plus SciPy and NiBabel. Protocol and
checkpoint/report integration tests can be run with:

```bash
python -m pytest tests/test_evaluate.py
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
