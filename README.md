## COMP3055 Coursework Experiment Suite

This repository contains reproducible code for all tasks described in the coursework brief:

1. **Task 1** – Loads CIFAR-10 and builds PCA feature sets that retain 10%, 30%, 50%, 70%, and 100% of the variance (plus the standardized original vectors). Metadata is exported to `outputs/task1/feature_metadata.json`.
2. **Task 2** – Trains an `sklearn.neural_network.MLPClassifier` with two experiment tracks:
   - *Feature sweep*: evaluates the PCA feature sets with 5-fold stratified cross-validation and test-set reporting.
   - *Hyper-parameter sweep*: varies the hidden layer layout, learning rate, and regularisation on the original features.
3. **Task 3** – Provides two alternative models:
   - *Random Forest* (`sklearn.ensemble.RandomForestClassifier`) mirroring the PCA sweeps from Task 2.
   - *CNN* (ResNet18 tailored for CIFAR-10) trained end-to-end on the raw images with data-augmentation sweeps and GPU-accelerated hyper-parameter searches.

The code is organised as a Python package (`src/mlcw`) so everything can be executed via `python -m mlcw.run_pipeline ...` or the helper shell script below. **All Task 3 variants expect an NVIDIA GPU**; the provided `scripts/run_all.sh` aborts early if `nvidia-smi` is unavailable.

> **Note:** No virtual environment is created. Install dependencies directly into your preferred Python environment.

### Install Dependencies

```bash
python3 -m pip install --user -r requirements.txt
```

### Run All Experiments

```bash
scripts/run_all.sh
```

Environment variables or CLI flags can override defaults:

| Option | Default | Description |
| --- | --- | --- |
| `DATA_ROOT` / `--data-root` | `./data` | Download/cache directory for CIFAR-10. |
| `OUTPUT_ROOT` / `--output-root` | `./outputs` | Where metrics, plots, and trained models are written. |
| `TRAIN_SUBSET` / `--train-subset` | `10000` | Training subset size. Set to `0` for all 50k images. |
| `TEST_SUBSET` / `--test-subset` | `2000` | Test subset size. Set to `0` for full 10k test set. |
| `--pca-targets` | `10 30 50 70 100` | Variance percentages for PCA compression. |
| `--cv-splits` | `5` | Number of folds for cross-validation. |
| `--skip-task2`, `--skip-task3` | `False` | Skip MLP or Task 3 experiments entirely. |
| `TASK3_VARIANTS` / `--task3-backends` | `rf` | Comma-separated list of Task 3 backends to run: `rf`, `cnn`, or both (e.g. `rf,cnn`). |

Additional MLP / Random Forest / CNN parameters can be passed (see `python -m mlcw.run_pipeline -h`).

### Outputs

```
outputs/
  task1/feature_metadata.json
  task2/
    mlp_feature_sweep/
    mlp_hparam_sweep/
    plots/
  task3/
    rf_feature_sweep/
    rf_hparam_sweep/
    cnn_feature_sweep/
    cnn_hparam_sweep/
    plots/
```

Each sub-folder contains CSV summaries (per-fold metrics + aggregated results), JSON classification reports (per class F1, precision, recall), trained model dumps (`joblib` for sklearn models and `.pt` for CNN weights), and the requested plots.
