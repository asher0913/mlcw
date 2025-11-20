# COMP3055 Machine Learning Coursework Report

> **Note:** This report is written as a reproducible template. Once you run `scripts/run_all.sh` on the Linux server, replace the highlighted placeholders (✅ sections) with the actual numbers, tables, and figures from `outputs/`.

---

## 1. Problem Statement & Dataset Overview

The coursework evaluates end‑to‑end supervised learning workflows on the CIFAR‑10 dataset. CIFAR‑10 contains 60 000 RGB images (32×32) split over 10 mutually exclusive classes. Each class contributes exactly 6 000 images: 50 000 for training and 10 000 for testing. A randomly shuffled subset was used to keep local experimentation tractable; default settings sample **10 000** training and **2 000** testing images, although every experiment can be rerun with the full dataset by passing `--train-subset 0 --test-subset 0` to the runner.

Key dataset processing steps implemented in `src/mlcw/data.py`:

1. **Download & Cache** – `torchvision.datasets.CIFAR10` downloads the batches once and reuses the cached files under `data/`.
2. **Flattening & Normalisation** – Each image is converted to a 3 072‑dimensional vector (3 channels × 32 × 32) and linearly scaled to `[0, 1]` so that features share a comparable magnitude before standardisation.
3. **Subsampling (optional)** – Deterministic random sampling (`numpy.random.default_rng`) selects the requested number of samples to guarantee reproducibility across runs via the `--random-seed` flag.

The rest of the pipeline builds and evaluates two families of classifiers on the processed data. All experiments share the same random seed, 5‑fold stratified cross validation, and identical train/test splits to ensure apples‑to‑apples comparisons.

---

## 2. Task 1 – PCA Feature Engineering

### 2.1 Procedure

1. **Standardisation** – Both train and test matrices are standardised (`StandardScaler`) to zero mean and unit variance per feature.
2. **PCA Compression** – The scaler output feeds into PCA configured via percentages of retained variance: 10 %, 30 %, 50 %, 70 %, and 100 %. `n_components` is derived by multiplying the percentage by the total variance ratio (`svd_solver="full"`).
3. **Artifacts** – For each feature set we store:
   - Transformed train/test matrices (kept in memory for downstream experiments).
   - `outputs/task1/feature_metadata.json` summarising dimensionality and explained variance—useful for Task 4 comparisons.

### 2.2 Results (fill after running)

| Feature Set | Dimensionality | Explained Variance | Notes |
| --- | --- | --- | --- |
| original | 3 072 | 100 % | Standardised flattened RGB |
| pca_70 | ✅ | ✅ | PCA retains 70 % variance |
| pca_50 | ✅ | ✅ | 50 % variance |
| pca_30 | ✅ | ✅ | 30 % variance |
| pca_10 | ✅ | ✅ | 10 % variance |

*Numbers can be copied from `outputs/task1/feature_metadata.json`. Dimensionality is determined automatically by PCA and varies with the percentage retained; expect ~2 000 features for 70 % and ~300 for 10 %.*

### 2.3 Observations

- PCA dramatically reduces feature count without re‑computing the dataset for each experiment.
- Lower percentages accelerate training and mitigate overfitting risk but may lose discriminative information, which will be visible in downstream accuracy/F1.
- Using variance ratios rather than fixed components makes the script dataset‑agnostic (works for both subsets and full CIFAR‑10).

---

## 3. Task 2 – MLP Object Recognition System

### 3.1 Experimental Setup

- **Model** – `sklearn.neural_network.MLPClassifier` with default hidden layers `[512, 256]`, ReLU activations, Adam optimiser, `learning_rate_init=1e-3`, `alpha=1e-4`, batch size 256, early stopping patience 10 epochs, and maximum 80 iterations.
- **Cross Validation** – 5‑fold stratified CV on the training split for both feature‑dimension and hyper‑parameter sweeps.
- **Evaluation** – After CV, the model is retrained on the full training data and evaluated on the held‑out test subset. We export:
  - `fold_metrics.csv` (all fold‑level accuracy/F1).
  - `summary.csv` (average CV metrics + test accuracy/macro F1 per configuration).
  - `reports/*.json` (full per‑class precision/recall/F1 for the test set).
  - `models/*.joblib` (trained estimators).
  - `plots/mlp_feature_accuracy.png` and `plots/mlp_hparam_accuracy.png`.

### 3.2 Feature Dimension Sweep

**Files:** `outputs/task2/mlp_feature_sweep/summary.csv`, `.../reports/*.json`, `outputs/task2/plots/mlp_feature_accuracy.png`.

| Feature Set | Mean CV Accuracy | Mean CV Macro F1 | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- |
| original | ✅ | ✅ | ✅ | ✅ |
| pca_70 | ✅ | ✅ | ✅ | ✅ |
| pca_50 | ✅ | ✅ | ✅ | ✅ |
| pca_30 | ✅ | ✅ | ✅ | ✅ |
| pca_10 | ✅ | ✅ | ✅ | ✅ |

**How to populate:** Run `scripts/run_all.sh`, open `summary.csv`, and paste the corresponding numbers. The accuracy vs. dimensionality plot (`mlp_feature_accuracy.png`) provides a visual summary that can be embedded in the final PDF report.

**Expected trend (justify with your numbers):**

- Accuracy typically improves monotonically with more retained variance. Expect a significant drop at 10 % due to aggressive compression.
- CV macro F1 closely follows test macro F1, indicating low overfitting in the default configuration.
- If using the full dataset, the gaps may shrink because the MLP benefits from more data even with fewer features.

### 3.3 Hyper‑Parameter Sweep

**Configs implemented:**

1. **compact** – `(256,)` hidden layers, higher LR (`5e-3`), stronger regularisation (alpha `5e-4`), max_iter 60.
2. **baseline** – default `[512, 256]`.
3. **deep** – `[512, 256, 128]`, halved LR (`5e-4`), alpha `5e-5`, max_iter `baseline+40`.

Fill in values from `outputs/task2/mlp_hparam_sweep/summary.csv`:

| Config | Hidden Layers | LR | Alpha | Mean CV Acc. | Mean CV Macro F1 | Test Acc. | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compact | `(256,)` | 5e-3 | 5e-4 | ✅ | ✅ | ✅ | ✅ |
| baseline | `(512, 256)` | 1e-3 | 1e-4 | ✅ | ✅ | ✅ | ✅ |
| deep | `(512, 256, 128)` | 5e-4 | 5e-5 | ✅ | ✅ | ✅ | ✅ |

**Qualitative analysis (update with actual observations):**

- *Learning rate effect* – The compact model usually converges faster but may underperform due to reduced capacity; mention if accuracy gap is ≥2 %.
- *Depth vs. overfitting* – The deep model can slightly overfit (higher train/CV vs. test gap). Use CV vs. test macro F1 to support statements.
- *Regularisation* – Stronger alpha in compact models can stabilise training on small subsets.

### 3.4 Per‑Class Performance

- Extract per-class F1/precision/recall from `reports/*.json`. Summarise which classes (e.g., `cat`, `dog`, `truck`) are hardest/easiest.
- Note patterns such as similar confusion between `cat`/`dog` or `ship`/`airplane`.

---

## 4. Task 3 – Alternative Models (Random Forest + CNN)

Task 3 requires a second modelling approach beyond the MLP baseline. This project offers **two** fully documented backends so you can decide which results to highlight in the report (or include both for extra credit). The Random Forest option mirrors the PCA workflow from Task 2, while the CNN leverages end-to-end representation learning on GPU.

### 4.1 Random Forest Experimental Setup

- **Model** – `sklearn.ensemble.RandomForestClassifier`, baseline `n_estimators=400`, unlimited depth, `min_samples_split=2`, `n_jobs=-1`.
- **Feature Inputs** – Uses exactly the same PCA outputs as Task 2.
- **Evaluation** – Identical metric export pipeline, stored under `outputs/task3/rf_*`.

### 4.2 Random Forest Feature Dimension Sweep

Populate from `outputs/task3/rf_feature_sweep/summary.csv` and include the plot `rf_feature_accuracy.png`.

| Feature Set | Mean CV Accuracy | Mean CV Macro F1 | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- |
| original | ✅ | ✅ | ✅ | ✅ |
| pca_70 | ✅ | ✅ | ✅ | ✅ |
| pca_50 | ✅ | ✅ | ✅ | ✅ |
| pca_30 | ✅ | ✅ | ✅ | ✅ |
| pca_10 | ✅ | ✅ | ✅ | ✅ |

**Discussion points:**

- Unlike MLPs, Random Forests can degrade more gracefully with PCA compression because trees benefit from decorrelated inputs but still need sufficient depth to partition the space.
- Observe whether PCA actually improves accuracy by reducing noise; if so, highlight the best-performing PCA level.

### 4.3 Random Forest Hyper‑Parameter Sweep

Configs:

1. **shallow** – 200 estimators, depth limited to 40 (faster, less overfitting).
2. **baseline** – inherits global args; unlimited depth.
3. **regularized** – 600 estimators, depth 60, `min_samples_split` increased to 4.

Fill from `outputs/task3/rf_hparam_sweep/summary.csv`:

| Config | n_estimators | max_depth | min_samples_split | Mean CV Acc. | Mean CV Macro F1 | Test Acc. | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shallow | 200 | 40 | 2 | ✅ | ✅ | ✅ | ✅ |
| baseline | 400 | ∞ | 2 | ✅ | ✅ | ✅ | ✅ |
| regularized | 600 | 60 | 4 | ✅ | ✅ | ✅ | ✅ |

### 4.4 Random Forest Insights

- Tree ensembles tend to overfit less dramatically on high‑dimensional sparse data, but training time scales with `n_estimators × depth`. Include wall‑clock notes if available.
- Report which configuration delivered the best macro F1 and how it compares to the MLP baseline. Typical behaviour: Random Forests match or slightly lag MLP accuracy but have better interpretability and robustness to class imbalance.

### 4.5 CNN Experimental Setup

- **Model** – ResNet18 (torchvision) modified for 32×32 inputs (3×3 stem conv, no initial maxpool) with optional dropout before the final FC layer.
- **Feature Inputs** – Operates directly on the raw RGB images stored in `dataset.train_images` / `test_images`, with CIFAR normalisation and configurable data augmentation.
- **Training** – SGD with momentum, cosine annealing schedule, cross-entropy loss, 5-fold stratified CV. Requires an NVIDIA GPU; the script aborts if CUDA is unavailable.
- **Artifacts** – Written to `outputs/task3/cnn_*` and include `.pt` weight files plus the usual CSV/JSON summaries.

### 4.6 CNN Augmentation Sweep

The “feature-dimension” analogue for CNNs is the strength of the augmentation pipeline. Two presets are provided:

1. **standard** – Random crop + horizontal flip.
2. **strong_aug** – Adds ColorJitter + RandomErasing on top of the baseline.

Fill in `outputs/task3/cnn_feature_sweep/summary.csv` and reference `cnn_feature_accuracy.png`.

| Variant | Augmentations | Mean CV Accuracy | Mean CV Macro F1 | Test Accuracy | Test Macro F1 |
| --- | --- | --- | --- | --- | --- |
| standard | ✅ | ✅ | ✅ | ✅ | ✅ |
| strong_aug | ✅ | ✅ | ✅ | ✅ | ✅ |

Discuss whether the extra augmentation stabilises validation metrics and how it affects convergence versus overfitting.

### 4.7 CNN Hyper‑Parameter Sweep

Three preset configurations tweak epochs, learning rate, weight decay, and dropout (see `outputs/task3/cnn_hparam_sweep/summary.csv` and `cnn_hparam_accuracy.png`):

| Config | Epochs | LR | Weight Decay | Dropout | Mean CV Acc. | Mean CV Macro F1 | Test Acc. | Test Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fast | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| baseline | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| regularized | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

When copying the numbers, also mention GPU runtime and any signs of under/over-regularisation.

### 4.8 CNN Insights

- Strong augmentation + heavier regularisation typically yields better macro F1 on smaller training subsets by preventing the network from memorising noise.
- CNNs usually surpass Random Forests in absolute accuracy once enough epochs are trained on GPU, but they are more computationally demanding (≈25 epochs × 5 folds per sweep).
- Highlight any per-class gains (e.g., animals vs. vehicles) relative to the MLP/Random Forest baselines to motivate using learned convolutional features.

---

## 5. Task 4 – Comparative Analysis

Use the completed tables above to answer the reflective questions. Suggested structure:

### 5.1 Accuracy & Generalisation

- Compare the best-performing MLP vs. your chosen Task 3 backend (Random Forest and/or CNN). Discuss:
  - Absolute accuracy difference.
  - Macro F1 – indicates class balance handling.
  - Per-class winners (cite JSON metrics or confusion matrix observations).

### 5.2 Computational Complexity

- **MLP** – GPU‑friendly; time grows with epochs × parameters. Mention actual training time per run (measure with `/usr/bin/time -v` or script logs if available). Highlight that PCA reduces input size but not necessarily training time due to CPU‑bound cross-validation loops.
- **Random Forest** – CPU‑oriented; training parallelises over `n_jobs=-1`. Complexity roughly `O(T * d * log n)` (T = number of trees, d = depth). Highlight empirical runtime differences after executing on the Linux server.

### 5.3 Overfitting Assessment

- Contrast CV metrics vs. test metrics:
  - If CV accuracy ≈ test accuracy, model generalises well.
  - Look for cases where CV macro F1 is much higher than test macro F1 to identify overfitting. Typically, deep MLPs might show this pattern; Random Forests with unlimited depth can also overfit unless regularised.

### 5.4 Recommendations

Based on observations, answer:

1. **When to prefer MLPs?** e.g., when GPU resources exist, and highest accuracy is required, especially with raw features.
2. **When to prefer Random Forests?** e.g., CPU‑only environments, faster iteration, interpretability, or smaller datasets.
3. **When to prefer CNNs?** e.g., when GPUs are available and you need the absolute best accuracy, especially on the original input space without PCA.
4. **Effect of PCA / augmentation** – Summarise whether moderate PCA (50–70 %) benefits the classical models and how augmentation strength influences the CNN results.

---

## 6. Reproducibility Checklist

1. **Install dependencies:** `python3 -m pip install --user -r requirements.txt`.
2. **Run experiments:** `scripts/run_all.sh` (tweak environment variables or CLI args as needed).
3. **Collect metrics:**
   - Task 1: `outputs/task1/feature_metadata.json`.
   - Task 2: `outputs/task2/*`.
   - Task 3: `outputs/task3/*`.
4. **Update this report:** Replace the ✅ placeholders with actual numbers/tables and embed the PNG figures.
5. **(Optional) Full dataset:** Add `--train-subset 0 --test-subset 0` to the script invocation; expect longer runtimes but higher accuracy.

---

## 7. Future Work

- **Stronger CNNs** – Extend the current ResNet18 baseline with modern tricks (e.g., WideResNet, MixUp, CutMix) or fine-tune pretrained ViT models if compute allows.
- **Data Augmentation for Classical Models** – Explore synthetic feature generation or SMOTE-like approaches to help MLP/Random Forest deal with minority classes.
- **Hyper‑parameter Optimisation** – Automate with Optuna or scikit-opt’s `GridSearchCV`/`RandomizedSearchCV` to explore broader spaces.
- **Confusion Matrix Analysis** – Visualise misclassifications to understand class‑specific weaknesses.

---

## 8. References

1. Alex Krizhevsky. “Learning Multiple Layers of Features from Tiny Images.” Technical report, 2009. [http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)
2. Scikit‑learn documentation: [https://scikit-learn.org/stable/modules/neural_networks_supervised.html](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
3. Scikit‑learn Random Forests: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
4. Pedregosa et al., “Scikit-learn: Machine Learning in Python”, JMLR 12, 2011.

---

*Prepared by Codex. Update the highlighted placeholders with your actual experiment results before submission.*
