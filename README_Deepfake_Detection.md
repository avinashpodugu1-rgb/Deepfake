# Universal AI-Generated / Deepfake Image Detector

A research-oriented binary image detector based on the paper:

> **Towards Universal Fake Image Detectors that Generalize Across Generative Models**  
> Utkarsh Ojha, Yuheng Li, Yong Jae Lee — CVPR 2023

This repository/notebook implements the paper's central **frozen CLIP feature-space** detection paradigm and extends it with complementary image-forensic features for experimentation on an AI-generated-vs-real image dataset.

---

## 1. Project Overview

The goal of this project is to classify an input image as:

- **REAL** — naturally captured/authentic image
- **FAKE / AI-GENERATED** — synthetic or manipulated image

The central research idea comes from Ojha et al. (CVPR 2023): instead of fine-tuning a conventional real/fake CNN classifier, use a **pretrained CLIP visual representation as a fixed feature space** and perform classification using methods such as nearest-neighbor search and linear probing.

The notebook then evaluates whether additional forensic information can complement the pretrained CLIP representation.

### Main pipeline

```text
                         INPUT IMAGE
                              |
                    ---------------------
                    |                   |
                    v                   v
              Frozen CLIP          Forensic Branch
              Visual Features      Frequency / Texture
                    |              Residual / Color
                    |              Edge / Recompression
                    |                   |
                    -----------+---------
                               |
                               v
                         Feature Fusion
                               |
                               v
                         Linear Probe
                               |
                         -------------
                         |           |
                         v           v
                       REAL        FAKE
```

---

## 2. Research Paper

### Primary Reference

**Ojha, U., Li, Y., & Lee, Y. J.**

**"Towards Universal Fake Image Detectors that Generalize Across Generative Models."**

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

The paper investigates whether pretrained visual representations can provide a more generalizable basis for fake-image detection than classifiers trained specifically on a particular set of generators.

The paper's central implementation used a **frozen CLIP visual encoder**, followed by classification in the resulting feature space.

### Research motivation

A detector trained only on one distribution of fake images can learn generator-specific fingerprints. Such a detector may perform well on familiar fake images but degrade when the image comes from an unseen generative model.

This project therefore separates:

1. **Paper-faithful frozen CLIP experiments**
2. **Additional forensic-feature experiments**
3. **Hybrid feature-fusion experiments**
4. **Robustness experiments**

---

## 3. Dataset

Dataset used for this project:

**FDS Dataset Deepfake**

Kaggle:
https://www.kaggle.com/datasets/rohithsaikambhampati/ai-generated-vs-real-images-dataset

The intended dataset organization is:

```text
FDS Deepfake Dataset V-1/
└── Images Dataset/
    ├── train/
    │   ├── FAKE/
    │   └── REAL/
    ├── valid/
    │   ├── FAKE/
    │   └── REAL/
    └── test/
        ├── FAKE/
        └── REAL/
```

### Important dataset note

The currently provided reference notebook contains a **leakage-aware dataset discovery and split stage**. It builds a dataframe from the available image directories, removes exact duplicates, and creates stratified train/validation/test partitions.

If your final experiment must use the already supplied `train/valid/test` directories exactly as fixed partitions, the split-generation cell should be disabled and the existing partitions should be loaded directly.

Do not use the test partition for model selection or threshold tuning.

---

## 4. Folder Structure for This Project

A recommended repository structure is:

```text
deepfake-image-detector/
│
├── FDS_Deepfake_Ojha_Upgraded_LayerSpatial_Prototype_Notebook.ipynb
├── README.md
│
├── deepfake_universal_artifacts/
│   ├── clip_linear_probe.joblib
│   ├── clip_forensic_hybrid_probe.joblib
│   ├── forensic_only_probe.joblib
│   ├── forensic_scaler.joblib
│   ├── clip_threshold.npy
│   ├── hybrid_threshold.npy
│   └── config.json
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── robustness_results.csv
```

The artifact/result directories are created after running the corresponding notebook cells.

---

## 5. Models and Experiments

The notebook is organized as a research experiment rather than a single classifier.

### Experiment A — Frozen CLIP + Nearest Neighbor

```text
Image
  |
Frozen CLIP
  |
Feature Vector
  |
Cosine Nearest Neighbor
  |
REAL / FAKE
```

Several values of `k` are evaluated.

The CLIP encoder is not trained.

---

### Experiment B — Frozen CLIP + Linear Probe

```text
Image
  |
Frozen CLIP
  |
Feature Vector
  |
Logistic Regression
  |
REAL / FAKE
```

Only the classifier is trained.

The CLIP encoder remains frozen.

---

### Experiment C — Forensic-Feature Baseline

The notebook extracts complementary low-level features including:

#### Frequency features
- FFT radial-spectrum statistics
- DCT low-frequency energy
- DCT mid-frequency energy
- DCT high-frequency energy

#### Residual features
- High-pass residual statistics
- Laplacian variance
- Absolute residual statistics

#### Texture features
- Local Binary Pattern (LBP)
- Gray-Level Co-occurrence Matrix (GLCM)

#### Image statistics
- RGB statistics
- HSV statistics
- Gradient magnitude
- Edge density

#### Recompression response
- JPEG/recompression-related image statistics

These features are evaluated independently as an additional baseline.

---

### Experiment D — Proposed Hybrid Feature Model

The hybrid model combines:

```text
Frozen CLIP features
        +
Forensic features
        |
        v
Standardization
        |
        v
Weighted feature fusion
        |
        v
Logistic Regression
        |
        v
REAL / FAKE
```

This allows the project to test whether low-level forensic information provides useful complementary information beyond the pretrained CLIP representation.

---

### Experiment E — Robustness Testing

The notebook evaluates predictions under controlled image modifications such as:

- JPEG compression
- Image blur
- Resizing
- Other controlled corruption/perturbation conditions implemented in the notebook

The objective is to determine how sensitive the detector is to common image transformations.

---

### Experiment F — Exact-Paper Backbone

For a closer reproduction of the primary research paper, the notebook supports:

```text
openai/clip-vit-large-patch14
```

The notebook also provides a smaller CLIP configuration for development when GPU memory is limited.

For a final research comparison, clearly report which CLIP backbone was actually used.

---

## 6. Evaluation Metrics

The notebook reports multiple metrics instead of relying only on accuracy.

### Accuracy

Percentage of correctly classified images.

### Precision

Measures how many images predicted as fake are actually fake.

### Recall

Measures how many actual fake images are detected.

### F1-score

Harmonic mean of precision and recall.

### ROC-AUC

Measures ranking/discrimination performance across classification thresholds.

### Average Precision (AP)

Summarizes precision-recall performance and is useful when class distributions are not perfectly balanced.

### Confusion Matrix

Shows:

```text
                 Predicted
               REAL     FAKE
Actual REAL     TN       FP
Actual FAKE     FN       TP
```

---

## 7. Validation Threshold Calibration

The notebook does not have to assume that `0.50` is the optimal classification threshold.

A threshold is selected using the **validation set**.

The selected threshold is then applied to the held-out test set.

This avoids using the test labels to tune the classification decision boundary.

---

## 8. Data Leakage Prevention

Deepfake datasets can contain duplicates or highly similar images.

The notebook includes an audit stage intended to reduce leakage risk through duplicate checking.

The project should follow this rule:

```text
TRAIN
  |
  +--> model fitting

VALIDATION
  |
  +--> threshold/model selection

TEST
  |
  +--> final evaluation only
```

The test set should not be repeatedly inspected to choose models or hyperparameters.

---

## 9. Visual and Research Analysis

The notebook includes several analysis components.

### Dataset audit

- Number of images
- Class distribution
- Image-readability checks
- Image examples

### Feature-space visualization

t-SNE is used to visualize the frozen CLIP feature space and inspect whether REAL and FAKE samples form distinguishable structures.

### ROC curve

Compares discrimination performance of the evaluated models.

### Precision-Recall curve

Shows the precision/recall trade-off.

### Confusion matrices

Allows inspection of false positives and false negatives.

### Robustness analysis

Measures performance after controlled image transformations.

### Ablation study

The notebook compares:

1. Frozen CLIP + NN
2. Frozen CLIP + linear probe
3. Forensic-only features
4. CLIP + forensic features

This provides evidence about the contribution of each component.

---

## 10. Installation

### Google Colab

Open the `.ipynb` file in Google Colab and run the installation cell.

The notebook installs the required packages, including the CLIP/transformer dependencies and image-processing libraries.

A GPU runtime is strongly recommended.

In Colab:

```text
Runtime
  ->
Change runtime type
  ->
GPU
```

---

### Kaggle

Open the notebook in Kaggle and attach the dataset.

Recommended:

```text
Notebook Settings
  ->
Accelerator
  ->
GPU
```

The notebook can use KaggleHub where appropriate for dataset access.

---

## 11. Main Python Libraries

The notebook uses libraries including:

- Python
- PyTorch
- Transformers
- Hugging Face CLIP
- scikit-learn
- OpenCV
- scikit-image
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- imagehash
- joblib
- KaggleHub

---

## 12. Reproducibility

A fixed random seed is used in the notebook where applicable.

For reproducible experiments, record:

- Dataset version
- Dataset split
- CLIP model ID
- CLIP embedding dimension
- Random seed
- Number of training samples
- Number of validation samples
- Number of test samples
- Feature-fusion weight
- Classification threshold
- Hardware/GPU
- Software/library versions

The notebook saves a `config.json` file containing important model configuration values.

---

## 13. Saved Model Artifacts

After execution, the notebook saves models and preprocessing artifacts.

Examples include:

```text
clip_linear_probe.joblib
clip_forensic_hybrid_probe.joblib
forensic_only_probe.joblib
forensic_scaler.joblib
clip_threshold.npy
hybrid_threshold.npy
config.json
```

These files allow the trained classifiers and preprocessing configuration to be reused without retraining the entire pipeline.

---

## 14. Single-Image Prediction

The notebook includes a single-image prediction function.

Conceptually:

```text
New Image
   |
Preprocessing
   |
Frozen CLIP Features
   +
Forensic Features
   |
Trained Classifier
   |
Prediction Probability
   |
REAL / FAKE
```

The predicted class should be interpreted together with the model probability/confidence and the limitations of the dataset.

---

## 15. Research Contributions of This Implementation

The project has two clearly separated parts.

### Paper-derived component

The implementation follows the research direction of Ojha et al.:

- Frozen pretrained visual representation
- CLIP visual encoder
- Feature-space classification
- Cosine nearest-neighbor detection
- Linear probing
- Investigation of generalization
- Feature-space visualization

### Project extension

The notebook additionally evaluates:

- FFT-based frequency features
- DCT features
- High-pass residual statistics
- LBP texture features
- GLCM/co-occurrence features
- Color statistics
- Gradient/edge statistics
- Recompression response
- CLIP + forensic feature fusion
- Robustness experiments
- Ablation analysis

The extension should be presented as an **experimental enhancement**, not as a claim that the CVPR paper proposed these exact combined features.

---

## 16. Research Questions

This project can be framed around the following questions:

### RQ1
How effective is a frozen CLIP feature space for distinguishing REAL and AI-GENERATED images on the selected dataset?

### RQ2
How does frozen CLIP linear probing compare with cosine nearest-neighbor classification?

### RQ3
Do low-level forensic features provide complementary information to pretrained CLIP features?

### RQ4
How robust are the models to common image transformations such as compression, blur and resizing?

### RQ5
Which feature representation produces the most useful separation between REAL and FAKE images?

---

## 17. Important Scientific Limitations

This project should not claim that it is a universal detector for every future generative model.

A dataset-level REAL/FAKE test does not automatically prove cross-generator generalization.

A true unseen-generator experiment requires generator identity information and a test protocol in which the generator family is not represented in training.

Therefore, results from this dataset should be described as performance **on the evaluated dataset and its experimental split**.

Do not claim:

> "The model detects every deepfake."

Instead use language such as:

> "The model classifies REAL and AI-GENERATED images on the evaluated dataset."

For a research paper, also report the exact dataset, split, model backbone, preprocessing, and evaluation protocol.

---

## 18. Accuracy Target

The project may use **75% accuracy as a target**, but the notebook must report the actual measured result.

Do not manually alter thresholds or results simply to reach 75%.

A valid research result should report:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
Average Precision
Confusion Matrix
```

If the model achieves less than the target, that result should be analyzed rather than hidden.

---

## 19. Recommended Final Results Table

After running the notebook, populate a table like:

| Experiment | Encoder | Features | AP | Accuracy | F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| CLIP + NN k=1 | Frozen CLIP | CLIP | — | — | — | — |
| CLIP + NN k=9 | Frozen CLIP | CLIP | — | — | — | — |
| CLIP Linear Probe | Frozen CLIP | CLIP | — | — | — | — |
| Forensic-only | None | Forensic | — | — | — | — |
| Proposed Hybrid | Frozen CLIP | CLIP + Forensic | — | — | — | — |

Replace the dashes with the values obtained from the actual notebook run.

---

## 20. How to Run

### Step 1

Open:

```text
FDS_Deepfake_Ojha_Upgraded_LayerSpatial_Prototype_Notebook.ipynb
```

### Step 2

Attach the dataset.

### Step 3

Enable GPU.

### Step 4

Run the installation/import cells.

### Step 5

Run dataset discovery and audit.

### Step 6

Run the frozen CLIP baseline.

### Step 7

Run the forensic feature branch.

### Step 8

Run feature fusion.

### Step 9

Run validation threshold calibration.

### Step 10

Run final evaluation.

### Step 11

Run robustness and ablation experiments.

### Step 12

Save the models and artifacts.

---

## 21. Expected Notebook Output

The notebook is designed to produce:

- Dataset statistics
- Image samples
- Feature dimensions
- CLIP baseline results
- Nearest-neighbor results
- Linear-probe results
- Forensic-only results
- Hybrid results
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion matrices
- ROC curves
- Precision-recall curves
- t-SNE visualization
- Robustness results
- Ablation results
- Saved model files
- Single-image prediction capability

---

## 22. Citation

If you use the research methodology in an academic report, cite the primary paper:

```bibtex
@inproceedings{ojha2023universal,
  title={Towards Universal Fake Image Detectors That Generalize Across Generative Models},
  author={Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

Dataset:

```text
FDS Dataset Deepfake
https://www.kaggle.com/datasets/avinashpodugu/fds-dataset-deepfake
```

---

## 23. Project Status

**Status:** Research/academic prototype

**Task:** Binary REAL vs AI-GENERATED/FAKE image classification

**Primary research basis:** Ojha et al., CVPR 2023

**Primary representation:** Frozen CLIP visual features

**Additional experimental representation:** Image-forensic features

**Classifier:** Nearest Neighbor / Logistic Regression

**Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision, confusion matrix, robustness and ablation analysis

---

## 24. Author

**Avinash Podugu**

B.Tech Computer Science and Engineering

This project was developed as an academic/research implementation for deepfake and AI-generated image detection.
