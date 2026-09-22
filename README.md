# Breast Cancer Diagnosis — UM5BM753 Project (Python component)

Predicting malignancy from cell nucleus measurements, with supervised and unsupervised learning. Project for UE 5BM753, Supervised and Unsupervised Learning for Biology (M2 AIDA, Sorbonne University).

This repository holds the Python component of a two-part project. A complementary
Orange3 component was also produced, outside this repository.

## Introduction

### Biological context

Breast cancer arises when mammary epithelial cells acquire alterations that disrupt
the normal control of proliferation, survival, genome maintenance, and tissue
organization. These changes also modify the phenotype of the cell, and in particular
the appearance of its nucleus. Malignant cells typically show enlarged, irregular
nuclei, more variation in nuclear size and shape, and disrupted chromatin patterns.
This nuclear atypia is a well-established cytological sign of malignancy, already used
in pathological assessment [1, 2].

The **Breast Cancer Wisconsin (Diagnostic)** dataset quantifies this signal. It
contains 569 cases from digitized images of fine-needle aspirate (FNA) biopsies of
breast masses. For each case, the contour and texture of the cell nuclei were
converted into 10 base measurements (radius, texture, perimeter, area, smoothness,
compactness, concavity, concave points, symmetry, fractal dimension), each summarized by its mean, standard error, and "worst" value (the average of the 3 most extreme cells), giving 30 continuous features [3, 4]. Larger, more irregular, or more
heterogeneous nuclei are plausible indicators of malignancy, but these features remain
image-derived phenotypic markers, not a direct readout of a specific mutation or tumor subtype.

### The questions this project asks

The goal is not to replace the pathologist, but to test whether these image-derived
measurements can support a second reading:

- **Q1 (supervised):** Can the 30 nuclear features reproduce the known diagnosis, and
  can a reasonably interpretable model identify which measurements contribute most to that distinction?
- **Q2 (unsupervised):** Without ever using the diagnostic labels, do the samples
  organize naturally into groups that reflect the benign/malignant divide?
- **Q3 (feature relevance):** Many size-related measurements are strongly correlated.
  Reducing this redundancy can reveal which dimensions of nuclear phenotype carry
  distinct diagnostic information.

Recall and the F2-score (which weighs recall more than precision) get particular
attention throughout, since missing a malignant case is far more costly than flagging
a benign one for review. The dataset is also small, historical, and single-site, so
any result here would need external validation before real clinical use.

## Project structure

Structure aligned with the course's AIDA methodology ("Working Effectively on an AIDA
Project"):

- `01_scripts/`: reusable Python functions, imported by the notebooks.
- `02_data/`: data files.
  - `raw/`: the original, unmodified source file.
  - `processed/`: the train/test sets restricted to the selected features, exported by
    `04_Feature_Selection.ipynb` and reused by the two "zoom" notebooks.
- `03_notebooks/`: the 7 notebooks of the final analysis, numbered in execution order:
  1. `01_Data_QC.ipynb`: loads the raw data and checks it (missing values,
     duplicates, outliers).
  2. `02_EDA.ipynb`: first visual look at the data (class balance, correlation,
     boxplots, PCA).
  3. `03_Initial_Screening.ipynb`: screens 10 supervised classifiers and 6
     clustering methods, to shortlist one of each.
  4. `04_Feature_Selection.ipynb`: reduces the 30 features to a small,
     non-redundant subset, reused by the notebooks below.
  5. `05_Zoom_LogReg.ipynb`: tunes and evaluates the final Logistic Regression
     model.
  6. `06_Zoom_Spectral_Clustering.ipynb`: tunes and evaluates the final Spectral
     Clustering model.
  7. `07_Prognostic_Data_Test.ipynb`: checks whether the tuned model still
     predicts malignant on an external cohort (WPBC).
- `04_results/`: exported tables (`.csv`).
- `05_figures/`: exported figures.
- `06_report/`: written deliverables (slide exports, notes).

## How to run

1. Create and activate the project's virtual environment. Two options are provided,
   depending on which tool you use:
   - **uv** (`requirements.txt`, pip-based):
     ```
     uv venv --python 3.11 .venv
     uv pip install --python .venv -r requirements.txt
     ```
   - **conda** (`environment.yml`):
     ```
     conda env create -f environment.yml
     conda activate breast-cancer-diagnosis
     ```

   Both install the exact same packages: `environment.yml` only sets up Python 3.11
   and pip, then installs everything from `requirements.txt`, so there is a single
   source of truth for package versions.
2. Run the notebooks in `03_notebooks/` in numbered order.
3. Outputs are written to `04_results/` (tables) and `05_figures/` (figures).

## Deliverables

The Python component's deliverable is the 6 notebooks above, plus an oral
presentation (10 minutes, followed by 10 minutes of questions).

## Team

Ann-Solène Adelson, Anaïs Hluszko, Corentin Osta.

## References

1. National Cancer Institute. *How Breast Cancer Is Diagnosed*.
   https://www.cancer.gov/types/breast/diagnosis
2. Bussolati, G., Marchiò, C., Gaetano, L., Lupo, R., & Sapino, A. (2008). Pleomorphism
   of the nuclear envelope in breast cancer: a new approach to an old problem. *Journal
   of Cellular and Molecular Medicine*, 12(1), 209–218.
   https://doi.org/10.1111/j.1582-4934.2007.00176.x
3. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature
   extraction for breast tumor diagnosis. *Proceedings of SPIE*, 1905, 861–870.
4. UCI Machine Learning Repository. *Breast Cancer Wisconsin (Diagnostic)*.
   https://doi.org/10.24432/C5DW2B
