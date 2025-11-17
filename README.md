# Steel Ingot Defect Detection via Machine Learning

This repository implements the full pipeline described in *Data-Driven Approach for Defect Identification in Steel Ingot Casting via Machine Learning*. It provides a modular Python codebase and a detailed Jupyter notebook that reproduces the exploratory analysis, model training, ensemble optimization, and explainability workflow presented in the paper.

## Dataset
- **Location:** `data/Steel.csv`
- **Content:** ~2.3k rows covering 17+ numeric process parameters (alloying elements, temperatures, casting speeds, ladle life, etc.) and a binary `Defect` column indicating defective vs. non-defective ingots.
- **Assumptions:** The dataset is clean (no missing values) and already lives in the `data/` directory. If not, place it there before running the notebook.

## Project Structure
```
.
├── data/Steel.csv
├── src/                # Modular pipeline implementation
├── notebooks/          # Reproducible research notebooks
├── tests/              # Optional lightweight tests
├── requirements.txt
└── README.md
```

Key source modules include configuration (`src/config.py`), data ingestion (`src/data_loading.py`), preprocessing (`src/preprocessing.py`), exploratory visuals (`src/eda.py`), model builders (`src/models.py`), training + ensemble tuning (`src/training.py`), evaluation utilities (`src/evaluation.py`), and explainability helpers (`src/explainability.py`).

## Setup
1. Create and activate a virtual environment (recommended).
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\\Scripts\\activate   # Windows PowerShell
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `data/Steel.csv` exists.

## Running the Full Pipeline
The notebook `notebooks/01_steel_defect_full_pipeline.ipynb` walks through:
- Dataset overview, density plots, and correlation heatmaps.
- Stratified splitting + scaling.
- Model training for Random Forest, XGBoost, RBF-SVM, and MLP.
- Ensemble optimization over class weights and decision thresholds.
- Detailed evaluation (confusion matrices, ROC, Precision–Recall, metric comparisons).
- SHAP global interpretability and linear SVM decision boundary analysis.

Open the notebook with Jupyter and execute all cells top-to-bottom:
```bash
jupyter notebook notebooks/01_steel_defect_full_pipeline.ipynb
```

The notebook will generate publication-ready figures showing density overlaps, correlation structure, ROC/PR curves, model comparison bars, SHAP summary and bar plots, and the explicit linear SVM decision function with coefficient-based importance.

## Reusing the Source Modules
You can also import the modules directly, for example:
```python
from src.data_loading import load_dataset
from src.preprocessing import split_and_scale
from src.training import train_and_optimize

 df, target, features = load_dataset()
 preprocessed = split_and_scale(df, features, target)
 artifacts = train_and_optimize(preprocessed)
```
This returns trained models, ensemble metrics, and helper structures for downstream analysis or integration into production pipelines.

## Future Work
Potential extensions include cross-validation, additional ensembles (stacking or weighted voting), and deployment hooks for real-time monitoring of casting operations.
