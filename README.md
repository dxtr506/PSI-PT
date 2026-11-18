# Post-Selection Inference in High-Dimensional Parameter Transfer with Fused-Regularizer (PSI-PT)
This package contains the experimental implementation for the paper "Post-Selection Inference in High-Dimensional Parameter Transfer with Fused-Regularizer".
PSI-PT computes valid selective $p$-values for the features selected by TransMission, a parameter-transfer Lasso with a fused regularizer whose tuning parameters are chosen by cross-validation. The core method is in the package `psi_pt/`: `transmission.py` and `practical_transmission.py` (the estimator and its cross-validation pipeline), `inference.py` (the truncation region and the selective $p$-values of PSI-PT), `sub_prob.py` and `utils.py` (helpers).

## Reproducibility
The synthetic experiment results stored under `results/` can be reproduced from the repository root with the scripts below. Every script runs the experiments and then draws the figures. The experiments use $p = 500$ features, $K = 4$ source tasks of size 200 and target sizes $n_0 = 40, 50, 60, 70$; the seeds are fixed in the scripts.
The scripts accept two environment variables: `PY` (Python executable, default `python`) and `N_JOBS` (processes used for one $p$-value, default 4).

### Environment
Create a fresh Python environment and install the pinned dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Synthetic FPR
Run:
```powershell
.\experiments\run_fpr.bat
```
This reproduces the false-positive-rate experiments (PSI-PT, PSI-PT-oc, data splitting, Bonferroni and the naive test).
Outputs are written to:
```text
results/fpr/
```
The final figures are:
```text
results/fpr.pdf
results/fpr.png
```

### Synthetic TPR
Run:
```powershell
.\experiments\run_tpr.bat
```
This reproduces the true-positive-rate experiments, against the target size $n_0$ and against the signal strength $\delta$.
Outputs are written to:
```text
results/tpr/
```
The final figures are:
```text
results/tpr_nt.pdf
results/tpr_nt.png
```

### Computational Time
Run (Linux, or Git Bash on Windows):
```bash
bash experiments/run_timing.sh
```
This measures the time of one $p$-value against the number of workers and against the number of features $p$.
Outputs are written to:
```text
results/timing/
```
The final figures are:
```text
results/timing_workers_median.pdf
results/timing_workers_median.png
results/timing_p_median.pdf
results/timing_p_median.png
```

## Example Notebooks
We provide Jupyter Notebooks that show how to use the code in the `/examples` directory:

- `ex1_pvalue.ipynb`: Example for computing $p$-values.
- `ex2_pivot.ipynb`: Check the uniformity of the pivot.
