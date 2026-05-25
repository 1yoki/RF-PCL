# Reproduction Guide

Run commands from the `RF-PCL/` repository root.

## 1. Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Use a PyTorch build compatible with your CUDA environment. The training scripts use CUDA by default.

## 2. Download the Artifact

Download the dataset and trained model records from Zenodo:

```text
DOI: 10.5281/zenodo.20375590
URL: https://doi.org/10.5281/zenodo.20375590
```

Extract the downloaded archive into the repository root.

## 3. Verify Paths

After extraction, verify that these paths exist:

```text
data_raw/eye_01.txt ... data_raw/eye_47.txt
data_2048/
model_record/
recon/
result/
```

## 4. Training

Main training scripts:

```bash
python scripts/RFG.py
python scripts/LRR.py
```

Check script-level configuration values before running, especially dataset paths, model output paths, `k`, latent dimension, and training epochs.

If you do not need to retrain, use the archived `model_record/` directory from Zenodo.

## 5. Splitting Point Clouds

Use:

```bash
python scripts/split.py
```

This generates fixed-boundary split metadata and optional train/test subsets based on the configured reference point cloud.

## 6. Evaluation

Chamfer distance:

```bash
python scripts/CD_caculate.py
```

Earth mover's distance:

```bash
python scripts/EMD_caculate.py
```

Both scripts should be checked for input/output paths before running.

## 7. Tables and Plots

Generate per-sample statistics and sensitivity plots:

```bash
python scripts/statics_one.py
```

The script reads an `experiment_errors.csv` file and writes table-ready outputs and figures.

## 8. Expected Outputs

Representative outputs are restored from Zenodo under:

```text
model_record/
recon/
result/
```
