# Dataset Description

## Access

The full dataset is archived on Zenodo:

```text
DOI: 10.5281/zenodo.20375590
URL: https://doi.org/10.5281/zenodo.20375590
```

Download and extract the dataset into the repository root before running the scripts.

## Overview

The dataset contains eye point-cloud samples used in the RF-PCL experiments. The raw point-cloud files are provided as:

```text
data_raw/eye_01.txt ... data_raw/eye_47.txt
```

Each file stores one point cloud in plain text format. The scripts assume at least three columns corresponding to `x`, `y`, and `z` coordinates.

## Expected Directories

After extracting the Zenodo artifact, the repository should contain:

```text
data_raw/      Raw point-cloud files
data_2048/     Sampled/processed point-cloud files used by selected experiments
recon/         Reconstruction examples and split outputs
result/        Experiment metrics, averaged reconstructions, and logs
```

The trained model records are described separately in `docs/model_record_description.md`.

## Privacy and Release Notes

Before redistribution or reuse, verify that:

- The point clouds contain no personal identifiers.
- The data source and consent/authorization status allow the intended use.
- The dataset license in `DATASET_LICENSE` is followed.
