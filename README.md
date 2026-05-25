# RF-PCL

This repository contains the source code and documentation for the RF-PCL paper artifact.

The full experiment dataset and trained model records are archived on Zenodo:

```text
DOI: 10.5281/zenodo.20375590
URL: https://doi.org/10.5281/zenodo.20375590
```

## Repository Layout

```text
RF-PCL/
|-- scripts/              # Training, evaluation, splitting, and statistics scripts
|-- docs/                 # Dataset, model, and reproduction documentation
|-- src/rfpcl/            # Reserved package namespace for reusable library code
|-- requirements.txt      # Python dependencies
|-- environment.yml       # Optional conda environment
|-- CITATION.cff          # Citation metadata
|-- LICENSE               # Code license placeholder
`-- DATASET_LICENSE       # Dataset license placeholder
```

After downloading the Zenodo archive, the full artifact should be restored as:

```text
RF-PCL/
|-- data_raw/             # Raw eye point-cloud data
|-- data_2048/            # Sampled/processed point-cloud data
|-- model_record/         # Trained checkpoints and training records
|-- recon/                # Reconstruction examples and split outputs
`-- result/               # Quantitative results and logs
```

## Installation

Create a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The experiments were developed with PyTorch and CUDA. Make sure the installed PyTorch build matches your GPU/CUDA environment.

## Download Data and Model Records

Download the dataset and `model_record` from Zenodo:

```text
https://doi.org/10.5281/zenodo.20375590
```

Extract the downloaded files into the repository root so that paths such as `data_raw/eye_01.txt` and `model_record/eye_01/` are available.

See `docs/downloads.md` for details.

## Main Scripts

Run scripts from the repository root:

```bash
python scripts/RFG.py
python scripts/LRR.py
python scripts/split.py
python scripts/CD_caculate.py
python scripts/EMD_caculate.py
python scripts/statics_one.py
```

The scripts currently contain experiment-specific paths. Check the configuration section at the top of each script before running.

## Reproduction

See `docs/reproduction.md` for the recommended workflow:

1. Prepare the environment.
2. Download and extract the Zenodo artifact.
3. Verify dataset and model-record paths.
4. Train RF-PCL models or use the archived model records.
5. Run reconstruction evaluation.
6. Generate tables and sensitivity plots.

In the main experiment workflow, the train/test split is performed inside
`scripts/LRR.py` after the ocular prosthesis point cloud is loaded. The split is
therefore generated during the experiment run rather than being stored as a
separate pre-split dataset.

## Citation

If you use this repository, please cite both the paper and the Zenodo artifact:

```text
10.5281/zenodo.20375590
```

Update `CITATION.cff` after the paper metadata is finalized.

## License

Code and data licenses are separated:

- `LICENSE`: code license placeholder
- `DATASET_LICENSE`: dataset license placeholder

Please finalize both files before making the repository public.
