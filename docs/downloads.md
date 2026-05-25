# Downloading Data and Model Records

The full RF-PCL dataset and trained model records are hosted on Zenodo:

```text
DOI: 10.5281/zenodo.20375590
URL: https://doi.org/10.5281/zenodo.20375590
```

## What to Download

Download the archive files that contain:

```text
data_raw/
data_2048/
model_record/
recon/
result/
```

The exact archive filenames may depend on the Zenodo upload, but after extraction the directory names above should be present.

## Where to Extract

Extract the downloaded files into the `RF-PCL/` repository root:

```text
RF-PCL/
|-- scripts/
|-- docs/
|-- data_raw/
|-- data_2048/
|-- model_record/
|-- recon/
`-- result/
```

## Required Checks

After extraction, verify:

```text
data_raw/eye_01.txt
data_raw/eye_47.txt
model_record/eye_01/
model_record/eye_47/
```

## Notes

The GitHub repository is intended for source code, documentation, and lightweight metadata. Large datasets and trained checkpoints should be obtained from Zenodo through the DOI above.
