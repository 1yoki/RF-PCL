# Model Record Description

## Access

The complete `model_record/` directory is archived on Zenodo:

```text
DOI: 10.5281/zenodo.20375590
URL: https://doi.org/10.5281/zenodo.20375590
```

Download and extract the archive into the repository root before using pretrained model records.

## Expected Layout

After extraction, the model records should be available as:

```text
model_record/
|-- eye_01/
|-- eye_02/
|-- ...
`-- eye_47/
```

Each `eye_xx/` directory contains trained checkpoints, loss curves, and logs for the corresponding point-cloud sample.

## GitHub Policy

The complete `model_record/` directory is large and should not be committed to ordinary Git history. Use the Zenodo DOI above for downloading and citing the archived model records.
