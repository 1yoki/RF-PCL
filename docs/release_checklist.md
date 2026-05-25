# Public Release Checklist

Use this checklist before making the GitHub repository public.

## Metadata

- Replace TODO fields in `README.md`.
- Replace TODO fields in `CITATION.cff`.
- Add the final paper title, author list, venue, and paper DOI if available.
- Confirm that the Zenodo DOI is listed as `10.5281/zenodo.20375590`.

## Licenses

- Replace `LICENSE` with the final code license.
- Replace `DATASET_LICENSE` with the final dataset license.
- Confirm that all authors and rights holders approve the selected licenses.

## Zenodo Artifact

- Confirm that the complete dataset is available at `https://doi.org/10.5281/zenodo.20375590`.
- Confirm that the complete `model_record/` directory is available at the same DOI.
- Confirm that the archive extracts into the paths described in `docs/downloads.md`.
- Confirm that mapping records are included:
  - `data_raw/match.txt`
  - `data_raw/rename_record.txt`
  - `model_record/rename_record_model_record.txt`

## GitHub Repository

- Keep GitHub focused on source code, documentation, and lightweight metadata.
- Do not commit the complete `model_record/` directory to ordinary Git history.
- Do not commit large dataset archives to ordinary Git history.
- If large files must be mirrored on GitHub, use Git LFS intentionally.

## Reproducibility

- Run `python -m py_compile scripts/*.py` or equivalent checks.
- Verify that commands in `docs/reproduction.md` match the final script names and paths.
- Run at least one small smoke test for training/evaluation if possible.
