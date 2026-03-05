---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# RIVeR: Rectification of Image Velocimetry Results

RIVeR (Rectification of Image Velocimetry Results) is a Python package designed for processing river flow videos to obtain velocity fields and discharge estimates. It supports three main filming scenarios:

**RIVeR-ICE: Rectification of Image Velocimetry Results - Integrated Channel Evolution**

RIVeR-ICE (Rectification of Image Velocimetry Results - Integrated Channel Evolution) is an extension of RIVeR that....

**Prerequisites**

Before starting, ensure you have:

- Python 3.11 or later
- RIVeR package installed
- Required dependencies (numpy, opencv-python, scipy)

**Required folder hierarchy**

...

**Required file terminology**

...


Make sure your environment is set up in this shell
source path/to/setup.sh

## extract_meta.py

**Run (overwrites CSVs by default)**

`extract_meta`

**Only process one camera**

`extract_meta --camera CAM1`

**Append instead of overwrite**

`extract_meta --append`

**Limit to specific extensions**

`extract_meta --ext .mp4 .mkv`


## extract_frames.py
**Execute with your defaults (no filters, every 20, no overwrite)**

`extract_frames`

**Only one camera**

`extract_frames --camera CAM1`

**Overwrite destination folders before extraction**

`extract_frames --overwrite`

**Use different extensions**

`extract_frames --ext .mp4 .avi`

```{code-cell} ipython3

```

```{code-cell} ipython3

```
