# Grain detection in XRF scans with StarDist

## Installation

* Install Anaconda (or the smaller MiniConda): [Windows](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/windows.html) | [MacOS](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html) | [Linux](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/linux.html)
* Clone this repository:

  ```sh
  git clone https://github.com/ezwelty/xgrains.git
  cd xgrains
  ```

  _Alternatively, download the repo as a zip file, unzip it, open a terminal, and navigate to the directory._

* Create and activate the `xgrains` Python environment:

  ```sh
  conda env create --file environment.yaml
  conda activate xgrains
  ```

## Usage

Open a Python interpreter (`python` or `ipython`) and
import and execute one of the `process_xrf_*` functions
from [`functions.py`](functions.py).

_Important: XRF scan filenames should be formatted as `{basename}_{element}.{extension}`._

```py
from pathlib import Path
from functions import process_xrf_scans, process_xrf_scans_multiple

# ---- Process XRF scans ----

process_xrf_scans(
  # XRF scans to process
  scans=Path('../data').glob('072122-101-2b_10um_10ms_coarse1_*.txt'),
  # Output directory
  output='results-single-2',
  # Stardist probability threshold
  prob_thresh=0.3,
  # Delimiter used in the XRF scans
  sep=';'
)

# ---- Process XRF scans with multiple parameter combinations ----

process_xrf_scans_multiple(
  # XRF scans to process
  scans=Path('../data').glob('072122-101-2b_10um_10ms_coarse1_*.txt'),
  # Output directory
  output='results-multiple-2',
  # Stardist probability thresholds
  prob_thresh=[0.3],
  # Delimiter used in the XRF scans
  sep=';'
)
```
