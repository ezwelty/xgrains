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
  conda config --set channel_priority strict
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
  scans=Path('data').glob('basename_*.txt'),
  # Output directory
  output='results-single',
  # Stardist probability threshold
  prob_thresh=0.3,
  # Delimiter used in the XRF scans
  sep=';',
  # Element overlaps (base: [overlap])
  element_overlaps={
    'Fe': ['S', 'Cr', 'Ca', 'Ti'],
    'K': ['Al']
  },
)

# ---- Process XRF scans with multiple parameter combinations ----

process_xrf_scans_multiple(
  # XRF scans to process
  scans=Path('../data').glob('basename_*.txt'),
  # Output directory
  output='results-multiple',
  # Stardist probability thresholds
  prob_thresh=[0.3],
  # Delimiter used in the XRF scans
  sep=';',
  # ... other parameters (see process_xrf_scans)
)
```

Grain overlaps (`overlaps.csv`) can always be recomputed from the results by running `process_overlaps` separately:

```py
from functions import process_overlaps

process_overlaps(
  # Results directory
  path='results-single',
  # Elements (base: [overlap])
  elements={
    'Fe': ['S', 'Cr', 'Ca', 'Ti'],
    'K': ['Al']
  }
)
```
