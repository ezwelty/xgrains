import csv
import datetime
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Union
import json

import geopandas as gpd
import pandas as pd
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pygeos
import skimage.color
import skimage.exposure
import skimage.filters
import skimage.io
import skimage.measure
import stardist
import stardist.models
import tifffile


def read_xrf_scan(path: Union[str, Path], sep: str = ';') -> np.ndarray:
  """
  Read XRF scan as a 16-bit integer array.

  Parameters
  ----------
  path
    Path to XRF scan.
  sep
    Delimiter.
  """
  with open(path, 'r') as file:
    rows = list(csv.reader(file, delimiter=sep))
  return np.asarray(rows, dtype=np.uint16)


def render_array_as_image(
  im: np.ndarray,
  path: Union[str, Path] = None,
  levels: int = 512,
  gamma: float = 1.0,
  cmap: str = 'inferno'
) -> Optional[np.ndarray]:
  """
  Render array as an RGB image.

  Parameters
  ----------
  im
    Input array.
  path
    Output path. If None, the image is returned as an array.
  levels
    Number of distinct values to represent in output image.
  gamma
    Gamma correction.
  cmap
    Matplotlib colormap.
  """
  im = skimage.exposure.adjust_gamma(im, gamma)
  im = skimage.exposure.rescale_intensity(
    im,
    in_range=(im.min(), im.max()),
    out_range=(0, levels - 1)
  ).astype(np.uint16)
  colors = matplotlib.cm.get_cmap(cmap, levels)(range(levels))
  colors = (colors[:, :3] * 255).astype(np.uint8)
  rgb = colors[im.flat[:]].reshape((*im.shape, 3))
  if not path:
    return rgb
  skimage.io.imsave(path, rgb)


# TODO: skimage.measure.regionprops(regions, values)
def describe_regions(
  regions: List[skimage.measure._regionprops.RegionProperties]
) -> pd.DataFrame:
  """
  Describe image regions.

  Parameters
  ----------
  regions
    List of region properties (skimage.measure.regionprops).
  """
  df = pd.DataFrame(
    {
      'label': prop.label,
      'row': prop.centroid_weighted[0],
      'col': prop.centroid_weighted[1],
      'area': prop.area,
      'max': prop.intensity_max,
      'mean': prop.intensity_mean,
      'min': prop.intensity_min,
      'minor_axis': prop.axis_minor_length,
      'major_axis': prop.axis_major_length,
      'solidity': prop.solidity,
    }
    for prop in regions
  )
  df['roundness'] = (
    4 * df['area'] / (np.pi * df['minor_axis'] * df['major_axis'])
  )
  for col in ('row', 'col', 'max', 'min'):
    df[col] = df[col].round().astype(int)
  for col in ('mean', 'minor_axis', 'major_axis', 'solidity', 'roundness'):
    df[col] = df[col].round(3)
  return df


def color_region_labels(labels: np.ndarray) -> np.ndarray:
  """
  Color region labels as an RGB image.

  Parameters
  ----------
  labels
    Region labels.
  """
  colors = skimage.color.label2rgb(labels, bg_label=0)
  return (colors * 255).astype('uint8')


def write_params_to_json(
  path: Union[str, Path],
  scans: Iterable[Union[str, Path]],
  timestamp: datetime.datetime = datetime.datetime.utcnow(),
  **params: Any
) -> None:
  """
  Write parameters to JSON file.

  Parameters
  ----------
  path
    Output path.
  scans
    Scan paths.
  timestamp
    Datetime. Defaults to the current time.
  **params
    Additional parameters.
  """
  params = {
    'scans': [str(scan) for scan in scans],
    'timestamp': timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
    **params
  }
  with open(path, 'w') as file:
    json.dump(params, file, indent=2)


def plot_polygons(
  image: np.ndarray,
  polygons: Iterable[np.ndarray],
  intersecting: Iterable[int],
  on_border: Iterable[int],
  path: Union[str, Path] = None,
  dpi: int = 30,
  cmap: str = 'viridis'
) -> None:
  """
  Plot polygon outlines on an image.

  Parameters
  ----------
  image
    Array to use as the background image.
  polygons
    Polygon coordinates.
  intersecting
    Indices of intersecting polygons (plotted with dashed outline).
  on_border
    Indices of polygons that intersect the image borde
    (plotted with dotted outlines).
  path
    Output path. If None, the figure is left open.
  dpi
    Resolution (dots per inch) of output.
  cmap
    Matplotlib colormap to use for the background image.
  """
  fig = plt.figure(
    figsize=(image.shape[1] / dpi, image.shape[0] / dpi), dpi=dpi, frameon=False
  )
  ax = plt.gca()
  plt.imshow(image, cmap=cmap)
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  for i, polygon in enumerate(polygons):
    if i in on_border:
      style = 'dotted'
    elif i in intersecting:
      style = 'dashed'
    else:
      style = 'solid'
    plt.fill(
      polygon[:, 0],
      polygon[:, 1],
      facecolor='none',
      edgecolor='white',
      linewidth=dpi / dpi,
      linestyle=style,
      alpha=0.5
    )
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  plt.axis('off')
  fig.tight_layout(pad=0)
  if path:
    plt.savefig(path, dpi=dpi)
    plt.close('all')


def process_xrf_scan(
  path: Union[str, Path],
  output: Union[str, Path],
  model: stardist.models.StarDist2D,
  prob_thresh: float = 0.3,
  nms_thresh: float = 0.3,
  sep: str = ';'
) -> None:
  """
  Process an XRF scan.

  Parameters
  ----------
  path
    Path to XRF scan.
  output
    Output directory.
  model
    Pretrained stardist model.
  prob_thresh
    Stardist probability threshold.
  nms_thresh
    Stardist non-maximum suppression threshold.
  sep
    Delimiter used in XRF scan.
  """
  path = Path(path)
  output = Path(output)

  # ---- Load XRF data ----

  xrf = read_xrf_scan(path, sep=sep)
  median_xrf = skimage.filters.median(xrf, footprint=np.ones((3, 3)))

  # ---- Predict with Stardist ----

  _, details = model.predict_instances(
    img=median_xrf / median_xrf.max(),
    prob_thresh=prob_thresh,
    nms_thresh=nms_thresh,
    return_labels=False,
    return_predict=False
  )

  # ---- Filter regions ----

  if len(details['coord']) == 0:
    return
  # Convert coordinates to x, y order
  polygons = pygeos.creation.polygons([x.T[:, ::-1] for x in details['coord']])
  tree = pygeos.STRtree(polygons)
  xij = tree.query_bulk(polygons, predicate='intersects')
  not_self = xij[0] != xij[1]
  # Find regions that intersect each other
  intersecting = set([*xij[0][not_self], *xij[1][not_self]])
  # Find regions that intersect image border
  ny, nx = xrf.shape
  border = pygeos.linearrings(
    np.array([(0, 0), (nx, 0), (nx, ny), (0, ny)]) - 0.5
  )
  on_border = tree.query(border, predicate='intersects')
  # Keep complete and distinct regions
  keep = [
    i for i in range(len(polygons))
    if i not in intersecting and i not in on_border
  ]

  # ---- Label regions ----

  labels = stardist.geometry.polygons_to_label_coord(
    coord=details['coord'][keep],
    shape=xrf.shape
  )
  regions = skimage.measure.regionprops(labels, median_xrf)
  probabilities = details['prob'][keep]

  # ---- Write results to file ----
  output.mkdir(parents=True, exist_ok=True)

  # Write CSV
  df = describe_regions(regions)
  df['probability'] = probabilities.round(3)
  df.to_csv(output / f'{path.stem}.csv', index=False, na_rep='')

  # Write stardist polygon GeoJSON
  gdf = gpd.GeoDataFrame(geometry=polygons)
  gdf['class'] = 'valid'
  gdf.loc[list(intersecting), 'class'] = 'intersect'
  gdf.loc[list(on_border), 'class'] = 'border'
  gdf.to_file(output / f'{path.stem}-polygons.geojson', index=False)

  # Write region labels
  tifffile.imwrite(
    output / f'{path.stem}-labels.tif',
    labels.astype('uint16'),
    compression='zlib'
  )

  # Write RGB-colored labels
  rgb = color_region_labels(labels)
  skimage.io.imsave(output / f'{path.stem}-rgb.jpg', rgb)

  # Write mask preview
  tifffile.imwrite(
    output / f'{path.stem}-mask.tif',
    255 * (labels > 0).astype('uint8'),
    compression='zlib'
  )

  # Write preview image
  image = render_array_as_image(xrf, gamma=0.5)
  skimage.io.imsave(output / f'{path.stem}.jpg', image)

  # Plot polygons
  plot_polygons(
    image=image,
    polygons=[pygeos.get_coordinates(geom) for geom in polygons],
    path=output / f'{path.stem}.pdf',
    on_border=on_border,
    intersecting=intersecting
  )


def process_overlaps(
  path: Union[str, Path],
  elements: Dict[str, List[str]] = {
    'Fe': ['S', 'Cr', 'Ca', 'Ti'],
    'K': ['Al']
  }
) -> None:
  """
  Measure grain overlaps and write them to a file (overlaps.csv).

  Parameters
  ----------
  path
    Directory containing element label files ({element}-labels.tif) and
    a grains table (grains.csv).
  elements
    Dictionary mapping base elements to a list of overlapping elements.

  Returns
  -------
  Creates a file overlaps.csv in the `path` with the following columns:
  * base_element: Base element name ('Fe')
  * base_label: Label of base element grain (1)
  * overlap_element: Overlapping element name ('S')
  * base_area_fraction_with_overlap: Fraction of base element grain area that is
    covered by the overlapping element (0.5)
  * overlap_grain_count: Number of overlapping element grains (1)
  * overlap_label: Label of overlapping element grain, if unique (1), otherwise
    null.
  """
  path = Path(path)
  # Find label file for each element
  label_paths = sorted(path.glob('*-labels.tif'))
  pattern = re.compile(r'^.+_(?P<element>[A-Z][a-z]?)-labels$')
  element_labels = {}
  for label_path in label_paths:
    match = pattern.match(label_path.stem)
    if not match:
      raise ValueError(f'Invalid label filename: {label_path}')
    element = match.group('element')
    if element in element_labels:
      raise ValueError(f'Multiple label files for element {element}')
    element_labels[element] = label_path
  # Read grains table
  grain_path = path / 'grains.csv'
  if not grain_path.exists():
    raise FileNotFoundError('grains.csv not found')
  df = pd.read_csv(grain_path)
  # Iterate over element pairs
  pairs = set()
  results = []
  for a, bs in elements.items():
    # Read base element
    if a not in element_labels:
      continue
    a_labels = tifffile.imread(element_labels[a])
    a_regions = skimage.measure.regionprops(a_labels)
    mask = df['element'].eq(a)
    if not mask.any():
      raise ValueError(f'Element {a} not found in grains.csv')
    for b in bs:
      # Read overlapping element
      if b not in element_labels:
        continue
      pair = frozenset([a, b])
      if pair in pairs:
        continue
      print(f'Computing overlap: {a} vs {b}')
      pairs.add(pair)
      b_labels = tifffile.imread(element_labels[b])
      # Measure overlap
      for a_region in a_regions:
        values = b_labels[tuple(a_region.coords.T)]
        unique_values = np.unique(values[values > 0])
        if len(unique_values) == 0:
          continue
        result = {
          'base_element': a,
          'base_label': a_region.label,
          'overlap_element': b,
          'base_area_fraction_with_overlap': (values > 0).sum() / len(values),
          'overlap_grain_count': len(unique_values),
          'overlap_label': unique_values[0] if len(unique_values) == 1 else None
        }
        results.append(result)
  if results:
    # Write results to file
    df = pd.DataFrame(results).convert_dtypes()
    df.to_csv(path / 'overlaps.csv', index=False)


def process_xrf_scans(
  scans: Iterable[Union[str, Path]],
  output: Union[str, Path],
  model_name: str = '2D_versatile_fluo',
  prob_thresh: float = 0.3,
  nms_thresh: float = 0.3,
  element_overlaps: Optional[Dict[str, List[str]]] = {
    'Fe': ['S', 'Cr', 'Ca', 'Ti'],
    'K': ['Al']
  },
  sep: str = ';'
) -> None:
  """
  Process a set of XRF scans.

  Parameters
  ----------
  scans
    Scan paths. Scan filenames are expected to be in the format
    {basename}_{element}.{extension}
  output
    Output directory.
  model_name
    Name of pretrained stardist model.
  prob_thresh
    Stardist probability threshold.
  nms_thresh
    Stardist non-maximum suppression threshold.
  sep
    Delimiter used in XRF scans.
  """
  scans = list(scans)
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)

  # ---- Write parameters ----

  write_params_to_json(
    path=output / 'parameters.json',
    scans=scans,
    method='stardist',
    model=model_name,
    prob_thresh=prob_thresh,
    nms_thresh=nms_thresh
  )

  # ---- Load pretrained stardist model ----

  model = stardist.models.StarDist2D.from_pretrained(model_name)

  for scan in scans:
    print(scan.stem)
    process_xrf_scan(
      scan,
      output=output,
      model=model,
      prob_thresh=prob_thresh,
      nms_thresh=nms_thresh,
      sep=sep
    )

  # ---- Compile results ----

  pattern = re.compile(r'(?P<scan>.+)_(?P<element>[A-Z][a-z]?)')

  dfs = []
  for scan in scans:
    info = pattern.match(str(scan.stem)).groupdict()
    csv_path = output / f'{scan.stem}.csv'
    if csv_path.exists():
      df = pd.read_csv(csv_path).assign(**info)
      dfs.append(df)
    else:
      print(f'{csv_path} missing')

  if not dfs:
    return
  pd.concat(dfs, ignore_index=True).to_csv(output / 'grains.csv', index=False)

  # ---- Grain overlaps ----

  if element_overlaps:
    process_overlaps(output, elements=element_overlaps)


def process_xrf_scans_multiple(
  scans: Iterable[Union[str, Path]],
  output: Union[str, Path],
  prob_thresh: Iterable[float],
  nms_thresh: Iterable[float] = [0.3],
  **kwargs: Any
) -> None:
  """
  Process a set of XRF scans using different Stardist parameters.

  Parameters
  ----------
  scans
    Scan paths. Scan filenames are expected to be in the format
    {basename}_{element}.{extension}
  output
    Output directory. Results for each parameter set will be written to
    subdirectory prob{prob_thresh}-nms{nms_thresh}
    Processing is skipped if subdirectory already exists.
  prob_thresh
    Stardist probability thresholds.
  nms_thresh
    Stardist non-maximum suppression thresholds.
  **kwargs
    Additional keyword arguments passed to process_xrf_scans.
  """
  scans = list(scans)
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)

  # ---- Process scans ----

  for prob in prob_thresh:
    for nms in nms_thresh:
      path = output / f'prob{prob}-nms{nms}'
      print(path)
      if path.exists():
        print('Already exists, skipping.')
        continue
      process_xrf_scans(
        scans=scans, output=path, prob_thresh=prob, nms_thresh=nms, **kwargs
      )

  # ---- Compile results ----

  dfs = []
  for path in sorted(output.rglob('*/grains.csv')):
    df = pd.read_csv(path).convert_dtypes()
    metadata = json.loads((path.parent / 'parameters.json').read_text())
    df['prob_thresh'] = metadata['prob_thresh']
    df['nms_thresh'] = metadata['nms_thresh']
    dfs.append(df)
  df = pd.concat(dfs, axis=0, ignore_index=True)
  df.to_csv(output / 'grains.csv', index=False)
