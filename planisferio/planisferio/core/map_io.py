"""
Saving an occupancy grid in the map_server format (PGM image + YAML file).

The files load directly in nav2_map_server, Sendero, RViz and most ROS tools:
  0   black   occupied (occupancy >= occupied_min)
  254 white   free     (0 <= occupancy <= free_max)
  205 grey    unknown  (-1, or between the two thresholds)
"""

import os

import numpy as np


def occupancy_to_pgm(values, free_max=50, occupied_min=65):
    """ROS occupancy values (-1 / 0..100, row 0 at the bottom) -> PGM pixels."""
    values = np.asarray(values)
    img = np.full(values.shape, 205, dtype=np.uint8)
    img[(values >= 0) & (values <= free_max)] = 254
    img[values >= occupied_min] = 0
    return np.flipud(img)   # PGM rows go top-down; grid row 0 is the bottom


def save_map(values, origin, resolution, directory, name, free_max=50, occupied_min=65):
    """Write <directory>/<name>.pgm and <name>.yaml; return both paths."""
    directory = os.path.expanduser(directory)
    os.makedirs(directory, exist_ok=True)
    pgm = os.path.join(directory, name + '.pgm')
    yml = os.path.join(directory, name + '.yaml')
    img = occupancy_to_pgm(values, free_max, occupied_min)
    with open(pgm, 'wb') as f:
        f.write(f'P5\n{img.shape[1]} {img.shape[0]}\n255\n'.encode())
        f.write(img.tobytes())
    with open(yml, 'w') as f:
        f.write(f'image: {name}.pgm\n'
                f'mode: trinary\n'
                f'resolution: {float(resolution)}\n'
                f'origin: [{float(origin[0]):.4f}, {float(origin[1]):.4f}, 0.0]\n'
                f'negate: 0\n'
                f'occupied_thresh: {occupied_min / 100.0:.2f}\n'
                f'free_thresh: {min(free_max, 25) / 100.0:.2f}\n')
    return pgm, yml
