"""
Saving and reloading a whole SLAM session: keyframes, scans and the pose graph.

The occupancy map alone (map_io.save_map) is enough to look at a map, but not to
keep working on it: Planisferio's map is built from keyframes, so a session file
stores those keyframes (their scan points in base_link), their optimized poses and
every edge of the pose graph. Loading it rebuilds the map exactly and lets the
robot localize on it again (SlamSystem.set_pose), with or without extending it.

File format: one compressed NumPy archive (.npz), no pickle.
  version, resolution          format version, map resolution [m]
  poses        (N, 3)          optimized keyframe poses (x, y, theta)
  stamps, path (N,)            keyframe time stamps, travelled distance
  origins      (N, 2)          LiDAR position in base_link per keyframe
  hits, misses, loop           point arrays of all keyframes, concatenated (float32)
  hits_n, misses_n, loop_n (N,) number of points of each keyframe in those arrays
  edge_ij (E, 2), edge_z (E, 3), edge_info (E, 9), edge_kind (E,)  0 = seq, 1 = loop
"""

import os

import numpy as np

from .keyframes import Keyframe, insert_keyframe
from .pose_graph import make_edge

FORMAT_VERSION = 1
_KINDS = {"seq": 0, "loop": 1}
_KIND_NAMES = {v: k for k, v in _KINDS.items()}


def _concat(arrays):
    arrays = [np.asarray(a, dtype=np.float32).reshape(-1, 2) for a in arrays]
    counts = np.array([a.shape[0] for a in arrays], dtype=np.int64)
    data = np.concatenate(arrays) if arrays else np.zeros((0, 2), np.float32)
    return data, counts


def _split(data, counts):
    out, k = [], 0
    for n in counts:
        out.append(np.asarray(data[k:k + n], dtype=np.float32))
        k += int(n)
    return out


def save_session(slam, path):
    """Write the session of a SlamSystem to `path` (.npz). Returns the path."""
    with slam.lock:
        n = slam.graph.n
        kfs = list(slam.keyframes[:n])
        poses = slam.graph.poses[:n].copy()
        edges = list(slam.graph.edges)
    if n == 0:
        raise ValueError("the map is empty: nothing to save")
    hits, hits_n = _concat([k.map_hits for k in kfs])
    misses, misses_n = _concat([k.map_misses for k in kfs])
    loop, loop_n = _concat([k.loop_points for k in kfs])
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        version=np.array(FORMAT_VERSION),
        resolution=np.array(slam.cfg.resolution),
        poses=poses,
        stamps=np.array([k.stamp for k in kfs]),
        path=np.array([k.path_dist for k in kfs]),
        origins=np.array([k.sensor_origin for k in kfs], dtype=np.float64).reshape(-1, 2),
        hits=hits, hits_n=hits_n, misses=misses, misses_n=misses_n, loop=loop, loop_n=loop_n,
        edge_ij=np.array([[e.i, e.j] for e in edges], dtype=np.int64).reshape(-1, 2),
        edge_z=np.array([e.z for e in edges], dtype=np.float64).reshape(-1, 3),
        edge_info=np.array([e.info for e in edges], dtype=np.float64).reshape(-1, 9),
        edge_kind=np.array([_KINDS.get(e.kind, 0) for e in edges], dtype=np.int64),
    )
    return path if path.endswith(".npz") else path + ".npz"


def load_session(slam, path):
    """Replace the content of a SlamSystem with a saved session.

    After loading, the map is complete but the robot's position on it is unknown:
    the system waits for SlamSystem.set_pose before it processes scans again.
    Returns the number of keyframes loaded.
    """
    with np.load(os.path.expanduser(path), allow_pickle=False) as f:
        version = int(f["version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"session format {version} is not supported (expected {FORMAT_VERSION})")
        res = float(f["resolution"])
        if abs(res - slam.cfg.resolution) > 1e-9:
            raise ValueError(f"session resolution {res} m differs from the configured "
                             f"{slam.cfg.resolution} m")
        poses = np.array(f["poses"], dtype=np.float64)
        stamps, travelled, origins = f["stamps"], f["path"], f["origins"]
        hits = _split(f["hits"], f["hits_n"])
        misses = _split(f["misses"], f["misses_n"])
        loop = _split(f["loop"], f["loop_n"])
        edge_ij, edge_z = f["edge_ij"], f["edge_z"]
        edge_info, edge_kind = f["edge_info"], f["edge_kind"]

    slam.reset()
    kfs = [Keyframe(id=i, stamp=float(stamps[i]), map_hits=hits[i], map_misses=misses[i],
                    sensor_origin=np.asarray(origins[i], dtype=np.float64),
                    loop_points=loop[i], path_dist=float(travelled[i]))
           for i in range(poses.shape[0])]
    with slam.lock:
        for kf, pose in zip(kfs, poses):
            slam.keyframes.append(kf)
            slam.graph.add_node(pose)
        for (i, j), z, info, kind in zip(edge_ij, edge_z, edge_info, edge_kind):
            slam.graph.add_edge(make_edge(i, j, z, info.reshape(3, 3),
                                          _KIND_NAMES.get(int(kind), "seq")))
    for kf, pose in zip(kfs, poses):
        insert_keyframe(slam.grid, kf, pose)
    slam.mark_loaded()
    return len(kfs)
