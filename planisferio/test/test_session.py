"""Map reuse: save a session, load it, place the robot and keep localizing (no ROS)."""

import math

import numpy as np
import pytest

from planisferio import benchmark as B
from planisferio.core import ScanPreprocessor, SlamSystem, load_session, save_session
from planisferio.core.se2 import relative_many


def _config(seed, **overrides):
    cfg = B.make_config('loop', 200, seed)
    cfg.loop.threaded = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _prepare(data):
    """The same scan preprocessing as the benchmark, and the map-frame ground truth."""
    pre = ScanPreprocessor(map_beam_step=max(1, data["n_beams"] // 900), match_beams=180,
                           map_max_range=min(10.0, data["max_range"]),
                           match_max_range=min(10.0, data["max_range"]))
    m = data["mount"]
    c, s = math.cos(m[2]), math.sin(m[2])
    pre.set_mounting([[c, -s], [s, c]], m[:2])
    data['pre'] = pre
    data['gt_map'] = relative_many(data['true'][0], data['true'])


@pytest.fixture(scope='module')
def run():
    """Map the small world once; keep the data, the system and the ground truth."""
    data = B.generate('small', n_beams=360, seed=1)
    _prepare(data)
    r = B.run_mode(data, 'loop')
    return data, r


def _replay(slam, data, first, last):
    """Feed scans first..last; return estimated poses and the matching ground truth."""
    est, gt = [], []
    for k, ranges in data['scans']:
        if k < first or k > last:
            continue
        odom = data['odom'][k]
        scan = data['pre'].prepare(ranges, data['angle_min'], data['angle_inc'],
                                   0.12, data['max_range'])
        res = slam.process(k * data['dt'], odom, scan)
        if slam.localized:
            est.append(res.pose.copy())
            gt.append(data['gt_map'][k])
    return np.array(est), np.array(gt)


def test_save_and_load_rebuild_the_same_map(run, tmp_path):
    data, r = run
    slam = r['slam']
    path = save_session(slam, str(tmp_path / 'session.npz'))
    other = SlamSystem(_config(1))
    n = load_session(other, path)
    assert n == slam.graph.n
    assert other.loaded and other.awaiting_pose and not other.localized
    np.testing.assert_allclose(other.graph.poses, slam.graph.poses)
    assert len(other.graph.edges) == len(slam.graph.edges)
    a, _ = slam.grid.to_occupancy_values(crop=True)
    b, _ = other.grid.to_occupancy_values(crop=True)
    assert a.shape == b.shape
    assert np.mean(a == b) > 0.99


def test_loaded_map_waits_for_a_pose_then_localizes(run, tmp_path):
    data, r = run
    path = save_session(r['slam'], str(tmp_path / 'session.npz'))
    slam = SlamSystem(_config(2, localization_only=True))
    load_session(slam, path)
    k0 = (len(data['odom']) // 3) // 5 * 5
    # Scans before set_pose are ignored.
    est, _ = _replay(slam, data, k0, k0 + 25)
    assert len(est) == 0 and slam.awaiting_pose
    # Place the robot 10 cm and 3 degrees off its true pose on the map.
    true = data['gt_map'][k0 + 30]
    slam.set_pose(true + np.array([0.10, -0.05, 0.05]), std=(0.1, 0.1, 0.08))
    n_kf = slam.graph.n
    est, gt = _replay(slam, data, k0 + 30, k0 + 2000)
    assert slam.localized and len(est) > 100
    err = np.hypot(*(est[50:, :2] - gt[50:, :2]).T)
    assert np.sqrt(np.mean(err ** 2)) < 0.10      # converged onto the map
    assert slam.graph.n == n_kf                    # localization only: map unchanged


def test_extending_a_loaded_map_links_the_new_keyframes(run, tmp_path):
    data, r = run
    path = save_session(r['slam'], str(tmp_path / 'session.npz'))
    slam = SlamSystem(_config(3))
    load_session(slam, path)
    k0 = (len(data['odom']) // 2) // 5 * 5
    slam.set_pose(data['gt_map'][k0], std=(0.03, 0.03, 0.03))
    n_kf = slam.graph.n
    _replay(slam, data, k0, k0 + 1000)
    assert slam.graph.n > n_kf                     # mapping continued
    first_new = n_kf
    linked = [e for e in slam.graph.edges if e.j == first_new]
    assert linked and linked[0].i < n_kf           # joined to an old keyframe
