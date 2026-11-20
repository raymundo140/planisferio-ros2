"""End to end: the SLAM core on simulated data with ground truth (no ROS)."""

import numpy as np
import pytest

from planisferio import benchmark as B


@pytest.fixture(scope='module')
def data():
    return B.generate('small', n_beams=360, seed=1)


def test_slam_beats_odometry_by_far(data):
    odo = B.run_mode(data, 'odometry')
    slam = B.run_mode(data, 'loop')
    ate_odo = B.ate(odo['est'], odo['gt'])
    ate_slam = B.ate(slam['est'], slam['gt'])
    assert ate_odo > 1.0                      # the odometry is badly biased on purpose
    assert ate_slam < 0.20                    # SLAM stays within a few centimetres
    loops = sum(1 for e in slam['events'] if e.startswith('loop closure'))
    assert loops > 0                          # the tour revisits places: loops must be found
    ghost = B.ghost_ratio(slam['grid'], data['world'], B.SCENARIOS['small']['start'],
                          B.axis_scale(slam['est'], slam['gt']))
    assert ghost < 0.05                       # no doubled walls


def test_threaded_backend_gives_a_consistent_map(data):
    r = B.run_mode(data, 'loop-thread', speed=0)
    assert B.ate(r['est'], r['gt']) < 0.25
    assert np.isfinite(r['times']).all()
