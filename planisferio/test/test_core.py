"""Unit tests of the SLAM core (no ROS needed)."""

import math
import os

import numpy as np
import pytest

from planisferio.core import (CorrelativeScanMatcher, OccupancyGrid, compose, inverse,
                              occupancy_to_pgm, relative, save_map, wrap_angle)
from planisferio.core.pose_graph import (_errors, _jacobians, make_edge, optimize)
from planisferio.core.se2 import compose_many, relative_many


def test_se2_compose_inverse_relative():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform([-5, -5, -math.pi], [5, 5, math.pi])
        b = rng.uniform([-5, -5, -math.pi], [5, 5, math.pi])
        np.testing.assert_allclose(compose(a, inverse(a)), [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(compose(a, relative(a, b)), b, atol=1e-9)
    assert abs(wrap_angle(3 * math.pi) - math.pi) < 1e-12 or abs(wrap_angle(3 * math.pi) + math.pi) < 1e-12


def test_se2_vectorized_matches_scalar():
    rng = np.random.default_rng(1)
    a = rng.uniform([-5, -5, -math.pi], [5, 5, math.pi])
    bs = rng.uniform([-5, -5, -math.pi], [5, 5, math.pi], size=(20, 3))
    np.testing.assert_allclose(relative_many(a, bs), [relative(a, b) for b in bs], atol=1e-9)
    np.testing.assert_allclose(compose_many(a, bs), [compose(a, b) for b in bs], atol=1e-9)


def test_pose_graph_jacobians_match_numerical_derivatives():
    rng = np.random.default_rng(2)
    X = rng.uniform([-3, -3, -math.pi], [3, 3, math.pi], size=(2, 3))
    I, J = np.array([0]), np.array([1])
    Z = rng.uniform([-1, -1, -1], [1, 1, 1], size=(1, 3))
    A, B = _jacobians(X, I, J, Z)
    eps = 1e-6
    for node, analytic in ((0, A[0]), (1, B[0])):
        num = np.zeros((3, 3))
        for k in range(3):
            Xp, Xm = X.copy(), X.copy()
            Xp[node, k] += eps
            Xm[node, k] -= eps
            d = _errors(Xp, I, J, Z)[0] - _errors(Xm, I, J, Z)[0]
            d[2] = wrap_angle(d[2])
            num[:, k] = d / (2 * eps)
        np.testing.assert_allclose(analytic, num, atol=1e-6)


def test_pose_graph_closes_a_drifting_square():
    true = np.array([[0, 0, 0], [1, 0, math.pi / 2], [1, 1, math.pi], [0, 1, -math.pi / 2]])
    edges = [make_edge(k, (k + 1) % 4, relative(true[k], true[(k + 1) % 4]), np.eye(3) * 100,
                       'loop' if k == 3 else 'seq') for k in range(4)]
    noisy = true + np.array([[0, 0, 0], [0.1, 0.05, 0.05], [0.2, 0.1, 0.1], [0.3, 0.1, 0.15]])
    X, stats = optimize(noisy, edges)
    assert np.abs(X - true).max() < 1e-4
    assert stats.chi2_after < stats.chi2_before


def test_pose_graph_rejects_a_wrong_loop_closure():
    n = 12
    true = np.array([[math.cos(2 * math.pi * k / n) * 3, math.sin(2 * math.pi * k / n) * 3,
                      wrap_angle(2 * math.pi * k / n + math.pi / 2)] for k in range(n)])
    info = np.eye(3) * 400
    edges = [make_edge(k, k + 1, relative(true[k], true[k + 1]), info, 'seq') for k in range(n - 1)]
    edges.append(make_edge(n - 1, 0, relative(true[n - 1], true[0]), info, 'loop'))
    bad = make_edge(3, 9, np.array([0.5, 0.0, 0.0]), info, 'loop')     # a false match
    edges.append(bad)
    X, stats = optimize(true, edges)
    assert bad.id in stats.rejected_edges
    assert np.abs(X[:, :2] - true[:, :2]).max() < 0.01


def test_scan_matcher_recovers_a_known_offset():
    rng = np.random.default_rng(3)
    walls = np.vstack([np.column_stack((np.linspace(0, 6, 300), np.zeros(300))),
                       np.column_stack((np.zeros(200), np.linspace(0, 4, 200))),
                       np.column_stack((np.linspace(2, 3, 50), np.full(50, 2.0)))])
    true = np.array([0.4, -0.25, math.radians(6)])
    inv = inverse(true)
    c, s = math.cos(inv[2]), math.sin(inv[2])
    query = np.column_stack((inv[0] + c * walls[:, 0] - s * walls[:, 1],
                             inv[1] + s * walls[:, 0] + c * walls[:, 1]))
    query += rng.normal(0, 0.01, query.shape)
    res = CorrelativeScanMatcher(resolution=0.05, sigma=0.05).match(
        walls, query, np.zeros(3), lin_window=0.8, ang_window=math.radians(15))
    assert res is not None
    assert np.hypot(*(res.pose[:2] - true[:2])) < 0.06
    assert abs(wrap_angle(res.pose[2] - true[2])) < math.radians(1.5)
    assert res.score > 0.5


def test_occupancy_grid_marks_hits_and_free_space():
    grid = OccupancyGrid(resolution=0.05, initial_size_m=4.0)
    hits = np.array([[1.5, 0.0], [1.5, 0.05], [1.5, -0.05]])
    for _ in range(3):
        grid.integrate_scan(np.zeros(2), hits, np.zeros((0, 2)))
    values, origin = grid.to_occupancy_values(crop=False)
    cell = lambda x, y: (int((y - origin[1]) / grid.res), int((x - origin[0]) / grid.res))  # noqa: E731
    assert values[cell(1.5, 0.0)] >= 65          # wall
    assert 0 <= values[cell(0.7, 0.0)] <= 25     # free space along the beam
    assert values[cell(-1.5, 1.5)] == -1         # never seen


def test_grid_grows_beyond_its_initial_size():
    grid = OccupancyGrid(resolution=0.05, initial_size_m=2.0)
    grid.integrate_scan(np.zeros(2), np.array([[7.0, -6.0]]), np.zeros((0, 2)))
    values, origin = grid.to_occupancy_values(crop=False)
    assert origin[0] + values.shape[1] * grid.res > 7.0
    assert origin[1] < -6.0


def test_save_map_writes_map_server_files(tmp_path):
    values = np.full((4, 6), -1, dtype=np.int8)
    values[0, :] = 100          # bottom row: wall
    values[1, :] = 10           # free
    pgm, yml = save_map(values, (-1.0, 2.0), 0.05, str(tmp_path), 'test_map')
    assert os.path.isfile(pgm) and os.path.isfile(yml)
    data = open(pgm, 'rb').read()
    assert data.startswith(b'P5\n6 4\n255\n')
    img = np.frombuffer(data[len(b'P5\n6 4\n255\n'):], dtype=np.uint8).reshape(4, 6)
    assert (img[-1] == 0).all() and (img[-2] == 254).all() and (img[0] == 205).all()
    text = open(yml).read()
    assert 'image: test_map.pgm' in text and 'resolution: 0.05' in text
    assert 'origin: [-1.0000, 2.0000, 0.0]' in text


def test_occupancy_to_pgm_thresholds():
    img = occupancy_to_pgm(np.array([[-1, 0, 50, 51, 64, 65, 100]]))
    assert img.tolist() == [[205, 254, 254, 205, 205, 0, 0]]


@pytest.mark.parametrize('bad', [np.zeros((0, 2))])
def test_empty_scan_is_harmless(bad):
    grid = OccupancyGrid()
    grid.integrate_scan(np.zeros(2), bad, bad)
    assert grid.occupied_count() == 0
