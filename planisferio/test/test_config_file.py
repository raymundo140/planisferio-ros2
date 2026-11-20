"""The shipped parameter file and the node declare exactly the same parameters."""

import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def declared_parameters():
    src = open(os.path.join(PKG, 'planisferio', 'slam_node.py')).read()
    return set(re.findall(r'\bp\(\s*"([a-z_0-9]+)"', src))


def file_parameters():
    data = yaml.safe_load(open(os.path.join(PKG, 'config', 'planisferio.yaml')))
    return set(data['/**']['ros__parameters'])


def test_every_file_parameter_is_declared_by_the_node():
    unknown = file_parameters() - declared_parameters()
    assert not unknown, f'in planisferio.yaml but not declared by the node: {sorted(unknown)}'


def test_every_node_parameter_is_documented_in_the_file():
    missing = declared_parameters() - file_parameters()
    assert not missing, f'declared by the node but missing in planisferio.yaml: {sorted(missing)}'


def test_file_values_match_the_core_defaults():
    """The file documents the defaults: they must not silently disagree."""
    from planisferio.core import LoopClosureConfig, SlamConfig
    p = yaml.safe_load(open(os.path.join(PKG, 'config', 'planisferio.yaml')))['/**']['ros__parameters']
    d, dl = SlamConfig(), LoopClosureConfig()
    pairs = {'resolution': d.resolution, 'num_particles': d.num_particles,
             'seq_sigma_along': d.seq_sigma_along, 'seq_sigma_cross': d.seq_sigma_cross,
             'seq_sigma_theta': d.seq_sigma_theta, 'local_map_window': d.local_map_window_m,
             'loop_search_radius': dl.search_radius, 'loop_min_score': dl.min_score,
             'loop_match_window': dl.window, 'loop_outlier_chi2': dl.outlier_chi2}
    for name, default in pairs.items():
        assert abs(float(p[name]) - float(default)) < 1e-3 * max(1.0, abs(default)), name
    assert list(p['motion_alphas']) == list(d.alphas)
