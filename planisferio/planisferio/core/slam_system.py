"""
SLAM system: particle-filter front-end + pose-graph back-end with loop
closure. Pure Python/NumPy, no ROS imports, so it can be unit-tested and
replayed offline.

FRONT-END (every scan, real time)
    if not initialized:
        pose = (0, 0, 0); first keyframe; done
    if the robot moved less than (update_min_d, update_min_a):
        pose = last corrected pose ⊕ odometry increment
    else:
        PREDICT particles with the odometry increment
        UPDATE weights with the likelihood field of the LOCAL map
        RESAMPLE if N_eff is low
        ESTIMATE corrected pose + covariance
        if moved more than (keyframe_d, keyframe_a) since the last keyframe:
            new KEYFRAME:
              - node in the pose graph + "seq" edge to the previous keyframe
              - inserted into the local map (front-end) and the global map
              - handed to the back-end

BACK-END (background thread, see loop_closure.py)
    detects loops, adds "loop" edges, optimizes the graph, rebuilds the map.

APPLYING A CORRECTION (start of every process() call)
    The back-end optimized keyframes 0..K while the front-end kept adding
    K+1..N. Keyframes 0..K take their optimized poses. The newer ones, and the
    current particles, are moved rigidly with keyframe K:

        T = X_K(optimized) ⊕ X_K(old)^-1,   X_i <- T ⊕ X_i   for i > K

    The rebuilt map is swapped in and the newer keyframes are inserted into it.

map -> odom (published as TF so odometry stays smooth):
    T_map_odom = T_map_base ⊕ (T_odom_base)^-1
"""

from dataclasses import dataclass, field
import math
import threading

import numpy as np

from .se2 import compose, compose_many, inverse, relative
from .occupancy_grid import OccupancyGrid
from .likelihood_field import LikelihoodField
from .particle_filter import ParticleFilter
from .keyframes import LocalMap, insert_keyframe, keyframe_from_scan
from .pose_graph import PoseGraph, make_edge, information_from_covariance
from .loop_closure import LoopClosureBackend, LoopClosureConfig


@dataclass
class SlamConfig:
    # Map
    resolution: float = 0.05
    initial_map_size_m: float = 20.0
    l_free: float = -0.4
    l_occ: float = 0.85
    l_min: float = -5.0
    l_max: float = 5.0
    occupied_threshold: float = 0.6
    # Likelihood field
    sigma_hit: float = 0.10
    z_hit: float = 0.9
    z_rand: float = 0.1
    field_max_dist: float = 0.5
    field_half_size_m: float = 12.0
    field_recenter_dist: float = 3.0
    # Particle filter
    num_particles: int = 200
    alphas: tuple = (0.05, 0.01, 0.05, 0.01)
    min_trans_std: float = 0.005
    min_rot_std: float = 0.005
    beta: float = 0.1
    resample_ratio: float = 0.5
    seed: int = None
    # Update policy
    update_min_d: float = 0.05
    update_min_a: float = 0.05
    keyframe_d: float = 0.15
    keyframe_a: float = math.radians(10.0)
    min_occupied_cells_for_matching: int = 30
    # Front-end map
    frontend_local_map: bool = True
    local_map_window_m: float = 20.0        # >= ~2x the LiDAR range (see README)
    local_map_min_keyframes: int = 10
    local_map_max_keyframes: int = 200
    # Uncertainty of the edge between consecutive keyframes (front-end step).
    # Anisotropic, in the robot frame: along-track (x), cross-track (y), heading.
    # Calibrated against ground truth in the offline simulation, using the drift
    # over hundreds of keyframes (not single steps): the front-end's heading
    # errors partly cancel from step to step, so long-range heading drift is far
    # smaller than independent per-step errors would suggest.
    seq_sigma_along: float = 0.003          # m
    seq_sigma_along_per_m: float = 0.01     # m per m travelled
    seq_sigma_cross: float = 0.001          # m
    seq_sigma_cross_per_m: float = 0.003
    seq_sigma_theta: float = 0.0004         # rad
    seq_sigma_theta_per_rad: float = 0.005  # rad per rad turned
    seq_sigma_theta_per_m: float = 0.001    # rad per m travelled
    seq_use_particle_cov: bool = False      # add the particle-cloud covariance
    # Map reuse: keep localizing on the map without adding keyframes (the map
    # stays exactly as loaded or built so far).
    localization_only: bool = False
    # Uncertainty of the edge that joins the first keyframe after set_pose to the
    # nearest existing keyframe (the robot was placed, not driven there).
    reseed_sigma_xy: float = 0.10           # m
    reseed_sigma_theta: float = 0.10        # rad
    # Loop closure / back-end
    loop: LoopClosureConfig = field(default_factory=LoopClosureConfig)


@dataclass
class SlamResult:
    pose: np.ndarray
    cov: np.ndarray
    filter_updated: bool = False
    keyframe: bool = False
    resampled: bool = False
    n_eff: float = 0.0
    match_ratio: float = float("nan")
    corrected: bool = False                 # a loop-closure correction was applied
    correction: np.ndarray = None           # rigid transform applied to the current pose
    backend: object = None                  # BackendResult of that correction


class SlamSystem:

    def __init__(self, config: SlamConfig = None):
        self.cfg = config or SlamConfig()
        c = self.cfg
        self._grid_kwargs = dict(resolution=c.resolution, initial_size_m=c.initial_map_size_m,
                                 l_free=c.l_free, l_occ=c.l_occ, l_min=c.l_min, l_max=c.l_max,
                                 occupied_threshold=c.occupied_threshold)
        self.lock = threading.Lock()
        self.graph = PoseGraph()
        self.keyframes = []
        self.backend = LoopClosureBackend(
            c.loop, self.keyframes, self.graph, self.lock, self._grid_kwargs,
            self._new_local_map if c.frontend_local_map else None)
        self.corrections = 0
        self.reset()

    def reset(self):
        c = self.cfg
        self.backend.reset()
        with self.lock:
            self.graph.clear()
            del self.keyframes[:]
        self.grid = OccupancyGrid(**self._grid_kwargs, clamp=True)
        self.local_map = self._new_local_map()
        self.pf = ParticleFilter(c.num_particles, c.alphas, c.min_trans_std,
                                 c.min_rot_std, c.resample_ratio, c.beta, c.seed)
        self.initialized = False
        self.pose = np.zeros(3)                 # corrected pose at last filter update
        self.cov = np.zeros((3, 3))
        self.odom_at_update = None              # odometry pose at last filter update
        self.last_keyframe_pose = None
        self._field = None
        self._field_version = -1
        self._field_occupied = 0
        # Map reuse (session.load_session / set_pose)
        self.loaded = False                     # the map came from a session file
        self.awaiting_pose = False              # map known, robot position not yet
        self.frontend_global = False            # match against the whole map
        self._pending_seed = None               # (pose, std) applied at the next scan
        self._link_next_keyframe = False        # join the next keyframe to the old graph

    def _new_local_map(self):
        c = self.cfg
        return LocalMap(self._grid_kwargs, c.local_map_window_m,
                        c.local_map_min_keyframes, c.local_map_max_keyframes)

    def shutdown(self):
        self.backend.shutdown()

    # ------------------------------------------------------------------
    def map_to_odom(self):
        """Correction transform T_map_odom (identity before initialization)."""
        if self.odom_at_update is None:
            return np.zeros(3)
        return compose(self.pose, inverse(self.odom_at_update))

    def pose_from_odom(self, odom_pose):
        return compose(self.map_to_odom(), odom_pose)

    def keyframe_poses(self):
        with self.lock:
            return self.graph.poses.copy()

    def graph_snapshot(self):
        """(poses, [(i, j, kind), ...]) for visualization."""
        with self.lock:
            return self.graph.poses.copy(), [(e.i, e.j, e.kind) for e in self.graph.edges]

    def request_optimization(self):
        self.backend.request_optimization()

    # ------------------------------------------------------------------
    def _frontend_grid(self):
        if self.frontend_global or not self.cfg.frontend_local_map:
            return self.grid
        return self.local_map.grid

    # ------------------------------------------------------------------
    # Map reuse
    # ------------------------------------------------------------------
    def mark_loaded(self):
        """Called by session.load_session after the keyframes are back in place."""
        self.loaded = True
        self.initialized = True                 # the map exists (published, saved)
        self.awaiting_pose = True               # but the robot is not placed on it yet
        self.frontend_global = True             # recent keyframes are not near the robot
        self._field = None

    def set_pose(self, pose, std=(0.05, 0.05, 0.05)):
        """Place the robot on the map at `pose` (x, y, theta) with spread `std`.

        Applied at the next processed scan (it needs the odometry at that time).
        Works after load_session (first placement) and during a run (re-localization).
        """
        self._pending_seed = (np.asarray(pose, dtype=np.float64).copy(),
                              tuple(float(s) for s in std))

    @property
    def localized(self):
        """True once the robot has a pose on the map (not waiting for set_pose)."""
        return self.initialized and not self.awaiting_pose

    def _apply_seed(self, odom_pose):
        pose, std = self._pending_seed
        self._pending_seed = None
        self.pf.initialize(pose, std)
        self.pose = pose.copy()
        self.cov = np.diag(np.square(std))
        self.odom_at_update = odom_pose.copy()
        self.last_keyframe_pose = pose.copy()
        self._field = None
        self.awaiting_pose = False
        with self.lock:
            has_graph = self.graph.n > 0
        self._link_next_keyframe = has_graph
        return SlamResult(self.pose.copy(), self.cov.copy())

    def _get_field(self, pose):
        c = self.cfg
        grid = self._frontend_grid()
        need = (self._field is None or self._field_version != grid.version
                or self._field_grid is not grid
                or np.hypot(*(pose[:2] - self._field.center)) > c.field_recenter_dist)
        if need:
            mask, origin = grid.occupied_window(pose[:2], c.field_half_size_m)
            self._field = LikelihoodField(mask, origin, c.resolution, c.sigma_hit,
                                          c.z_hit, c.z_rand, c.field_max_dist)
            self._field_version = grid.version
            self._field_grid = grid
            self._field_occupied = int(mask.sum())
        return self._field

    def _seq_information(self, prev_pose, z):
        """Information matrix of the edge between consecutive keyframes."""
        c = self.cfg
        d = math.hypot(z[0], z[1])
        s_along = c.seq_sigma_along + c.seq_sigma_along_per_m * d
        s_cross = c.seq_sigma_cross + c.seq_sigma_cross_per_m * d
        s_th = c.seq_sigma_theta + c.seq_sigma_theta_per_rad * abs(z[2]) + c.seq_sigma_theta_per_m * d
        cov = np.zeros((3, 3))
        if c.seq_use_particle_cov:
            # Particle covariance (map frame) rotated into the previous keyframe's frame.
            cth, sn = math.cos(prev_pose[2]), math.sin(prev_pose[2])
            R = np.array([[cth, -sn, 0.0], [sn, cth, 0.0], [0.0, 0.0, 1.0]])
            cov = R.T @ self.cov @ R
        return information_from_covariance(cov, (s_along, s_cross, s_th))

    def _add_keyframe(self, stamp, scan):
        if self._link_next_keyframe:
            self._add_reseeded_keyframe(stamp, scan)
            return
        with self.lock:
            kid = len(self.keyframes)
            if kid == 0:
                path = 0.0
            else:
                prev_pose = self.graph.poses[kid - 1].copy()
                z = relative(prev_pose, self.pose)
                path = self.keyframes[-1].path_dist + math.hypot(z[0], z[1])
            kf = keyframe_from_scan(kid, stamp, scan, path)
            self.keyframes.append(kf)
            self.graph.add_node(self.pose)
            if kid > 0:
                self.graph.add_edge(make_edge(kid - 1, kid, z,
                                              self._seq_information(prev_pose, z), "seq"))
        insert_keyframe(self.grid, kf, self.pose)
        if self.cfg.frontend_local_map:
            self.local_map.add(kf, self.pose)
        self.last_keyframe_pose = self.pose.copy()
        self.backend.notify_keyframe(kid)

    def _add_reseeded_keyframe(self, stamp, scan):
        """First keyframe after set_pose: joined to the nearest old keyframe.

        The robot was placed, not driven, so there is no odometry edge to the last
        keyframe; instead a loop-type edge (robust kernel, may be rejected as an
        outlier) links it to the closest keyframe with the uncertainty of the seed.
        """
        c = self.cfg
        self._link_next_keyframe = False
        with self.lock:
            kid = len(self.keyframes)
            poses = self.graph.poses[:kid]
            j = int(np.argmin(np.hypot(poses[:, 0] - self.pose[0], poses[:, 1] - self.pose[1])))
            path = self.keyframes[-1].path_dist
            kf = keyframe_from_scan(kid, stamp, scan, path)
            self.keyframes.append(kf)
            self.graph.add_node(self.pose)
            z = relative(poses[j], self.pose)
            info = information_from_covariance(
                np.zeros((3, 3)), (c.reseed_sigma_xy, c.reseed_sigma_xy, c.reseed_sigma_theta))
            self.graph.add_edge(make_edge(j, kid, z, info, "loop"))
        insert_keyframe(self.grid, kf, self.pose)
        if c.frontend_local_map:
            self.local_map.add(kf, self.pose)
        self.last_keyframe_pose = self.pose.copy()
        self.backend.notify_keyframe(kid)

    def _apply_backend_result(self):
        res = self.backend.poll()
        if res is None:
            return None, None
        K = res.last_id
        with self.lock:
            n = self.graph.n
            if K >= n:
                return None, None
            poses = self.graph.poses           # view: edits go into the graph
            T = compose(res.poses[K], inverse(poses[K]))
            poses[: K + 1] = res.poses
            if n > K + 1:
                poses[K + 1:n] = compose_many(T, poses[K + 1:n])
            new_poses = poses[:n].copy()
            kfs = self.keyframes[:n]

        # Swap in the rebuilt map and add the keyframes it does not contain yet.
        self.grid = res.grid
        for i in range(K + 1, n):
            insert_keyframe(self.grid, kfs[i], new_poses[i])
        if self.cfg.frontend_local_map:
            # The back-end already rebuilt the local map up to K; add the rest.
            self.local_map = res.local_map
            for i in range(K + 1, n):
                self.local_map.add(kfs[i], new_poses[i])

        # Move the current estimate and every particle with the newest keyframe.
        self.pose = compose(T, self.pose)
        self.pf.particles = compose_many(T, self.pf.particles)
        self.last_keyframe_pose = new_poses[n - 1].copy()
        self._field = None
        self.corrections += 1
        return T, res

    # ------------------------------------------------------------------
    def process(self, stamp, odom_pose, scan):
        c = self.cfg
        odom_pose = np.asarray(odom_pose, dtype=np.float64)
        T, backend_res = self._apply_backend_result()
        corrected = T is not None

        if self._pending_seed is not None:
            return self._apply_seed(odom_pose)
        if self.awaiting_pose:
            return SlamResult(self.pose.copy(), self.cov.copy())

        if not self.initialized:
            self.pose = np.zeros(3)
            self.pf.initialize(self.pose)
            self.odom_at_update = odom_pose.copy()
            self._add_keyframe(stamp, scan)
            self.initialized = True
            return SlamResult(self.pose.copy(), self.cov.copy(), keyframe=True)

        delta = relative(self.odom_at_update, odom_pose)
        if math.hypot(delta[0], delta[1]) < c.update_min_d and abs(delta[2]) < c.update_min_a:
            return SlamResult(compose(self.pose, delta), self.cov.copy(), corrected=corrected,
                              correction=T, backend=backend_res)

        # ---- PREDICT ----
        self.pf.predict(self.odom_at_update, odom_pose)
        predicted = compose(self.pose, delta)

        # ---- UPDATE (only when the map already has structure to match) ----
        field = self._get_field(predicted)
        resampled, ratio = False, float("nan")
        enough = self._field_occupied >= c.min_occupied_cells_for_matching
        if enough and scan.match_base.shape[0] > 0:
            loglik = field.score_particles(self.pf.particles, scan.match_base)
            self.pf.update(loglik)
            n_eff = self.pf.effective_sample_size()
            resampled = self.pf.resample_if_needed()
        else:
            n_eff = self.pf.effective_sample_size()

        # ---- ESTIMATE ----
        self.pose, self.cov = self.pf.estimate()
        self.odom_at_update = odom_pose.copy()
        if enough:
            ratio = field.match_ratio(self.pose, scan.match_base, 2.0 * c.sigma_hit)

        # ---- KEYFRAME ----
        kf_delta = relative(self.last_keyframe_pose, self.pose)
        keyframe = (not c.localization_only
                    and (math.hypot(kf_delta[0], kf_delta[1]) >= c.keyframe_d
                         or abs(kf_delta[2]) >= c.keyframe_a))
        if keyframe:
            self._add_keyframe(stamp, scan)

        return SlamResult(self.pose.copy(), self.cov.copy(), True, keyframe, resampled,
                          n_eff, ratio, corrected, T, backend_res)
