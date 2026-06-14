"""
view_selection.py — Unified View Selector for ExploreGS ablation study.

Provides four strategies for selecting B virtual cameras from a candidate pool:
  1. Coverage-Greedy  (基于几何覆盖的最大化)
  2. Fisher-Greedy    (基于 Fisher 信息增益)
  3. Random           (随机选择)
  4. Farthest-Point   (最远点采样)

Usage:
    from view_selection import ViewSelector
    selector = ViewSelector(scene, gaussians, pipe, bg_color, render_func, render_fisher)
    selected_indices = selector.select(candidate_cams, strategy="coverage", B=10)
"""

import numpy as np
import torch
import copy
import time
import heapq
from tqdm import tqdm
from einops import reduce, repeat
from utils.search_utils import make_sphere_view_directions

_EPS = 1e-9


class ViewSelector:
    """
    Unified view selector for ablation study.

    Args:
        gaussians: GaussianModel (trained, from Stage 1)
        pipe: PipelineParams
        bg_color: background color tensor
        render_func: render function (render_upgrade)
        render_fisher: render function that supports backward (modified_render)
        train_cameras: list of training Camera objects
        filter_params: list of parameter indices for Fisher computation
    """

    def __init__(self, gaussians, pipe, bg_color, render_func, render_fisher,
                 train_cameras=None, filter_params=None, device="cuda"):
        self.gaussians = gaussians
        self.pipe = pipe
        self.bg_color = bg_color
        self.render_func = render_func
        self.render_fisher = render_fisher
        self.train_cameras = train_cameras if train_cameras is not None else []
        self.device = device
        self.filter_params = filter_params if filter_params is not None else [0, 1, 2, 3, 5]

        # For Coverage-Greedy: precompute view directions
        self.dir_angle = 30  # default
        self.view_dirs = None
        self.grid_view_dirs = None

        # For Fisher-Greedy: precompute I_train
        self.I_train = None

        # For timing
        self.timing = {}

    def build_view_directions(self, dir_angle=30):
        """Build spherical view directions for coverage computation."""
        self.dir_angle = dir_angle
        up_vec = np.array([cam.c2w[:3, 1].numpy() for cam in self.train_cameras])
        mean_up_vec = np.mean(up_vec, axis=0)
        mean_up_vec = mean_up_vec / np.linalg.norm(mean_up_vec)
        views = make_sphere_view_directions(dir_angle, dir_angle, np.array([0, 0, 0]), mean_up_vec)
        self.view_dirs = np.array(views)[:, :3, 2]  # forward vectors (M, 3)
        self.n_view_dirs = len(self.view_dirs)
        print(f"[ViewSelector] Built {self.n_view_dirs} view directions (angle={dir_angle}°)")
        return self.view_dirs

    def build_fisher_I_train(self):
        """Precompute I_train = 1/(H_train + lambda) for Fisher-Greedy."""
        params = list(self._capture_params())
        H_train = torch.zeros(sum(p.numel() for p in params), device=self.device)
        for cam in tqdm(self.train_cameras, desc="[Fisher] Building I_train"):
            render_pkg = self.render_fisher(cam, self.gaussians, self.pipe, self.bg_color)
            pred_img = render_pkg["render"]
            pred_img.backward(gradient=torch.ones_like(pred_img))
            cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
            H_train += cur_H
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        reg_lambda = 1e-6
        self.I_train = torch.reciprocal(H_train + reg_lambda).cpu()
        print(f"[ViewSelector] Built I_train (dim={self.I_train.shape[0]})")
        return self.I_train

    def _capture_params(self):
        """Capture trainable parameters for Fisher computation."""
        key_filter = self.filter_params
        params = self.gaussians.capture()[1:7]
        return [p.requires_grad_(True) for i, p in enumerate(params) if i in key_filter]

    # ========== Selection Strategies ==========

    def select(self, candidate_cams, strategy="coverage", B=10, **kwargs):
        """
        Unified selection interface.

        Args:
            candidate_cams: list of Camera objects (candidate pool, size K)
            strategy: one of ["coverage", "fisher", "random", "farthest"]
            B: number of cameras to select

        Returns:
            selected_indices: list of indices into candidate_cams (length B)
            timing: dict with timing info
        """
        t_start = time.time()

        if strategy == "coverage":
            selected = self._coverage_greedy(candidate_cams, B, **kwargs)
        elif strategy == "fisher":
            selected = self._fisher_greedy(candidate_cams, B, **kwargs)
        elif strategy == "random":
            selected = self._random_select(candidate_cams, B)
        elif strategy == "farthest":
            selected = self._farthest_point(candidate_cams, B, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        elapsed = time.time() - t_start
        self.timing[strategy] = elapsed

        return selected, elapsed

    def _coverage_greedy(self, candidate_cams, B, grid_gs=None):
        """
        Coverage-Greedy: greedy maximum coverage on (gaussian, view_dir) pairs.

        This is the core logic from SpaceSearch.policy_view_coverage_refine(),
        extracted and extended to work as a pure set selector (not DFS).
        """
        n_gs = len(self.gaussians.get_xyz)
        n_views = self.n_view_dirs
        grid_view_dirs = np.zeros((n_gs, n_views), dtype=bool)
        selected = []
        remaining = list(range(len(candidate_cams)))

        # Precompute for each candidate: visible mask and best view_dir index
        print(f"[Coverage-Greedy] Precomputing gains for {len(candidate_cams)} candidates...")
        cam_cache = []
        for idx, cam in enumerate(tqdm(candidate_cams, desc="[Coverage] Caching")):
            render_pkg = self.render_func(cam, self.gaussians, self.pipe, self.bg_color)
            mask_visible = render_pkg["visibility_filter"].detach().cpu().numpy()

            # Compute per-Gaussian view direction
            cam_view_dirs = self.gaussians.get_xyz[mask_visible].detach().cpu().numpy() - cam.c2w[:3, 3].numpy()
            cam_view_dirs = cam_view_dirs / (np.linalg.norm(cam_view_dirs, axis=1, keepdims=True) + 1e-15)

            # Best view_dir per Gaussian
            angle_sim = cam_view_dirs @ self.view_dirs.T  # (N_visible, M)
            best_dir_idx = np.argmax(angle_sim, axis=1)  # (N_visible,)

            cam_cache.append((mask_visible, best_dir_idx))

        # Greedy selection
        print(f"[Coverage-Greedy] Running greedy selection B={B}...")
        for round_i in range(B):
            best_gain = -1
            best_idx = -1

            for i in remaining:
                mask_visible, best_dir_idx = cam_cache[i]

                # Count newly covered (gaussian, view_dir) pairs
                new_coverage = ~grid_view_dirs[mask_visible, best_dir_idx]
                gain = new_coverage.sum()

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1 or best_gain <= 0:
                print(f"[Coverage] No more gain at round {round_i}, selecting remaining by distance...")
                # Fallback: pick by farthest distance from already selected
                for i in remaining:
                    if best_idx == -1:
                        best_idx = i
                        continue
                    pos_i = candidate_cams[i].c2w[:3, 3].numpy()
                    pos_best = candidate_cams[best_idx].c2w[:3, 3].numpy()
                    # Compare with existing selected
                    dist_i = 0
                    for s in selected:
                        dist_i += np.linalg.norm(pos_i - candidate_cams[s].c2w[:3, 3].numpy())
                    dist_best = 0
                    for s in selected:
                        dist_best += np.linalg.norm(pos_best - candidate_cams[s].c2w[:3, 3].numpy())
                    if dist_i > dist_best:
                        best_idx = i

            # Update covered set
            mask_visible, best_dir_idx = cam_cache[best_idx]
            grid_view_dirs[mask_visible, best_dir_idx] = True

            selected.append(best_idx)
            remaining.remove(best_idx)

            total_covered = grid_view_dirs.sum()
            coverage_ratio = total_covered / (n_gs * n_views) * 100
            print(f"  Round {round_i+1}/{B}: selected #{best_idx}, gain={best_gain}, "
                  f"total_coverage={total_covered}/{n_gs * n_views} ({coverage_ratio:.2f}%)")

        return selected

    def _fisher_greedy(self, candidate_cams, B):
        """
        Fisher-Greedy: greedy selection based on Fisher information gain.
        Based on SpaceSearch.policy_nbvs() + next_best_view_selection().
        """
        if self.I_train is None:
            print("[Fisher] I_train not precomputed. Computing now...")
            self.build_fisher_I_train()

        params = self._capture_params()
        optim = torch.optim.SGD(params, 0.)
        original_optim = self.gaussians.optimizer
        self.gaussians.optimizer = optim

        selected = []
        remaining = list(range(len(candidate_cams)))

        # For lazy-greedy: maintain upper bounds for each candidate
        # (for speed comparison, we implement standard greedy here)

        print(f"[Fisher-Greedy] Running greedy selection B={B}...")
        for round_i in range(B):
            best_score = -float('inf')
            best_idx = -1

            for i in remaining:
                cam = candidate_cams[i]
                render_pkg = self.render_fisher(cam, self.gaussians, self.pipe, self.bg_color)
                pred_img = render_pkg["render"]
                pred_img.backward(gradient=torch.ones_like(pred_img))
                cur_H = torch.cat([p.grad.detach().reshape(-1) for p in params])
                score = torch.sum(cur_H * self.I_train.to(cur_H.device)).item()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)
            print(f"  Round {round_i+1}/{B}: selected #{best_idx}, score={best_score:.4f}")

        self.gaussians.optimizer = original_optim
        return selected

    def _random_select(self, candidate_cams, B):
        """Random selection."""
        n = len(candidate_cams)
        B_actual = min(B, n)
        selected = np.random.choice(n, B_actual, replace=False).tolist()
        print(f"[Random] Selected {len(selected)} cameras randomly from {n}")
        return selected

    def _farthest_point(self, candidate_cams, B):
        """
        Farthest-Point Sampling (FPS) based on camera positions.
        Uses camera centers in 3D space.
        """
        n = len(candidate_cams)
        B_actual = min(B, n)

        positions = np.array([cam.c2w[:3, 3].numpy() for cam in candidate_cams])

        # Start with the camera farthest from the mean of training cameras
        if len(self.train_cameras) > 0:
            train_pos = np.array([cam.c2w[:3, 3].numpy() for cam in self.train_cameras])
            train_center = train_pos.mean(axis=0)
            dists = np.linalg.norm(positions - train_center, axis=1)
            selected = [np.argmax(dists)]
        else:
            selected = [0]

        remaining = list(set(range(n)) - set(selected))

        print(f"[Farthest-Point] Running FPS B={B_actual}...")
        for round_i in range(1, B_actual):
            max_min_dist = -1
            best_idx = -1

            for i in remaining:
                pos_i = positions[i]
                min_dist = np.min(np.linalg.norm(positions[selected] - pos_i, axis=1))
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx == -1:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)
            print(f"  Round {round_i+1}/{B_actual}: selected #{best_idx}, min_dist={max_min_dist:.4f}")

        return selected


def generate_candidate_pool(scene, gaussians, cam_generator, pipe, bg_color,
                            render_func, cam_params, subsample=1):
    """
    Generate a pool of candidate virtual cameras from the scene.

    This mirrors the candidate generation in SpaceSearch.action_anchor_set()
    but decoupled from DFS: for each training camera (or subset), generate
    all 14 candidate actions (6t + 4r + 4o) and filter invalid ones.

    Returns:
        candidate_cams: list of Camera objects (the pool)
    """
    from scene.space_search import SpaceSearch
    from utils.general_utils import inverse_sigmoid

    # Precompute depth threshold and mesh for rejection
    # We reuse SpaceSearch's init_mesh logic
    # But for simplicity, we create a lightweight version

    train_cameras = scene.getTrainCameras()
    train_cameras_sorted = sorted(train_cameras, key=lambda x: x.image_name)
    train_cameras_pos = np.array([cam.c2w[:3, 3].numpy() for cam in train_cameras_sorted])

    # Simplified rejection: use SpaceSearch's camera_rejection if available
    # Here we use a direct approach

    device = "cuda"
    bg_tensor = torch.tensor(bg_color, device=device, dtype=torch.float32)

    candidate_cams = []
    anchor_subset = train_cameras_sorted[::subsample] if subsample > 0 else train_cameras_sorted

    print(f"[CandidatePool] Generating candidates from {len(anchor_subset)} anchor cameras...")
    print(f"[CandidatePool] Using anchor_set = {cam_params.search_anchor_set}")

    from scene.space_search import Trajectory
    # We create a minimal SpaceSearch just for candidate generation
    # Or we directly use the functions

    # Actually, let's directly use SpaceSearch's action_anchor_set and camera_rejection
    # We'll create a dummy SpaceSearch instance for this

    # First do a minimal initialization to get rejection working
    search = SpaceSearch.__new__(SpaceSearch)
    search.gaussians = gaussians
    search.pipe = pipe
    search.bg_color = bg_tensor
    search.render_func = render_func
    search.render_fisher = render_func  # not needed for coverage
    search.train_cameras = train_cameras_sorted
    search.train_cameras_pos = train_cameras_pos
    search.search_space = np.array([[0, 0, 0]])  # dummy, will be replaced
    search.labels = np.array(['free'])  # dummy
    search.voxel_size = 0.1
    search.depth_threshold = 0.5
    search.skip_rejection = False
    search.skip_freespace_rejection = False
    search.cam_params = cam_params
    search.dbg_dir = f"{scene.model_path}/candidate_pool"
    os.makedirs(search.dbg_dir, exist_ok=True)
    search.device = device
    search.image_height = train_cameras_sorted[0].image_height
    search.image_width = train_cameras_sorted[0].image_width

    # Set params needed by camera_rejection
    search.space_unit = 0.1
    search.bg_filter = torch.tensor([10.0, 10.0, 10.0], device=device, dtype=torch.float32)
    search.occ_threshold = cam_params.search_occ_threshold
    search.reject_depth = cam_params.search_reject_depth
    search.anchor_set = cam_params.search_anchor_set
    search.anchor_set_angle = cam_params.search_anchor_set_angle
    search.anchor_set_angle_orbit = cam_params.search_anchor_set_angle_orbit
    search.fix_right_vec = cam_params.search_fix_right_vec
    search.fix_up_vec = cam_params.search_fix_up_vec
    search.voxel_check = lambda point, space, labels, radius: True  # skip freespace check
    search.bg_reject_ratio = cam_params.search_bg_reject_ratio
    search.cam_radius = 1.0
    search.t_cache = None

    # Create a Trajectory-like object for get_lookat_depth
    class DummyTraj:
        def __init__(self):
            self.path = []
            self.directions = []
            self.terminate = False
            self._traj_id = 0
            self.grid_view_dirs = None
            self.root = ""

    # This is the anchor cam we'll carry through
    for anchor_cam in tqdm(anchor_subset, desc="[CandidatePool] Generating"):
        # Get lookat depth
        with torch.no_grad():
            render_pkg = render_func(anchor_cam, gaussians, pipe, bg_tensor)
        pred_depth = render_pkg["depth"]
        h, w = pred_depth.shape[-2:]
        lookat_depth = pred_depth[0, int(h//4):int(h*3//4), int(w//4):int(w*3//4)].mean().item()
        anchor_cam.lookat_depth = lookat_depth

        c2w_np = anchor_cam.c2w.numpy()
        candidate_c2ws, names = search.action_anchor_set(
            c2w_np, depth=lookat_depth if "4o" in search.anchor_set else None
        )

        for cand_c2w, name in zip(candidate_c2ws, names):
            cand_w2c = np.linalg.inv(cand_c2w)
            v_cam = search.create_vcam(
                anchor_cam,
                R=cand_w2c[:3, :3].transpose(),
                T=cand_w2c[:3, 3],
                image_name=f"candidate_{anchor_cam.image_name}_{name}"
            )

            # Apply rejection (skip freespace check)
            with torch.no_grad():
                render_reject_pkg = render_func(v_cam, gaussians, pipe, bg_tensor)
            pred_img = render_reject_pkg["render"]

            # Simple bg rejection
            bg_mask = pred_img.mean(dim=0) > 9.0
            bg_ratio = bg_mask.sum() / bg_mask.numel()
            if bg_ratio > cam_params.search_bg_reject_ratio:
                continue

            # Simple depth rejection
            depth = render_reject_pkg["depth"]
            depth_mask = depth < search.depth_threshold * 0.5
            invalid_depth_ratio = depth_mask.sum() / depth_mask.numel()
            if invalid_depth_ratio > 0.5:
                continue

            candidate_cams.append(v_cam)

    print(f"[CandidatePool] Generated {len(candidate_cams)} valid candidates from "
          f"{len(anchor_subset)} anchor cameras")
    return candidate_cams


def export_selected_cameras(selected_cams, save_path):
    """Export selected cameras to JSON."""
    import json
    from utils.camera_utils import camera_to_JSON
    json_cams = []
    for id_, cam in enumerate(selected_cams):
        json_cams.append(camera_to_JSON(id_, cam))
    with open(save_path, 'w') as f:
        json.dump(json_cams, f, indent=4)
    print(f"[Export] Saved {len(selected_cams)} cameras to {save_path}")


def load_selected_cameras(load_path, device="cuda"):
    """Load selected cameras from JSON."""
    import json
    from scene.cameras import Camera
    import numpy as np
    import torch

    with open(load_path, 'r') as f:
        json_cams = json.load(f)

    cameras = []
    for cam_data in json_cams:
        R = np.array(cam_data["rotation"])
        T = np.array(cam_data["position"])
        FoVx = cam_data["FoVx"]
        FoVy = cam_data["FoVy"]

        cam = Camera(
            colmap_id=cam_data["id"],
            R=R, T=T,
            FoVx=FoVx, FoVy=FoVy,
            image=None,
            image_name=cam_data["img_name"],
            uid=cam_data["id"],
            trans=np.array([0, 0, 0]),
            scale=1.0,
            data_device=device,
            gt_alpha_mask=None,
        )
        cameras.append(cam)

    return cameras


def compute_selection_timing(candidate_cams, strategies, B_values, selector):
    """
    Measure timing for each strategy across different K and B.
    Used for the runtime analysis experiment.
    """
    import time

    results = {}
    for strategy in strategies:
        results[strategy] = {"B_timing": {}}

        for B in B_values:
            t_start = time.time()
            selected, elapsed = selector.select(candidate_cams, strategy=strategy, B=B)
            results[strategy]["B_timing"][B] = elapsed
            print(f"[Timing] {strategy} B={B}: {elapsed:.3f}s")

    return results


import os  # needed for the module
