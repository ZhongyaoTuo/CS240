"""
train_selected.py — Train 3DGS with selected virtual cameras.

This script takes a Stage 1 checkpoint and a set of selected virtual camera
paths (JSON), adds them to the training dataset, trains for N iterations,
and evaluates on the test set.

Usage:
    python train_selected.py \
        -s /path/to/dataset \
        --eval -r 1 \
        --start_checkpoint output/stage1_default/bicycle/chkpnt30000.pth \
        --selected_cams output/selected_cams/coverage_B10.json \
        --iterations 30000 \
        --expname coverage_B10
"""

import os
import torch
import copy
import sys
import json
import numpy as np
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams, MaskingParams
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from gaussian_renderer import render_upgrade as render_func
from lpipsPyTorch import lpips

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import wandb


def load_selected_cameras_from_paths(json_paths, device="cuda"):
    """
    Load selected cameras from one or more JSON files.
    """
    cameras = []
    for jp in json_paths:
        with open(jp, 'r') as f:
            data = json.load(f)
        for cam_data in data:
            R = np.array(cam_data["rotation"])
            T = np.array(cam_data["position"])
            FoVx = cam_data["FoVx"]
            FoVy = cam_data["FoVy"]
            cam = Camera(
                colmap_id=cam_data["id"],
                R=R, T=T,
                FoVx=FoVx, FoVy=FoVy,
                image=None,  # No GT image for virtual cameras
                image_name=cam_data["img_name"],
                uid=cam_data["id"],
                trans=np.array([0, 0, 0]),
                scale=1.0,
                data_device=device,
                gt_alpha_mask=None,
            )
            cam.virtual = True
            cameras.append(cam)
    return cameras


def training(dataset, opt, pipe, mask_params, selected_cam_paths,
             testing_iterations, checkpoint_iterations,
             start_checkpoint, debug_from, logger=None, args=None):
    """Training loop with selected virtual cameras added to training set."""

    # Setup
    tb_writer = None
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load model and scene
    gaussians = GaussianModel(dataset, opt, mask_params)
    scene = Scene(dataset, mask_params, gaussians)

    # Load checkpoint
    if start_checkpoint:
        print(f"[INFO] Loading checkpoint: {start_checkpoint}")
        (model_params, ckpt_iter) = torch.load(start_checkpoint)
        gaussians.restore(model_params, opt, mask_params)
        del model_params
    else:
        raise ValueError("No checkpoint found")

    # Load selected virtual cameras
    print(f"[INFO] Loading selected cameras from: {selected_cam_paths}")
    vcam_list = load_selected_cameras_from_paths(selected_cam_paths)

    # Add selected cameras to training set
    train_cams = list(scene.getTrainCameras())
    print(f"[INFO] Original training cameras: {len(train_cams)}")
    for vcam in vcam_list:
        vcam.original_image = None  # no GT
    train_cams_extended = train_cams + vcam_list
    print(f"[INFO] Extended training cameras: {len(train_cams_extended)} (added {len(vcam_list)} virtual)")

    # Setup optimizer
    gaussians.training_setup(opt, mask_params)

    # Training loop
    first_iter = 1
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training")

    viewpoint_stack = None
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    for iteration in range(first_iter, opt.iterations + 1):
        # Update learning rate
        gaussians.update_learning_rate(iteration)
        gaussians.update_sh_degree(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera (from extended set)
        if not viewpoint_stack:
            viewpoint_stack = train_cams_extended.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render_func(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]

        # Loss: only compute if GT image exists
        gt_image = viewpoint_cam.original_image
        if gt_image is not None and viewpoint_cam.original_image is not None:
            gt_image = gt_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            # Virtual camera with no GT → use only photometric consistency?
            # For ablation: skip loss for virtual cameras (they don't have GT)
            # OR use pseudo-GT if available
            loss = torch.tensor(0.0, device="cuda", requires_grad=True)
            Ll1 = torch.tensor(0.0, device="cuda")

        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Evaluation
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                evaluate(scene, gaussians, render_func, pipe, background,
                         iteration, opt, mask_params, logger, args)

            # Densification (only with original training cameras?)
            if iteration < opt.densify_until_iter:
                render_pkg_full = render_func(viewpoint_cam, gaussians, pipe, bg)
                visibility_filter = render_pkg_full["visibility_filter"]
                radii = render_pkg_full["radii"]
                viewspace_points = render_pkg_full["viewspace_points"]

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_points, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity,
                                                scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           os.path.join(args.model_path, f"chkpnt{iteration}.pth"))

    # Final evaluation at specified test iterations
    for test_iter in testing_iterations:
        if test_iter <= opt.iterations:
            torch.cuda.empty_cache()
            evaluate(scene, gaussians, render_func, pipe, background,
                     test_iter, opt, mask_params, logger, args)

    print("\nTraining complete.")
    wandb.finish()


@torch.no_grad()
def evaluate(scene, gaussians, render_func, pipe, background,
             iteration, opt, mask_params, logger, args):
    """Evaluate on test set."""
    test_cameras = scene.getTestCameras()

    psnr_list = []
    ssim_list = []
    lpips_list = []

    render_path = os.path.join(args.model_path, f"test_{iteration:05d}")
    os.makedirs(os.path.join(render_path, "renders"), exist_ok=True)
    os.makedirs(os.path.join(render_path, "gt"), exist_ok=True)

    import torchvision

    for idx, viewpoint in enumerate(tqdm(test_cameras, desc="[Eval] Testing")):
        render_dict = render_func(viewpoint, gaussians, pipe, background)
        image = torch.clamp(render_dict["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)

        # Save images
        torchvision.utils.save_image(
            image, os.path.join(render_path, "renders", f"{idx:05d}.png"))
        torchvision.utils.save_image(
            gt_image, os.path.join(render_path, "gt", f"{idx:05d}.png"))

        # Metrics
        psnr_val = psnr(image, gt_image).mean().item()
        ssim_val = ssim(image, gt_image).item()
        lpips_val = lpips(image, gt_image, net_type='vgg').item()

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)

    # Aggregate
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_lpips = np.mean(lpips_list)

    print(f"\n[ITER {iteration}] Test Results:")
    print(f"  PSNR:  {avg_psnr:.4f}")
    print(f"  SSIM:  {avg_ssim:.4f}")
    print(f"  LPIPS: {avg_lpips:.4f}")

    # Save results
    results = {
        "iteration": iteration,
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips,
        "per_view_PSNR": psnr_list,
        "per_view_SSIM": ssim_list,
        "per_view_LPIPS": lpips_list,
    }
    with open(os.path.join(args.model_path, f"results_{iteration:05d}.json"), 'w') as f:
        json.dump(results, f, indent=2)

    if logger is not None:
        wandb.log({
            "test/PSNR": avg_psnr,
            "test/SSIM": avg_ssim,
            "test/LPIPS": avg_lpips,
        }, step=iteration)

    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Train 3DGS with selected views")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    mp = MaskingParams(parser)

    parser.add_argument('--start_checkpoint', type=str, required=True)
    parser.add_argument('--selected_cams', type=str, nargs='+', required=True,
                        help='Path(s) to JSON file(s) with selected virtual cameras')
    parser.add_argument('--expname', type=str, default="ablation")
    parser.add_argument('--iterations', type=int, default=30_000)
    parser.add_argument('--test_iterations', nargs='+', type=int,
                        default=[7_000, 15_000, 30_000])
    parser.add_argument('--save_iterations', nargs='+', type=int, default=[30_000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--image_log_interval', type=int, default=5000)

    args = parser.parse_args()

    # Override iterations
    op.iterations = args.iterations

    # Setup output path
    scene_name = os.path.basename(args.source_path)
    args.model_path = os.path.join("output", "ablation", f"{args.expname}", scene_name)
    os.makedirs(args.model_path, exist_ok=True)
    print(f"Output: {args.model_path}")

    # Config overrides
    args.eval = True
    args.loader = "Nerfbusters"  # adjust as needed

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(False)

    logger = wandb.init(name=f"ablation_{args.expname}_{scene_name}",
                        project='3dgs-ablation', config=vars(args))

    training(lp.extract(args), op.extract(args), pp.extract(args), mp.extract(args),
             args.selected_cams, args.test_iterations, args.save_iterations,
             args.start_checkpoint, -1, logger=logger, args=args)
