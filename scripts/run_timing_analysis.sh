#!/bin/bash
# ============================================================================
# run_timing_analysis.sh — 运行时间分析实验
#
# 测量当候选相机数量 K 增长时，各策略每轮贪心选择的时间成本，
# 以及 Lazy-Greedy 的加速效果。
# ============================================================================

SCENE=${1:-bicycle}
GPU=${2:-0}
DATASET_DIR=${3:-/data/zhongyao/code/ExploreGS/data/bicycle}
STAGE1_CKPT="output/stage1_default/${SCENE}/chkpnt30000.pth"
OUTPUT_DIR="output/timing_analysis/${SCENE}"

export CUDA_VISIBLE_DEVICES=${GPU}
mkdir -p ${OUTPUT_DIR}

echo "===== Timing Analysis: Runtime Scaling with K ====="

# ============================================================================
# 使用不同大小的 K（候选相机数量）测量贪心选择时间
# ============================================================================
python -c "
import sys, json, torch, numpy as np, time
sys.path.insert(0, '.')
from view_selection import ViewSelector
from gaussian_renderer import render_upgrade as render_func
from gaussian_renderer import modified_render as render_fisher
from scene import Scene, GaussianModel
from scene.cameras import Camera

device = 'cuda'

# Load model
args = type('Args', (), {
    'source_path': '${DATASET_DIR}', 'model_path': '${OUTPUT_DIR}',
    'sh_degree': 3, 'images': 'images', 'depths': '',
    'resolution': -1, 'white_background': False, 'data_device': 'cuda',
    'eval': True, 'random_init': False, 'loader': 'Nerfbusters',
    'isotropic': False, 'oracle': False, 'dataset_distribution': 'easy'
})()
opt = type('Opt', (), {'appearance_model': '', 'inv_depth': True})()
pipe = type('Pipe', (), {
    'convert_SHs_python': False, 'compute_cov3D_python': False,
    'debug': False, 'antialiasing': False
})()
mp = type('MP', (), {'use_view_direction': False, 'use_visibility': False,
    'use_visibility_mask': False, 'use_scale_mask': False,
    'use_viewdirection_mask': False, 'visibility_weight': False,
    'use_fisher': False, 'fisher_current': False,
    'filter': [1,2], 'pruning': -1, 'replace_gt': False,
    'use_ref': False, 'nodepth': False
})()

gaussians = GaussianModel(args, opt, mp)
scene = Scene(args, mp, gaussians)
ckpt = torch.load('${STAGE1_CKPT}')
gaussians.restore(ckpt[0], opt, mp)
del ckpt

bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)
train_cameras = scene.getTrainCameras()

selector = ViewSelector(
    gaussians=gaussians, pipe=pipe, bg_color=bg_tensor,
    render_func=render_func, render_fisher=render_fisher,
    train_cameras=train_cameras, device=device
)
selector.build_view_directions(dir_angle=30)

# Generate dummy candidate cameras (use training camera positions + noise)
# to test timing at different K values
train_positions = np.array([cam.c2w[:3, 3].numpy() for cam in train_cameras])
K_values = [50, 100, 200, 500, 1000, 2000]
B_test = 10

results = {}
real_cams = list(train_cameras)  # use real cameras as proxies

for K in K_values:
    print(f'\n--- K = {K} ---')
    # Sample/duplicate cameras to reach K
    if K <= len(real_cams):
        np.random.seed(42)
        idx = np.random.choice(len(real_cams), K, replace=False)
        test_cams = [real_cams[i] for i in idx]
    else:
        repeats = (K // len(real_cams)) + 1
        test_cams = (real_cams * repeats)[:K]

    K_actual = len(test_cams)
    results[K_actual] = {}

    # Test each strategy at this K
    for strategy in ['coverage', 'random', 'farthest']:
        t_start = time.time()
        selected, elapsed = selector.select(test_cams, strategy=strategy, B=B_test)
        results[K_actual][strategy] = {
            'elapsed': elapsed,
            'selected': selected,
            'time_per_round': elapsed / min(B_test, len(test_cams))
        }
        print(f'  {strategy}: {elapsed:.3f}s total, {elapsed/min(B_test, len(test_cams)):.3f}s/round')

# Save results
with open('${OUTPUT_DIR}/timing_vs_K.json', 'w') as f:
    json.dump({
        'K_values': list(results.keys()),
        'B': B_test,
        'results': {str(k): v for k, v in results.items()}
    }, f, indent=2)

print(f'\nTiming results saved to ${OUTPUT_DIR}/timing_vs_K.json')
"

echo ""
echo "===== Timing Analysis Complete ====="
echo "Results saved to ${OUTPUT_DIR}/"
