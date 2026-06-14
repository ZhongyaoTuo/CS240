#!/bin/bash
# ============================================================================
# run_ablation.sh — 完整消融实验脚本
#
# 用途：对四种视图选择策略（coverage / fisher / random / farthest）在
#       四种预算 B = 5,10,20,40 下进行实验。
#
# 流程：
#   1. Stage 1: 训练基础 3DGS（30K 迭代）
#   2. 生成候选相机池（每个策略共享同一候选池）
#   3. 对每种策略 + 预算组合：
#      a. 运行视图选择（生成选中的 B 个相机 JSON）
#      b. 训练 3DGS（从 Stage 1 checkpoint 继续训练 30K 迭代）
#      c. 评估 PSNR/SSIM/LPIPS
#   4. 汇总所有结果
#
# 用法：
#   bash scripts/run_ablation.sh <scene> <gpu> <dataset_dir>
#   示例：
#   bash scripts/run_ablation.sh bicycle 0 /data/zhongyao/code/ExploreGS/data/bicycle
# ============================================================================

set -e

# ---- 参数 ----
SCENE=${1:-bicycle}
GPU=${2:-0}
DATASET_DIR=${3:-/data/zhongyao/code/ExploreGS/data/bicycle}
OUTPUT_BASE="output/ablation_study"

# ---- 策略列表 ----
STRATEGIES=("coverage" "fisher" "random" "farthest")
BUDGETS=(5 10 20 40)

# ---- Stage 1 配置 ----
STAGE1_CFG="default"
STAGE1_OUTPUT="output/stage1_${STAGE1_CFG}/${SCENE}"
STAGE1_CKPT="${STAGE1_OUTPUT}/chkpnt30000.pth"
STAGE1_CONFIG_YAML="configs/stage1/${STAGE1_CFG}.yaml"

echo "=========================================="
echo "   Ablation Study: ExploreGS View Selection"
echo "   Scene: ${SCENE}"
echo "   GPU: ${GPU}"
echo "   Dataset: ${DATASET_DIR}"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=${GPU}

mkdir -p ${OUTPUT_BASE}/${SCENE}

# ============================================================================
# 第一步：Stage 1 训练（如果 checkpoint 不存在）
# ============================================================================
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo ""
    echo "===== Stage 1: Training base 3DGS ====="
    python train_stage1.py \
        -s ${DATASET_DIR} \
        --eval -r 1 \
        -c ${STAGE1_CONFIG_YAML} \
        --load Nerfbusters
    echo "[Stage 1] Done"
else
    echo "[Stage 1] Checkpoint exists: ${STAGE1_CKPT}"
fi

# ============================================================================
# 第二步：Stage 1 渲染 + 评估（获取基准指标）
# ============================================================================
if [ ! -d "${STAGE1_OUTPUT}/test/ours_30000" ]; then
    echo ""
    echo "===== Stage 1: Render & Evaluate ====="
    python render.py \
        -m ${STAGE1_OUTPUT} \
        -c ${STAGE1_CONFIG_YAML} \
        -s ${DATASET_DIR} --load Nerfbusters --stage1
    python metrics_stage1.py -m ${STAGE1_OUTPUT}
    echo "[Stage 1] Render done"
fi

# ============================================================================
# 第三步：为所有策略生成候选相机池（只需一次）
# ============================================================================
echo ""
echo "===== Generating Candidate Camera Pool ====="
python -c "
import sys
sys.path.insert(0, '.')
import torch
from arguments import ModelParams, PipelineParams, OptimizationParams, MaskingParams, VirtualCamParams
from scene import Scene, GaussianModel
from gaussian_renderer import render_upgrade as render_func
from gaussian_renderer import modified_render as render_fisher
from view_selection import generate_candidate_pool, export_selected_cameras

device = 'cuda'
dataset_dir = '${DATASET_DIR}'
output_base = '${OUTPUT_BASE}/${SCENE}'

# Minimal args
class Args:
    source_path = dataset_dir
    model_path = f'{output_base}/candidate_pool'
    sh_degree = 3
    images = 'images'
    depths = ''
    resolution = -1
    white_background = False
    data_device = 'cuda'
    eval = True
    random_init = False
    loader = 'Nerfbusters'
    isotropic = False
    oracle = False
    dataset_distribution = 'easy'

args = Args()
opt = type('Opt', (), {'appearance_model': '', 'inv_depth': True})()
pipe = type('Pipe', (), {
    'convert_SHs_python': False,
    'compute_cov3D_python': False,
    'debug': False,
    'antialiasing': False
})()
mask_params = type('MP', (), {
    'use_view_direction': False,
    'use_visibility': False,
    'use_visibility_mask': False,
    'use_scale_mask': False,
    'use_viewdirection_mask': False,
    'visibility_weight': False,
    'use_fisher': False,
    'fisher_current': False,
    'filter': [1, 2],
    'pruning': -1,
    'replace_gt': False,
    'use_ref': False,
    'nodepth': False
})()

# VirtualCamParams for candidate generation
cam_params = type('CP', (), {
    'search_anchor_set': ['6t', '4r', '4o'],
    'search_anchor_set_angle': 15,
    'search_anchor_set_angle_orbit': 15,
    'search_fix_right_vec': True,
    'search_fix_up_vec': True,
    'search_occ_threshold': 0.5,
    'search_reject_depth': 'mix_0.5_4.0',
    'search_bg_reject_ratio': 0.5,
    'search_mesh_dist_threshold': '1.0',
    'search_use_frontier': False,
    'search_voxel_resolution': 64,
    'search_coarse_voxel_resolution': 16,
    'search_bbox': 'freespace-obb',
    'search_mesh_res': 256,
    'search_mesh_leave_largest': 50,
    'search_skip_rejection': False,
    'search_skip_freespace_rejection': False,
})()

# Load Stage 1 model
if not os.path.exists(f'{output_base}/candidate_pool'):
    gaussians = GaussianModel(args, opt, mask_params)
    scene = Scene(args, mask_params, gaussians)
    ckpt = torch.load('${STAGE1_CKPT}')
    gaussians.restore(ckpt[0], opt, mask_params)
    del ckpt

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    bg_tensor = torch.tensor(bg_color, dtype=torch.float32, device=device)

    os.makedirs(f'{output_base}/candidate_pool', exist_ok=True)

    candidates = generate_candidate_pool(
        scene, gaussians, None, pipe, bg_tensor,
        render_func, cam_params, subsample=1
    )

    # Save candidate pool
    export_selected_cameras(candidates, f'{output_base}/candidate_pool.json')
    print(f'Candidate pool saved: {len(candidates)} cameras')
    torch.save(gaussians.capture(), f'{output_base}/candidate_pool/gaussians.pth')
else:
    print('Candidate pool already exists')
"
echo "[Candidate Pool] Done"

# ============================================================================
# 第四步：对每种策略 × 预算运行视图选择 + 训练
# ============================================================================
for STRATEGY in "${STRATEGIES[@]}"; do
    for B in "${BUDGETS[@]}"; do
        EXP_NAME="${STRATEGY}_B${B}"
        EXP_DIR="${OUTPUT_BASE}/${SCENE}/${EXP_NAME}"
        SELECTED_JSON="${OUTPUT_BASE}/${SCENE}/selected_${STRATEGY}_B${B}.json"

        echo ""
        echo "=========================================="
        echo "  Strategy: ${STRATEGY}, Budget: ${B}"
        echo "  Experiment: ${EXP_NAME}"
        echo "=========================================="

        # --- Step 4a: View Selection ---
        echo "[Step 4a] Running view selection..."
        python -c "
import sys, json, torch, numpy as np
sys.path.insert(0, '.')
from view_selection import ViewSelector
from gaussian_renderer import render_upgrade as render_func
from gaussian_renderer import modified_render as render_fisher
from scene import Scene, GaussianModel
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams, OptimizationParams, MaskingParams
from utils.camera_utils import cameraList_from_camInfos

device = 'cuda'

# Load candidate cameras
with open('${OUTPUT_BASE}/${SCENE}/candidate_pool.json') as f:
    cam_data_list = json.load(f)

candidate_cams = []
for cd in cam_data_list:
    R = np.array(cd['rotation'])
    T = np.array(cd['position'])
    cam = Camera(
        colmap_id=cd['id'], R=R, T=T,
        FoVx=cd['FoVx'], FoVy=cd['FoVy'],
        image=None, image_name=cd['img_name'],
        uid=cd['id'], trans=np.array([0,0,0]), scale=1.0,
        data_device=device, gt_alpha_mask=None)
    cam.virtual = True
    candidate_cams.append(cam)

print(f'Loaded {len(candidate_cams)} candidate cameras')

# Load model for view selection
args = type('Args', (), {
    'source_path': '${DATASET_DIR}', 'model_path': f'{EXP_DIR}_tmp',
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
mp = type('MP', (), {
    'use_view_direction': False, 'use_visibility': False,
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

# Create selector
train_cameras = scene.getTrainCameras()
selector = ViewSelector(
    gaussians=gaussians, pipe=pipe, bg_color=bg_tensor,
    render_func=render_func, render_fisher=render_fisher,
    train_cameras=train_cameras, device=device
)

# Build view directions for coverage
selector.build_view_directions(dir_angle=30)

# Run selection
selected_indices, elapsed = selector.select(
    candidate_cams, strategy='${STRATEGY}', B=${B}
)
print(f'Selection completed in {elapsed:.3f}s')
print(f'Selected indices: {selected_indices}')

# Save selected cameras
selected_cams = [candidate_cams[i] for i in selected_indices]
from view_selection import export_selected_cameras
export_selected_cameras(selected_cams, '${SELECTED_JSON}')

# Save timing
with open('${OUTPUT_BASE}/${SCENE}/timing_${STRATEGY}_B${B}.json', 'w') as f:
    json.dump({'strategy': '${STRATEGY}', 'B': ${B}, 'elapsed': elapsed,
               'K': len(candidate_cams), 'indices': selected_indices}, f)
print(f'Timing saved')
"

        # --- Step 4b: Train 3DGS with selected views ---
        echo "[Step 4b] Training 3DGS with selected views..."
        python train_selected.py \
            -s ${DATASET_DIR} \
            --eval -r 1 \
            --start_checkpoint ${STAGE1_CKPT} \
            --selected_cams ${SELECTED_JSON} \
            --iterations 30000 \
            --test_iterations 30000 \
            --save_iterations 30000 \
            --expname ${EXP_NAME} \
            --quiet

        echo "[${EXP_NAME}] Done"
    done
done

# ============================================================================
# 第五步：汇总结果
# ============================================================================
echo ""
echo "===== Collecting Results ====="
python -c "
import json, os, glob

base = '${OUTPUT_BASE}/${SCENE}'
results = {}

strategies = ['coverage', 'fisher', 'random', 'farthest']
budgets = [5, 10, 20, 40]

for strategy in strategies:
    results[strategy] = {}
    for B in budgets:
        exp_name = f'{strategy}_B{B}'
        result_file = f'{base}/results_030000.json'
        # Try finding in output structure
        alt_path = f'output/ablation/{exp_name}/${SCENE}/results_030000.json'
        if os.path.exists(alt_path):
            with open(alt_path) as f:
                data = json.load(f)
            results[strategy][B] = {
                'PSNR': data['PSNR'],
                'SSIM': data['SSIM'],
                'LPIPS': data['LPIPS'],
            }
        else:
            # Search for it
            for root, dirs, files in os.walk(f'output/ablation/{exp_name}'):
                for fn in files:
                    if fn.startswith('results_') and fn.endswith('.json'):
                        with open(os.path.join(root, fn)) as f:
                            data = json.load(f)
                        results[strategy][B] = {
                            'PSNR': data['PSNR'],
                            'SSIM': data['SSIM'],
                            'LPIPS': data['LPIPS'],
                        }
                        break

with open(f'{base}/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
" 2>&1 || echo "[Warning] Results collection had issues - check manually"

echo ""
echo "=========================================="
echo "  Ablation Study Complete!"
echo "  Results: ${OUTPUT_BASE}/${SCENE}/all_results.json"
echo "=========================================="
