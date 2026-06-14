#!/bin/bash
# ============================================================================
# run_all.sh — 全自动消融实验流程（8 GPU 并行）
#
# 用法：
#   bash scripts/run_all.sh <scene>                    # 单场景
#   bash scripts/run_all.sh                              # 所有场景
#   bash scripts/run_all.sh bicycle,aloe,flowers         # 指定场景列表
#
# 环境变量控制：
#   DATA_DIR=/path/to/dataset       默认为 data/curated_nb
#   GPU_LIST="0,1,2,3,4,5,6,7"      默认使用所有 8 块 GPU
# ============================================================================

set -e

# ---- 配置 ----
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

CONDA_ENV="exploregs"
CONDA_PATH="/home/zhongyao/.conda/envs/${CONDA_ENV}/bin"

# 可调参数
DATA_DIR="${DATA_DIR:-data/curated_nb}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"

STRATEGIES=("coverage" "fisher" "random" "farthest")
BUDGETS=(5 10 20 40)

echo "=========================================="
echo "  ExploreGS Full Ablation Pipeline"
echo "  GPUs: ${GPU_LIST}"
echo "  Data: ${DATA_DIR}"
echo "=========================================="

# ---- 解析场景列表 ----
if [ $# -ge 1 ]; then
    IFS=',' read -ra SCENES <<< "$1"
else
    SCENES=(aloe art bicycle flowers garbage picnic pipe roses table)
fi

echo "Scenes: ${SCENES[*]}"

# ---- 函数：等待至少 N 块 GPU 空闲 ----
wait_for_gpus() {
    local needed=$1
    while true; do
        local free=0
        for gpu in "${GPU_ARRAY[@]}"; do
            local mem_used=$(${CONDA_PATH}/python -c "
import subprocess
r = subprocess.run(['nvidia-smi', '--id=${gpu}', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                   capture_output=True, text=True)
print(int(r.stdout.strip()))
" 2>/dev/null)
            if [ "${mem_used}" -lt 20000 ]; then
                ((free++))
            fi
        done
        if [ "${free}" -ge "${needed}" ]; then
            break
        fi
        echo "[wait] Waiting for ${needed} free GPU(s)... (${free} available)"
        sleep 30
    done
}

# ---- 函数：在指定 GPU 上运行命令 ----
run_on_gpu() {
    local gpu=$1
    local cmd=$2
    local log=$3
    echo "[GPU ${gpu}] Starting: ${log}"
    CUDA_VISIBLE_DEVICES=${gpu} bash -c "${cmd}" > "${REPO_DIR}/output/logs/${log}.log" 2>&1 &
    echo $!
}

# 创建日志目录
mkdir -p output/logs

# ============================================================================
# Step 1: 验证数据集
# ============================================================================
echo ""
echo "===== Step 1: Verifying Dataset ====="
SCENE_OK=true
for scene in "${SCENES[@]}"; do
    if [ ! -d "${DATA_DIR}/${scene}" ]; then
        echo "[Warning] Scene ${scene} not found at ${DATA_DIR}/${scene}"
        SCENE_OK=false
    else
        echo "  ✓ ${scene}: $(ls ${DATA_DIR}/${scene}/images 2>/dev/null | wc -l) images"
    fi
done

if [ "$SCENE_OK" = false ]; then
    echo ""
    echo "Dataset incomplete. Download first:"
    echo "  bash scripts/download_data.sh"
    echo "Or set DATA_DIR to correct path."
    exit 1
fi

# ============================================================================
# Step 2: Stage 1 训练（所有场景并行）
# ============================================================================
echo ""
echo "===== Step 2: Stage 1 Training ====="
PIDS=()
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    gpu=${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}

    if [ -f "output/stage1_default/${scene}/chkpnt30000.pth" ]; then
        echo "  [GPU ${gpu}] ${scene}: checkpoint exists, skipping"
        continue
    fi

    wait_for_gpus 1
    run_on_gpu ${gpu} "
        cd ${REPO_DIR}
        source activate ${CONDA_ENV}
        ${CONDA_PATH}/python train_stage1.py \
            -s ${DATA_DIR}/${scene} \
            --eval -r 1 \
            -c configs/stage1/default.yaml \
            --load Nerfbusters
    " "stage1_${scene}"
    PIDS+=($!)
done

# 等待所有 Stage 1 完成
for pid in "${PIDS[@]}"; do
    wait ${pid} 2>/dev/null || true
done
echo "[Stage 1] All scenes complete"

# ============================================================================
# Step 3: 生成候选相机池（所有场景并行）
# ============================================================================
echo ""
echo "===== Step 3: Candidate Pool Generation ====="
PIDS=()
for i in "${!SCENES[@]}"; do
    scene="${SCENES[$i]}"
    gpu=${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}

    CKPT="output/stage1_default/${scene}/chkpnt30000.pth"
    POOL_FILE="output/ablation_study/${scene}/candidate_pool.json"

    if [ -f "${POOL_FILE}" ]; then
        echo "  [GPU ${gpu}] ${scene}: pool exists, skipping"
        continue
    fi

    wait_for_gpus 1
    run_on_gpu ${gpu} "
        cd ${REPO_DIR}
        ${CONDA_PATH}/python -c \"
import sys, torch, json, os
sys.path.insert(0, '.')
from view_selection import generate_candidate_pool, export_selected_cameras
from gaussian_renderer import render_upgrade as render_func
from scene import Scene, GaussianModel

device = 'cuda'
dataset_dir = '${DATA_DIR}/${scene}'
ckpt_path = '${CKPT}'

args = type('A', (), {'source_path': dataset_dir, 'model_path': '/tmp/pool_${scene}',
    'sh_degree': 3, 'images': 'images', 'depths': '', 'resolution': -1,
    'white_background': False, 'data_device': 'cuda', 'eval': True,
    'random_init': False, 'loader': 'Nerfbusters',
    'isotropic': False, 'oracle': False, 'dataset_distribution': 'easy',
})()

opt = type('O', (), {'appearance_model': '', 'inv_depth': True})()
pipe = type('P', (), {'convert_SHs_python': False, 'compute_cov3D_python': False,
    'debug': False, 'antialiasing': False})()

mp = type('M', (), {'use_view_direction': False, 'use_visibility': False,
    'use_visibility_mask': False, 'use_scale_mask': False,
    'use_viewdirection_mask': False, 'visibility_weight': False,
    'use_fisher': False, 'fisher_current': False,
    'filter': [1,2], 'pruning': -1, 'replace_gt': False,
    'use_ref': False, 'nodepth': False,
})()

cam_params = type('C', (), {
    'search_anchor_set': ['6t','4r','4o'],
    'search_anchor_set_angle': 15, 'search_anchor_set_angle_orbit': 15,
    'search_fix_right_vec': True, 'search_fix_up_vec': True,
    'search_occ_threshold': 0.5, 'search_reject_depth': 'mix_0.5_4.0',
    'search_bg_reject_ratio': 0.5,
})()

gaussians = GaussianModel(args, opt, mp)
scene = Scene(args, mp, gaussians)
gaussians.restore(torch.load(ckpt_path)[0], opt, mp)

bg = [1,1,1] if args.white_background else [0,0,0]
bg_t = torch.tensor(bg, dtype=torch.float32, device=device)
os.makedirs('output/ablation_study/${scene}', exist_ok=True)

candidates = generate_candidate_pool(scene, gaussians, None, pipe, bg_t, render_func, cam_params, subsample=2)
export_selected_cameras(candidates, 'output/ablation_study/${scene}/candidate_pool.json')
print(f'Generated {len(candidates)} candidates for ${scene}')
\"
    " "pool_${scene}"
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait ${pid} 2>/dev/null || true
done
echo "[Pool] All pools generated"

# ============================================================================
# Step 4: 视图选择 + Stage 2 训练（策略×预算组合，并行运行）
# ============================================================================
echo ""
echo "===== Step 4: View Selection + Stage 2 Training ====="

# 为每个场景分别运行
for scene in "${SCENES[@]}"; do
    echo ""
    echo "--- Processing scene: ${scene} ---"
    CKPT="output/stage1_default/${scene}/chkpnt30000.pth"
    POOL_FILE="output/ablation_study/${scene}/candidate_pool.json"
    PIDS=()

    CMD_IDX=0
    for strategy in "${STRATEGIES[@]}"; do
        for B in "${BUDGETS[@]}"; do
            EXP_NAME="${strategy}_B${B}"
            GPU_IDX=$((CMD_IDX % ${#GPU_ARRAY[@]}))
            gpu=${GPU_ARRAY[${GPU_IDX}]}
            CMD_IDX=$((CMD_IDX + 1))

            wait_for_gpus 1
            run_on_gpu ${gpu} "
                cd ${REPO_DIR}
                ${CONDA_PATH}/python -c \"
import sys, torch, json, numpy as np
sys.path.insert(0, '.')
from view_selection import ViewSelector, export_selected_cameras
from gaussian_renderer import render_upgrade as render_func
from gaussian_renderer import modified_render as render_fisher
from scene import Scene, GaussianModel
from scene.cameras import Camera

device = 'cuda'
dataset_dir = '${DATA_DIR}/${scene}'
ckpt_path = '${CKPT}'
pool_path = '${POOL_FILE}'

# 加载候选相机
with open(pool_path) as f:
    cam_data_list = json.load(f)
candidate_cams = []
for cd in cam_data_list:
    R, T = np.array(cd['rotation']), np.array(cd['position'])
    cam = Camera(colmap_id=cd['id'], R=R, T=T, FoVx=cd['FoVx'], FoVy=cd['FoVy'],
        image=None, image_name=cd['img_name'], uid=cd['id'],
        trans=np.zeros(3), scale=1.0, data_device=device, gt_alpha_mask=None)
    cam.virtual = True
    candidate_cams.append(cam)

args = type('A', (), {'source_path': dataset_dir, 'model_path': '/tmp/ablation_sel',
    'sh_degree': 3, 'images': 'images', 'depths': '', 'resolution': -1, 'white_background': False,
    'data_device': 'cuda', 'eval': True, 'random_init': False, 'loader': 'Nerfbusters',
    'isotropic': False, 'oracle': False, 'dataset_distribution': 'easy'})()
opt = type('O', (), {'appearance_model': '', 'inv_depth': True})()
pipe = type('P', (), {'convert_SHs_python': False, 'compute_cov3D_python': False,
    'debug': False, 'antialiasing': False})()
mp = type('M', (), {'use_view_direction': False, 'use_visibility': False,
    'use_visibility_mask': False, 'use_scale_mask': False, 'use_viewdirection_mask': False,
    'visibility_weight': False, 'use_fisher': False, 'fisher_current': False,
    'filter': [1,2], 'pruning': -1, 'replace_gt': False, 'use_ref': False, 'nodepth': False
})()

gaussians = GaussianModel(args, opt, mp)
scene_obj = Scene(args, mp, gaussians)
gaussians.restore(torch.load(ckpt_path)[0], opt, mp)

bg_t = torch.tensor([0,0,0], dtype=torch.float32, device=device)
train_cams = scene_obj.getTrainCameras()
selector = ViewSelector(gaussians, pipe, bg_t, render_func, render_fisher, train_cams, device=device)
selector.build_view_directions(dir_angle=30)

selected, elapsed = selector.select(candidate_cams, strategy='${strategy}', B=${B})
export_selected_cameras([candidate_cams[i] for i in selected],
    'output/ablation_study/${scene}/selected_${strategy}_B${B}.json')
import json as j2
with open('output/ablation_study/${scene}/timing_${strategy}_B${B}.json', 'w') as f:
    j2.dump({'strategy': '${strategy}', 'B': ${B}, 'K': len(candidate_cams), 'elapsed': elapsed}, f)
print(f'Selection done: {${strategy}} B=${B} in {elapsed:.2f}s')
\"

            # Stage 2 训练
            ${CONDA_PATH}/python train_selected.py \
                -s ${DATA_DIR}/${scene} \
                --eval -r 1 \
                --start_checkpoint ${CKPT} \
                --selected_cams output/ablation_study/${scene}/selected_${strategy}_B${B}.json \
                --iterations 30000 \
                --test_iterations 30000 \
                --save_iterations 30000 \
                --expname ${EXP_NAME} \
                --quiet
            " "select+train_${scene}_${EXP_NAME}"
            PIDS+=($!)
        done
    done

    # 等待当前场景的所有任务完成
    for pid in "${PIDS[@]}"; do
        wait ${pid} 2>/dev/null || true
    done
    echo "[${scene}] All experiments complete"
done

# ============================================================================
# Step 5: 汇总结果 + 生成图表
# ============================================================================
echo ""
echo "===== Step 5: Collecting Results & Generating Figures ====="

${CONDA_PATH}/python -c "
import json, os, glob

base = 'output/ablation_study'
results = {}
strategies = ${STRATEGIES[*]}
budgets = ${BUDGETS[*]}

for scene in ${SCENES[*]}:
    for strategy in strategies:
        if strategy not in results:
            results[strategy] = {}
        for B in budgets:
            exp_name = f'{strategy}_B{B}'
            pattern = f'output/ablation/{exp_name}/{scene}/results_*.json'
            files = glob.glob(pattern)
            if files:
                with open(files[0]) as f:
                    data = json.load(f)
                results[strategy][B] = {
                    'PSNR': data['PSNR'],
                    'SSIM': data['SSIM'],
                    'LPIPS': data['LPIPS'],
                }

    # 每个场景保存一份
    os.makedirs(f'{base}/{scene}', exist_ok=True)
    with open(f'{base}/{scene}/all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

print('All results collected')
"

# 生成图表
${CONDA_PATH}/python scripts/generate_figures.py \
    --results_dir output/ablation_study/${SCENES[0]} \
    --output_dir output/figures

echo ""
echo "=========================================="
echo "  Full Pipeline Complete!"
echo "=========================================="
echo "  Results:    output/ablation_study/"
echo "  Figures:    output/figures/"
echo "  Logs:       output/logs/"
echo "=========================================="
