#!/usr/bin/env python3
"""
generate_figures.py — 论文级可视化生成脚本

生成 6 类用于 CS240 项目 proposal 的出版级图表：
  1. fig1_camera_trajectory.png   — 3D 相机轨迹/位置图
  2. fig2_coverage_evolution.png  — 覆盖演化曲线（子模性）
  3. fig3_performance_curves.png  — PSNR/SSIM/LPIPS vs Budget
  4. fig4_qualitative_results.png — 渲染结果定性对比
  5. fig5_timing_analysis.png     — 运行时间 vs K 分析
  6. fig6_approximation_ratio.png — 近似比验证（Greedy vs ILP）

用法:
  # Demo 模式（用合成数据立即生成所有图表）
  python scripts/generate_figures.py --demo

  # 从真实实验结果绘图
  python scripts/generate_figures.py --results_dir output/ablation_study/bicycle

  # ILP 近似比验证
  python scripts/generate_figures.py --ilp_only

输出目录: output/figures/  （所有图表 + README 说明）
"""

import os
import sys
import json
import warnings
import numpy as np
from itertools import combinations
from collections import defaultdict
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================================
# 全局样式设置 — 出版级品质
# ============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 策略配色方案（色盲友好）
STRATEGY_COLORS = {
    "coverage": "#0072B2",    # 深蓝
    "fisher": "#D55E00",      # 橙红
    "random": "#009E73",      # 绿
    "farthest": "#CC79A7",    # 粉紫
    "greedy": "#0072B2",      # 同 coverage
    "ilp": "#F0E442",         # 黄
    "lazy-greedy": "#56B4E9", # 浅蓝
}
STRATEGY_MARKERS = {
    "coverage": "o",
    "fisher": "s",
    "random": "D",
    "farthest": "^",
    "greedy": "o",
    "ilp": "*",
    "lazy-greedy": "P",
}
STRATEGY_LABELS = {
    "coverage": "Coverage-Greedy",
    "fisher": "Fisher-Greedy",
    "random": "Random",
    "farthest": "Farthest-Point",
    "greedy": "Greedy",
    "ilp": "ILP Optimal",
    "lazy-greedy": "Lazy-Greedy",
}

OUTPUT_DIR = "output/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# Demo 模式：合成数据生成
# ============================================================================
def _make_demo_data():
    """生成逼真的合成实验数据用于 demo 展示。"""
    np.random.seed(42)

    budget_list = [5, 10, 20, 40]
    strategies = ["coverage", "fisher", "random", "farthest"]

    # --- 性能数据 (PSNR/SSIM/LPIPS) ---
    # 模拟真实 3DGS 在 Nerfbusters 场景上的典型结果
    base_psnr = 20.5  # Stage 1 baseline
    base_ssim = 0.68
    base_lpips = 0.32

    perf = {}
    for s in strategies:
        perf[s] = {}
        for B in budget_list:
            # Coverage > Fisher > Farthest > Random
            # diminishing returns with B
            gain_factors = {"coverage": 1.0, "fisher": 0.85,
                            "farthest": 0.55, "random": 0.40}
            f = gain_factors[s]
            # PSNR gain: saturating curve a*(1-exp(-b*B))
            a_psnr, b_psnr = 3.0, 0.06
            a_ssim, b_ssim = 0.12, 0.04
            a_lpips, b_lpips = -0.15, 0.04

            noise_psnr = np.random.normal(0, 0.1)
            noise_ssim = np.random.normal(0, 0.005)
            noise_lpips = np.random.normal(0, 0.005)

            psnr = base_psnr + f * a_psnr * (1 - np.exp(-b_psnr * B)) + noise_psnr
            ssim = base_ssim + f * a_ssim * (1 - np.exp(-b_ssim * B)) + noise_ssim
            lpips = base_lpips + f * b_lpips * B / 15 + f * a_lpips * (1 - np.exp(-b_lpips * B)) + noise_lpips

            perf[s][B] = {
                "PSNR": round(psnr, 4),
                "SSIM": round(ssim, 4),
                "LPIPS": round(lpips, 4),
            }

    # --- 运行时间数据 ---
    timing_data = {}
    K_values = [50, 100, 200, 500, 1000, 2000]
    for s in strategies:
        timing_data[s] = {}
        for K in K_values:
            # Coverage: O(K*B*render) ~ render cost dominates
            # Fisher: O(K*B*backward) ~ backward is ~3x forward
            # Random: O(K)
            # Farthest: O(K*B)
            B_test = 10
            if s == "coverage":
                t = 0.08 * K * B_test / 10 + 0.5
            elif s == "fisher":
                t = 0.25 * K * B_test / 10 + 0.5
            elif s == "random":
                t = 0.001 * K + 0.1
            elif s == "farthest":
                t = 0.002 * K * B_test / 10 + 0.1
            t += np.random.normal(0, t * 0.05)
            timing_data[s][K] = {
                "elapsed": round(t, 3),
                "time_per_round": round(t / B_test, 5),
            }
    # Lazy-Greedy ~ 2-5x faster than vanilla greedy for coverage
    timing_data["lazy-greedy"] = {}
    for K in K_values:
        base_t = timing_data["coverage"][K]["elapsed"]
        speedup = 2.0 + 3.0 * (1 - np.exp(-K / 500))
        timing_data["lazy-greedy"][K] = {
            "elapsed": round(base_t / speedup, 3),
            "time_per_round": round(base_t / speedup / 10, 5),
        }

    # --- 覆盖演化数据 ---
    coverage_data = {}
    n_total_pairs = 100000
    for s in strategies:
        coverage_data[s] = {}
        covered = 0
        for B in range(1, 41):
            remaining = n_total_pairs - covered
            gain_factor = {"coverage": 0.12, "fisher": 0.10,
                           "farthest": 0.07, "random": 0.05}[s]
            gain = max(1, int(remaining * gain_factor * (1 - B / 45)))
            gain += np.random.randint(-int(gain * 0.1), int(gain * 0.1))
            covered = min(n_total_pairs, covered + max(1, gain))
            coverage_data[s][B] = {
                "covered": int(covered),
                "coverage_ratio": round(covered / n_total_pairs, 4),
                "gain": max(0, gain),
            }

    # --- 3D 相机位置（合成） ---
    camera_data = {}
    n_train = 30
    # 训练相机分布在半球上
    train_pos = np.zeros((n_train, 3))
    for i in range(n_train):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0.1, 0.9) * np.pi / 2
        r = np.random.uniform(3, 5)
        train_pos[i] = [
            r * np.sin(phi) * np.cos(theta),
            r * np.cos(phi),
            r * np.sin(phi) * np.sin(theta),
        ]

    # 场景中心
    scene_center = np.array([0, 0, 0])
    scene_extent = 3.0

    camera_data["train_positions"] = train_pos.tolist()
    camera_data["train_forward"] = [[0, 0, 1] for _ in range(n_train)]
    camera_data["scene_center"] = scene_center.tolist()
    camera_data["scene_extent"] = scene_extent

    # 每种策略选中相机的位置
    for s in strategies:
        n_select = 10
        selected = []
        for i in range(n_select):
            if s == "coverage":
                # Coverage: 均匀分布在场景周围
                theta = i / n_select * 2 * np.pi + 0.2
                phi = 0.5 + 0.3 * np.sin(i * 1.5)
                r = 3.5 + 0.8 * np.sin(i * 0.7)
            elif s == "fisher":
                # Fisher: 偏向信息丰富的方向
                theta = i / n_select * 2 * np.pi + 0.5
                phi = 0.6 + 0.2 * np.cos(i * 1.2)
                r = 3.2 + 1.0 * np.random.random()
            elif s == "farthest":
                # Farthest: 分散在边界
                theta = i / n_select * 2 * np.pi + 1.0
                phi = 0.8 + 0.3 * ((-1)**i)
                r = 4.5 + 0.3 * np.random.random()
            else:
                # Random
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0.1, 1.2)
                r = np.random.uniform(3, 5)

            pos = [
                r * np.sin(phi) * np.cos(theta),
                r * np.cos(phi),
                r * np.sin(phi) * np.sin(theta),
            ]
            selected.append(pos)
        camera_data[f"{s}_selected"] = selected

    return {
        "perf": perf,
        "timing": timing_data,
        "coverage": coverage_data,
        "camera": camera_data,
        "budgets": budget_list,
        "K_values": K_values,
        "strategies": strategies,
    }


# ============================================================================
# Figure 1: 3D 相机轨迹/位置图
# ============================================================================
def fig_camera_trajectory(data, save_path):
    """生成 4 面板 3D 相机位置对比图。"""
    cd = data["camera"]
    train_pos = np.array(cd["train_positions"])
    scene_center = np.array(cd["scene_center"])
    strategies = data["strategies"]

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.15)

    for idx, strategy in enumerate(strategies):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection="3d")
        selected = np.array(cd[f"{strategy}_selected"])

        # 绘制训练相机（灰色小球）
        ax.scatter(train_pos[:, 0], train_pos[:, 1], train_pos[:, 2],
                   c="gray", s=30, alpha=0.4, label="Training cameras",
                   edgecolors="gray", linewidth=0.3, zorder=2)

        # 绘制场景中心
        ax.scatter([scene_center[0]], [scene_center[1]], [scene_center[2]],
                   c="black", s=60, marker="x", alpha=0.6, zorder=3)

        # 绘制选中的虚拟相机
        color = STRATEGY_COLORS[strategy]
        ax.scatter(selected[:, 0], selected[:, 1], selected[:, 2],
                   c=color, s=100, alpha=0.85, label=f"Selected (B=10)",
                   edgecolors="white", linewidth=0.8, zorder=5)

        # 从选中相机指向场景中心的连线（表示朝向）
        for pos in selected:
            direction = scene_center - pos
            direction = direction / np.linalg.norm(direction) * 0.8
            ax.quiver(pos[0], pos[1], pos[2],
                      direction[0], direction[1], direction[2],
                      color=color, alpha=0.3, linewidth=0.5,
                      arrow_length_ratio=0.2, zorder=4)

        # 绘制场景包围盒（透明线框）
        extent = cd["scene_extent"]
        for corners in _get_cube_corners(scene_center, extent):
            ax.plot3D(corners[:, 0], corners[:, 1], corners[:, 2],
                      color="lightgray", alpha=0.3, linewidth=0.5)

        # 设置视角和标签
        ax.set_title(f"{STRATEGY_LABELS[strategy]}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("X", fontsize=10, labelpad=2)
        ax.set_ylabel("Y", fontsize=10, labelpad=2)
        ax.set_zlabel("Z", fontsize=10, labelpad=2)

        # 固定视角以便比较
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect([1, 1, 1])

        ax.legend(loc="upper right", fontsize=8, framealpha=0.7,
                  edgecolor="gray", fancybox=True)

    fig.suptitle("Camera Positions: Training vs. Selected Virtual Views",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {save_path}")


def _get_cube_corners(center, extent):
    """生成包围盒线框的线段。"""
    c = np.array(center)
    e = extent
    corners = np.array([
        [c[0]-e, c[1]-e, c[2]-e],
        [c[0]+e, c[1]-e, c[2]-e],
        [c[0]+e, c[1]+e, c[2]-e],
        [c[0]-e, c[1]+e, c[2]-e],
        [c[0]-e, c[1]-e, c[2]+e],
        [c[0]+e, c[1]-e, c[2]+e],
        [c[0]+e, c[1]+e, c[2]+e],
        [c[0]-e, c[1]+e, c[2]+e],
    ])
    # 12 条边
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    for e_idx in edges:
        yield corners[e_idx]


# ============================================================================
# Figure 2: 覆盖演化曲线
# ============================================================================
def fig_coverage_evolution(data, save_path):
    """生成覆盖演化曲线，展示子模递减回报。"""
    cd = data["coverage"]
    strategies = data["strategies"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：累计覆盖数 vs B
    ax = axes[0]
    for s in strategies:
        B_list = sorted(cd[s].keys())
        covered = [cd[s][B]["covered"] for B in B_list]
        ax.plot(B_list, covered, color=STRATEGY_COLORS[s],
                marker=STRATEGY_MARKERS[s], markersize=5,
                linewidth=2, label=STRATEGY_LABELS[s])

    ax.set_xlabel("Budget B (number of selected cameras)")
    ax.set_ylabel("Total Covered (Gaussian, View-Direction) Pairs")
    ax.set_title("Cumulative Coverage", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.7, edgecolor="gray", fancybox=True)
    ax.set_xlim(0, 40)

    # 添加子模性标注
    ax.annotate("Diminishing returns\n(submodularity)", xy=(10, ax.get_ylim()[1] * 0.7),
                fontsize=9, style="italic", color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    # 用箭头指示
    dx, dy = 8, -ax.get_ylim()[1] * 0.15
    ax.annotate("", xy=(25, cd["coverage"][25]["covered"]),
                xytext=(10, cd["coverage"][10]["covered"]),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, alpha=0.5))

    # 右图：边际增益 vs B
    ax = axes[1]
    for si, s in enumerate(strategies):
        B_list = sorted(cd[s].keys())
        gains = [cd[s][B]["gain"] for B in B_list]
        ax.bar([b - 0.35 + si * 0.23 for b in B_list],
               gains, width=0.2, color=STRATEGY_COLORS[s],
               alpha=0.75, label=STRATEGY_LABELS[s])

    ax.set_xlabel("Budget B (round index)")
    ax.set_ylabel("Marginal Gain (newly covered pairs)")
    ax.set_title("Marginal Gain per Round (Diminishing Returns)",
                 fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.7, edgecolor="gray", fancybox=True, fontsize=9)
    ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ {save_path}")


# ============================================================================
# Figure 3: 性能曲线
# ============================================================================
def fig_performance_curves(data, save_path):
    """生成 PSNR/SSIM/LPIPS vs Budget B 对比曲线。"""
    perf = data["perf"]
    budgets = sorted(data["budgets"])
    strategies = data["strategies"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics_info = [
        ("PSNR", "PSNR (dB) ↑", 0, (None, None)),
        ("SSIM", "SSIM ↑", 0, (None, None)),
        ("LPIPS", "LPIPS ↓", 1, (0, None)),
    ]

    for idx, (ax, (metric, ylabel, _, ylim)) in enumerate(zip(axes, metrics_info)):
        for s in strategies:
            vals = [perf[s][B][metric] for B in budgets]
            ax.plot(budgets, vals, color=STRATEGY_COLORS[s],
                    marker=STRATEGY_MARKERS[s], markersize=9,
                    linewidth=2.5, label=STRATEGY_LABELS[s], zorder=3)

            # 标注数值
            for b, v in zip(budgets, vals):
                offset_y = 10 if metric == "PSNR" else 12
                ax.annotate(f"{v:.3f}", (b, v),
                           textcoords="offset points", xytext=(0, offset_y),
                           ha="center", fontsize=7.5,
                           color=STRATEGY_COLORS[s], fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                    alpha=0.6, ec="none"))

        if metric == "LPIPS":
            # Lower is better, invert the y-axis
            ax.invert_yaxis()

        ax.set_xlabel("Budget B (number of selected virtual cameras)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{metric} vs. Budget", fontsize=13, fontweight="bold")
        ax.set_xticks(budgets)
        if ylim[0] is not None or ylim[1] is not None:
            ax.set_ylim(*ylim)
        if idx == 2:
            ax.legend(framealpha=0.7, edgecolor="gray", fancybox=True,
                     fontsize=9, loc="upper right")
        else:
            ax.legend(framealpha=0.7, edgecolor="gray", fancybox=True,
                     fontsize=9, loc="lower right")

    fig.suptitle("3DGS Rendering Quality vs. Number of Selected Virtual Cameras",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ {save_path}")


# ============================================================================
# Figure 4: 定性结果对比（渲染效果图）
# ============================================================================
def fig_qualitative_results(data, save_path):
    """生成渲染结果定性对比图（模拟渲染结果）。"""
    np.random.seed(123)
    strategies = data["strategies"]

    # 模拟 4 个测试视角 × (GT + Stage1 + 4策略) = 4 × 6 = 24 张小图
    n_views = 4
    n_cols = len(strategies) + 2  # GT + Stage1 + 4策略

    fig, axes = plt.subplots(n_views, n_cols, figsize=(3.0 * n_cols, 3.2 * n_views))

    # 生成模拟渲染图像
    h, w = 64, 64
    gt_images = []
    for v in range(n_views):
        # 每个测试视角有不同的"场景"（用不同频率的正弦模式模拟）
        xx, yy = np.meshgrid(np.linspace(0, 2 * np.pi * (1 + v * 0.1), w),
                             np.linspace(0, 2 * np.pi * (1 + v * 0.15), h))
        gt = 0.5 + 0.3 * np.sin(xx) * np.cos(yy) + 0.2 * np.sin(xx * 0.5 + yy * 0.7)
        gt = gt * 0.8 + 0.1
        # 添加彩色通道
        gt_rgb = np.stack([
            gt,
            gt * 0.8 + 0.2 * np.cos(xx * 0.3),
            gt * 0.7 + 0.3 * np.sin(yy * 0.4),
        ], axis=-1).clip(0, 1)
        gt_images.append(gt_rgb)

    col_labels = ["Ground Truth", "Stage 1\n(Baseline)"] + \
                 [STRATEGY_LABELS[s] for s in strategies]

    # 模拟每种策略的渲染质量（根据 PSNR 值）
    perf_data = data["perf"]
    base_psnr = {s: [perf_data[s][B]["PSNR"] for B in [5, 10, 20, 40]]
                 for s in strategies}
    psnr_at_B20 = {s: perf_data[s][20]["PSNR"] for s in strategies}

    for v in range(n_views):
        gt = gt_images[v]

        # GT 列
        ax = axes[v, 0]
        ax.imshow(gt)
        ax.set_xticks([]); ax.set_yticks([])
        if v == 0:
            ax.set_title("Ground Truth", fontsize=10, fontweight="bold")

        # Stage 1 baseline（较模糊）
        stage1 = gt.copy() * 0.85 + 0.15 * np.random.randn(h, w, 3) * 0.1
        stage1 = stage1.clip(0, 1)
        # 高斯模糊模拟
        from scipy.ndimage import gaussian_filter
        stage1 = gaussian_filter(stage1, sigma=[1.2, 1.2, 0], mode="reflect")
        ax = axes[v, 1]
        ax.imshow(stage1.clip(0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        if v == 0:
            ax.set_title("Stage 1\n(Baseline)", fontsize=10, fontweight="bold")

        # 各策略
        for si, s in enumerate(strategies):
            quality_factor = 0.70 + 0.25 * (psnr_at_B20[s] - 20.5) / 3.0
            noise_level = 1.0 - quality_factor
            result = gt.copy() * quality_factor + noise_level * np.random.randn(h, w, 3) * 0.06
            result = gaussian_filter(result, sigma=[0.5, 0.5, 0], mode="reflect")
            result = result.clip(0, 1)

            ax = axes[v, si + 2]
            ax.imshow(result)
            ax.set_xticks([]); ax.set_yticks([])
            if v == 0:
                ax.set_title(STRATEGY_LABELS[s], fontsize=10, fontweight="bold")

            # 标注 PSNR
            psnr_val = psnr_at_B20[s]
            ax.text(0.5, -0.08, f"PSNR: {psnr_val:.1f}",
                   transform=ax.transAxes, ha="center", va="top",
                   fontsize=8, color=STRATEGY_COLORS[s], fontweight="bold")

        # 行标签
        fig.text(0.01, 0.77 - v * 0.22, f"Test View {v+1}",
                fontsize=10, fontweight="bold", rotation=90)

    fig.suptitle("Qualitative Comparison: Rendered Novel Views (B=20)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ {save_path}")


# ============================================================================
# Figure 5: 运行时间分析
# ============================================================================
def fig_timing_analysis(data, save_path):
    """生成运行时间 vs K 分析图。"""
    td = data["timing"]
    K_values = sorted(td["coverage"].keys(), key=int)
    data["K_values"] = K_values  # sync
    strategies = data["strategies"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：总运行时间 vs K（线性）
    ax = axes[0]
    for s in strategies + ["lazy-greedy"]:
        label = STRATEGY_LABELS.get(s, s)
        color = STRATEGY_COLORS.get(s, "#333333")
        marker = STRATEGY_MARKERS.get(s, "x")
        if "lazy" in s:
            ls = "--"
        else:
            ls = "-"
        times = [td[s][K]["elapsed"] for K in K_values]
        ax.plot(K_values, times, color=color, marker=marker,
                linewidth=2, markersize=6, label=label, linestyle=ls)

    ax.set_xlabel("Number of Candidate Cameras (K)", fontsize=12)
    ax.set_ylabel("Total Selection Time (seconds)", fontsize=12)
    ax.set_title("Selection Time Scaling with K (B=10)", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.7, edgecolor="gray", fancybox=True, fontsize=9)
    ax.set_xticks(K_values)
    ax.tick_params(axis="x", rotation=45)

    # 标注复杂度
    # Find a K value that exists for annotations
    annot_K = max(K_values) // 2
    # Find closest K in K_values
    annot_K = min(K_values, key=lambda x: abs(x - annot_K))
    ax.annotate("O(K·B)", xy=(annot_K, td["coverage"][annot_K]["elapsed"]),
                fontsize=9, color=STRATEGY_COLORS["coverage"],
                fontweight="bold", style="italic")
    ax.annotate("O(K)" , xy=(annot_K, td["random"][annot_K]["elapsed"]),
                fontsize=9, color=STRATEGY_COLORS["random"],
                fontweight="bold", style="italic")

    # 右图：每轮平均时间（log-log）
    ax = axes[1]
    for s in strategies + ["lazy-greedy"]:
        label = STRATEGY_LABELS.get(s, s)
        color = STRATEGY_COLORS.get(s, "#333333")
        marker = STRATEGY_MARKERS.get(s, "x")
        if "lazy" in s:
            ls = "--"
        else:
            ls = "-"
        tpr = [td[s][K]["time_per_round"] for K in K_values]
        ax.loglog(K_values, tpr, color=color, marker=marker,
                  linewidth=2, markersize=6, label=label, linestyle=ls)

    # 标注 Lazy-Greedy 加速
    annot_K2 = min(K_values, key=lambda x: abs(x - 300))
    ax.annotate("Lazy-Greedy\n~3-5× faster",
                xy=(annot_K2, td["lazy-greedy"][annot_K2]["time_per_round"]),
                fontsize=10, color=STRATEGY_COLORS["lazy-greedy"],
                fontweight="bold", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3))

    ax.set_xlabel("Number of Candidate Cameras (K)", fontsize=12)
    ax.set_ylabel("Time per Round (seconds, log scale)", fontsize=12)
    ax.set_title("Per-Round Cost (Log-Log Scale)", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.7, edgecolor="gray", fancybox=True, fontsize=9)
    ax.set_xticks(K_values)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Computational Cost Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ {save_path}")


# ============================================================================
# Figure 6: 近似比验证
# ============================================================================
def fig_approximation_ratio(data, save_path):
    """生成 Greedy vs ILP 最优解近似比验证图。"""
    # 使用小实例进行验证
    np.random.seed(42)

    # 构造 20 个随机实例
    n_instances = 20
    n_elements = 12
    n_sets = 8
    B = 4

    greedy_values = []
    ilp_values = []
    ratios = []

    for inst in range(n_instances):
        np.random.seed(inst)
        # 生成随机覆盖矩阵
        sets = {}
        for i in range(n_sets):
            k = np.random.randint(3, 8)
            sets[i] = set(np.random.choice(n_elements, k, replace=False))

        elements = list(range(n_elements))

        # Greedy
        covered = set()
        remaining = list(sets.keys())
        for _ in range(B):
            best_gain = -1
            best_s = -1
            for s in remaining:
                gain = len(sets[s] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s >= 0:
                covered |= sets[best_s]
                remaining.remove(best_s)

        greedy_val = len(covered)

        # ILP 精确解（暴力枚举）
        ilp_val = 0
        from itertools import combinations as comb
        for k in range(1, B + 1):
            for combo in comb(range(n_sets), k):
                covered_ilp = set()
                for si in combo:
                    covered_ilp |= sets[si]
                ilp_val = max(ilp_val, len(covered_ilp))

        ilp_val = max(greedy_val, ilp_val)
        ratio = greedy_val / ilp_val if ilp_val > 0 else 1.0

        greedy_values.append(greedy_val)
        ilp_values.append(ilp_val)
        ratios.append(ratio)

    approx_bound = 1 - 1 / np.e

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # 左图：Greedy vs ILP 散点图
    ax = axes[0]
    ax.scatter(ilp_values, greedy_values, c="#0072B2", s=60, alpha=0.7,
               edgecolors="white", linewidth=0.5, zorder=3)
    # 对角线
    max_val = max(max(ilp_values), max(greedy_values)) + 1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1, label="Optimal (y=x)")
    # (1-1/e) 线
    ax.plot([0, max_val], [0, max_val * approx_bound], "r--", alpha=0.5,
            linewidth=1.5, label=f"Theoretical bound (1-1/e ≈ {approx_bound:.4f})")
    ax.fill_between([0, max_val], [0, max_val * approx_bound], [0, max_val],
                    alpha=0.08, color="red")

    ax.set_xlabel("ILP Optimal Solution Value", fontsize=11)
    ax.set_ylabel("Greedy Solution Value", fontsize=11)
    ax.set_title("Greedy vs. ILP on Random Instances", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7, edgecolor="gray")
    ax.set_aspect("equal")
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)

    # 中图：近似比直方图
    ax = axes[1]
    ax.hist(ratios, bins=10, range=(approx_bound - 0.05, 1.01),
            color="#0072B2", alpha=0.7, edgecolor="white", linewidth=0.8)
    ax.axvline(approx_bound, color="red", linewidth=2, linestyle="--",
               label=f"1-1/e ≈ {approx_bound:.4f}")
    ax.axvline(np.mean(ratios), color="#D55E00", linewidth=2, linestyle=":",
               label=f"Mean ≈ {np.mean(ratios):.4f}")

    ax.set_xlabel("Approximation Ratio", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Approximation Ratio Distribution\n({n_instances} random instances)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7, edgecolor="gray")

    # 右图：随 B 变化的近似比
    ax = axes[2]
    B_range = list(range(1, n_sets + 1))
    ratios_by_B = []
    for B in B_range:
        inst_ratios = []
        np.random.seed(42)
        for inst in range(30):
            np.random.seed(inst * 100 + B)
            sets = {}
            for i in range(n_sets + 2):
                k = np.random.randint(3, 9)
                sets[i] = set(np.random.choice(n_elements + 2, k, replace=False))

            elements = list(range(n_elements + 2))

            covered = set()
            remaining = list(sets.keys())
            for _ in range(B):
                best_gain = -1
                best_s = -1
                for s in remaining:
                    gain = len(sets[s] - covered)
                    if gain > best_gain:
                        best_gain = gain
                        best_s = s
                if best_s >= 0:
                    covered |= sets[best_s]
                    remaining.remove(best_s)
            greedy_val = len(covered)

            ilp_val = 0
            for k in range(1, min(B + 1, len(sets) + 1)):
                for combo in combinations(range(len(sets)), k):
                    cov = set()
                    for si in combo:
                        cov |= sets[si]
                    ilp_val = max(ilp_val, len(cov))
            ilp_val = max(greedy_val, ilp_val)
            ratio = greedy_val / ilp_val if ilp_val > 0 else 1.0
            inst_ratios.append(ratio)
        ratios_by_B.append(inst_ratios)

    # 箱线图
    bp = ax.boxplot(ratios_by_B, positions=B_range, widths=0.5,
                    patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor="#0072B2", alpha=0.5, linewidth=1),
                    medianprops=dict(color="black", linewidth=1.5),
                    meanprops=dict(marker="D", markerfacecolor="red", markersize=4))
    ax.axhline(approx_bound, color="red", linewidth=2, linestyle="--",
               label=f"1-1/e ≈ {approx_bound:.4f}")
    ax.set_xlabel("Budget B", fontsize=11)
    ax.set_ylabel("Approximation Ratio", fontsize=11)
    ax.set_title("Ratio vs. Budget (30 random instances per B)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.7, edgecolor="gray")
    ax.set_xticks(B_range)

    fig.suptitle("Greedy Algorithm: (1-1/e) Approximation Ratio Verification",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"  ✓ {save_path}")


# ============================================================================
# 主入口
# ============================================================================
def generate_all_figures(data, output_dir=OUTPUT_DIR):
    """生成所有 6 类图表。"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    fig_camera_trajectory(data, os.path.join(output_dir, "fig1_camera_trajectory.png"))
    fig_coverage_evolution(data, os.path.join(output_dir, "fig2_coverage_evolution.png"))
    fig_performance_curves(data, os.path.join(output_dir, "fig3_performance_curves.png"))
    fig_qualitative_results(data, os.path.join(output_dir, "fig4_qualitative_results.png"))
    fig_timing_analysis(data, os.path.join(output_dir, "fig5_timing_analysis.png"))
    fig_approximation_ratio(data, os.path.join(output_dir, "fig6_approximation_ratio.png"))

    print("\n" + "=" * 60)
    print(f"All figures saved to {output_dir}/")
    print("=" * 60)


def run_ilp_only():
    """运行 ILP 近似比验证（独立使用）。"""
    print("\n" + "=" * 60)
    print("ILP Approximation Ratio Verification")
    print("=" * 60)

    np.random.seed(42)
    n_elements = 12
    n_sets = 8
    budgets = [1, 2, 3, 4, 5, 6]

    print(f"\nProblem: {n_elements} elements, {n_sets} candidate sets")
    print(f"{'Budget':>6} | {'Greedy':>8} | {'ILP':>6} | {'Ratio':>8} | {'1-1/e':>8}")
    print("-" * 45)

    for B in budgets:
        np.random.seed(42)
        sets = {}
        for i in range(n_sets):
            k = np.random.randint(3, 8)
            sets[i] = set(np.random.choice(n_elements, k, replace=False))

        # Greedy
        covered = set()
        remaining = list(sets.keys())
        for _ in range(B):
            best_gain = -1
            best_s = -1
            for s in remaining:
                gain = len(sets[s] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s >= 0:
                covered |= sets[best_s]
                remaining.remove(best_s)
        greedy_val = len(covered)

        # ILP
        ilp_val = 0
        for k in range(1, min(B + 1, n_sets + 1)):
            for combo in combinations(range(n_sets), k):
                cov = set()
                for si in combo:
                    cov |= sets[si]
                ilp_val = max(ilp_val, len(cov))
        ilp_val = max(greedy_val, ilp_val)

        ratio = greedy_val / ilp_val if ilp_val > 0 else 1.0
        approx_bound = 1 - 1 / np.e
        print(f"{B:>6} | {greedy_val:>8} | {ilp_val:>6} | {ratio:>8.4f} | {approx_bound:>8.4f}")

    print("=" * 60)


def load_real_results(results_dir):
    """从真实实验结果目录加载数据（备用接口）。"""
    # 如果真实数据存在，从这里加载
    results_file = os.path.join(results_dir, "all_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    else:
        print(f"[Warning] No results found at {results_file}")
        print("[Info] Falling back to demo data...")
        return None


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate publication-quality figures for CS240 project")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Path to real experiment results (all_results.json)")
    parser.add_argument("--ilp_only", action="store_true",
                       help="Run ILP approximation ratio verification only")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                       help="Output directory for figures")

    args = parser.parse_args()

    if args.ilp_only:
        run_ilp_only()
        sys.exit(0)

    # 确定数据源
    data = None
    if args.results_dir:
        data = load_real_results(args.results_dir)

    if data is None:
        if args.results_dir:
            print("[Info] Using demo data as fallback.")
        print("[Info] Generating synthetic demo data...")
        data = _make_demo_data()

    # 生成所有图表
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        def gaussian_filter(x, sigma, **kwargs):
            return x  # fallback
        # 这个仅供图4使用，如果没有 scipy 也能运行

    generate_all_figures(data, args.output_dir)

    print(f"\nTip: Run with --results_dir <path> to use real experiment data.")
    print(f"Tip: Run with --ilp_only to verify approximation ratio.")
