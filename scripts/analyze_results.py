#!/usr/bin/env python3
"""
analyze_results.py — 实验结果分析脚本

功能：
  1. 读取 all_results.json，绘制 PSNR vs. B 对比曲线
  2. 打印每种策略在不同 B 下的 PSNR/SSIM/LPIPS 表格
  3. 绘制运行时间 (timing) 对比
  4. （可选）小实例下 Greedy vs ILP 近似比验证

用法：
  python scripts/analyze_results.py --results_dir output/ablation_study/bicycle
  python scripts/analyze_results.py --results_dir output/ablation_study/bicycle --timing_dir output/ablation_study/bicycle
  python scripts/analyze_results.py --run_ilp  # 运行 ILP 小实例验证
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict


def load_results(results_path):
    """加载 all_results.json"""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data


def plot_psnr_vs_b(results, save_path="psnr_vs_b.png"):
    """绘制 PSNR vs Budget B 曲线"""
    strategies = list(results.keys())
    colors = {
        'coverage': '#2E86AB',   # blue
        'fisher': '#A23B72',     # magenta
        'random': '#F18F01',     # orange
        'farthest': '#C73E1D',   # red
    }
    markers = {
        'coverage': 'o',
        'fisher': 's',
        'random': 'D',
        'farthest': '^',
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['PSNR', 'SSIM', 'LPIPS']
    ylabels = ['PSNR ↑', 'SSIM ↑', 'LPIPS ↓']
    ylims = [(None, None), (None, None), (0, None)]

    for idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[idx]

        for strategy in strategies:
            if strategy not in results:
                continue
            B_vals = []
            metric_vals = []
            for B in sorted(results[strategy].keys()):
                B_vals.append(int(B))
                metric_vals.append(results[strategy][B][metric])

            B_vals = np.array(B_vals)
            metric_vals = np.array(metric_vals)

            color = colors.get(strategy, 'gray')
            marker = markers.get(strategy, 'x')

            ax.plot(B_vals, metric_vals, color=color, marker=marker,
                    linewidth=2, markersize=8, label=strategy, zorder=3)

            # 标注数值
            for b, v in zip(B_vals, metric_vals):
                if metric == 'PSNR':
                    ax.annotate(f'{v:.2f}', (b, v), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=8, color=color)
                elif metric == 'SSIM':
                    ax.annotate(f'{v:.4f}', (b, v), textcoords="offset points",
                                xytext=(0, 10), ha='center', fontsize=8, color=color)
                else:
                    ax.annotate(f'{v:.4f}', (b, v), textcoords="offset points",
                                xytext=(0, -15), ha='center', fontsize=8, color=color)

        ax.set_xlabel('Budget B', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{metric} vs Budget', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(sorted(set(int(B) for s in results.values() for B in s.keys())))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Saved to {save_path}")
    plt.close()


def plot_timing(timing_data, save_path="timing_comparison.png"):
    """绘制运行时间对比（每个策略 vs K 或 B）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1：各策略各 B 的运行时间
    ax = axes[0]
    strategies = list(timing_data.keys())
    colors = {'coverage': '#2E86AB', 'fisher': '#A23B72',
              'random': '#F18F01', 'farthest': '#C73E1D'}
    markers = {'coverage': 'o', 'fisher': 's', 'random': 'D', 'farthest': '^'}

    for strategy in strategies:
        B_vals = sorted(timing_data[strategy].keys(), key=int)
        times = [timing_data[strategy][int(B)]['elapsed'] for B in B_vals]
        ax.plot(B_vals, times, color=colors.get(strategy, 'gray'),
                marker=markers.get(strategy, 'x'), linewidth=2,
                markersize=8, label=strategy)
        for b, t in zip(B_vals, times):
            ax.annotate(f'{t:.1f}s', (b, t), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=8)

    ax.set_xlabel('Budget B')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Selection Time vs Budget')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 图2：log 尺度
    ax = axes[1]
    for strategy in strategies:
        B_vals = sorted(timing_data[strategy].keys(), key=int)
        times = [timing_data[strategy][int(B)]['elapsed'] for B in B_vals]
        ax.plot(B_vals, times, color=colors.get(strategy, 'gray'),
                marker=markers.get(strategy, 'x'), linewidth=2,
                markersize=8, label=strategy)

    ax.set_xlabel('Budget B')
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title('Selection Time (log scale)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Plot] Timing saved to {save_path}")
    plt.close()


def print_results_table(results):
    """打印结果表格"""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    metrics = ['PSNR', 'SSIM', 'LPIPS']
    header = f"{'Strategy':<15}"
    for B in sorted(next(iter(results.values())).keys(), key=int):
        for m in metrics:
            header += f" | {m}@B={B:<4}"
    print(header)
    print("-" * len(header))

    for strategy in results.keys():
        row = f"{strategy:<15}"
        for B in sorted(results[strategy].keys(), key=int):
            for m in metrics:
                val = results[strategy][B][m]
                if m == 'PSNR':
                    row += f" | {val:<8.4f}"
                else:
                    row += f" | {val:<8.4f}"
        print(row)

    print("=" * 80)


def run_timing_analysis(timing_dir, save_path="timing_analysis.png"):
    """分析不同 K 下的运行时间成本"""
    import glob
    timing_files = glob.glob(os.path.join(timing_dir, "timing_*.json"))
    if not timing_files:
        print(f"[Warning] No timing files found in {timing_dir}")
        return

    timing_data = {}
    for tf in timing_files:
        with open(tf) as f:
            data = json.load(f)
        strategy = data['strategy']
        B = data['B']
        if strategy not in timing_data:
            timing_data[strategy] = {}
        timing_data[strategy][B] = data

    if timing_data:
        plot_timing(timing_data, save_path)


# ============================================================================
# ILP 小实例近似比验证
# ============================================================================
def solve_max_coverage_ilp(elements, sets, B):
    """
    使用 ILP (整数线性规划) 求解最大覆盖问题的最优解。
    通过暴力枚举（小实例）或 pulp/ortools（稍大实例）。

    参数:
        elements: list of element IDs
        sets: dict {set_id: set of elements covered}
        B: 最多选 B 个集合

    返回:
        optimal_value: 最优覆盖数
        selected_sets: 选中的集合 ID 列表
    """
    n = len(sets)
    set_ids = list(sets.keys())
    all_elements = set(elements)

    # 暴力枚举所有组合（仅适用于集合数 ≤ 25，B ≤ 10）
    from itertools import combinations

    best_value = -1
    best_combo = None

    for k in range(1, B + 1):
        for combo in combinations(range(n), k):
            covered = set()
            for idx in combo:
                covered |= sets[set_ids[idx]]
            value = len(covered)
            if value > best_value:
                best_value = value
                best_combo = combo

    selected = [set_ids[i] for i in best_combo] if best_combo else []
    return best_value, selected


def run_ilp_approx_verification():
    """
    用一个小实例验证 Greedy 的 (1 - 1/e) 近似比。

    构造一个场景：
    - 10 个高斯
    - 6 个候选视角，每个覆盖不同的子集
    """
    print("\n" + "=" * 60)
    print("ILP Approximation Ratio Verification")
    print("=" * 60)

    # 构造一个小实例（所有高斯都在同一个平面上）
    np.random.seed(42)
    n_gaussians = 10
    n_candidates = 6

    # 人为定义每个候选视角覆盖的高斯
    coverage_sets = {
        0: {0, 1, 2, 3},       # 覆盖 4 个
        1: {2, 3, 4, 5},       # 覆盖 4 个
        2: {4, 5, 6, 7},       # 覆盖 4 个
        3: {6, 7, 8, 9},       # 覆盖 4 个
        4: {0, 4, 8},          # 覆盖 3 个
        5: {1, 5, 9},          # 覆盖 3 个
    }
    elements = list(range(n_gaussians))

    print(f"\nProblem instance:")
    print(f"  Elements (Gaussians): {n_gaussians}")
    print(f"  Candidate views: {n_candidates}")
    print(f"  Coverage matrix:")
    for sid, covered in coverage_sets.items():
        mask = ''.join(['1' if i in covered else '0' for i in elements])
        print(f"    View {sid}: [{mask}] covers {len(covered)} elements")

    budgets = [1, 2, 3, 4, 5]
    print(f"\n{'Budget':>6} | {'Greedy':>8} | {'ILP Optimal':>12} | {'Ratio':>8} | {'1-1/e':>8}")
    print("-" * 50)

    for B in budgets:
        # Greedy
        covered_greedy = set()
        selected_greedy = []
        remaining = list(coverage_sets.keys())

        for _ in range(B):
            best_gain = -1
            best_s = -1
            for s in remaining:
                gain = len(coverage_sets[s] - covered_greedy)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s >= 0:
                covered_greedy |= coverage_sets[best_s]
                selected_greedy.append(best_s)
                remaining.remove(best_s)

        # ILP 最优解
        optimal_val, selected_ilp = solve_max_coverage_ilp(elements, coverage_sets, B)
        optimal_val = max(len(covered_greedy), optimal_val)  # ensure at least greedy

        ratio = len(covered_greedy) / optimal_val if optimal_val > 0 else 0
        approx_bound = 1 - 1 / np.e

        print(f"{B:>6} | {len(covered_greedy):>8} | {optimal_val:>12} | {ratio:>8.4f} | {approx_bound:>8.4f}")

    # 随机实例测试
    print("\n\nRandom instance test (20 runs):")
    n_gaussians = 15
    n_candidates = 10

    ratios = []
    for run in range(20):
        np.random.seed(run)
        coverage_sets = {}
        for i in range(n_candidates):
            # 随机子集，大小 3~8
            k = np.random.randint(3, 9)
            coverage_sets[i] = set(np.random.choice(n_gaussians, k, replace=False))

        elements = list(range(n_gaussians))
        B = min(5, n_candidates)

        # Greedy
        covered_greedy = set()
        remaining = list(coverage_sets.keys())
        for _ in range(B):
            best_gain = -1
            best_s = -1
            for s in remaining:
                gain = len(coverage_sets[s] - covered_greedy)
                if gain > best_gain:
                    best_gain = gain
                    best_s = s
            if best_s >= 0:
                covered_greedy |= coverage_sets[best_s]
                remaining.remove(best_s)

        # ILP
        optimal_val, _ = solve_max_coverage_ilp(elements, coverage_sets, B)
        optimal_val = max(len(covered_greedy), optimal_val)

        ratio = len(covered_greedy) / optimal_val if optimal_val > 0 else 0
        ratios.append(ratio)

    print(f"  Mean ratio: {np.mean(ratios):.4f} (theoretical bound: {1 - 1/np.e:.4f})")
    print(f"  Min ratio:  {np.min(ratios):.4f}")
    print(f"  Always ≥ 1-1/e: {all(r >= (1 - 1/np.e) - 1e-6 for r in ratios)}")
    print("=" * 60)


# ============================================================================
# 主函数
# ============================================================================
if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze ablation study results")
    parser.add_argument('--results_dir', type=str, default="output/ablation_study/bicycle",
                        help='Directory containing all_results.json')
    parser.add_argument('--timing_dir', type=str, default=None,
                        help='Directory containing timing_*.json files')
    parser.add_argument('--output_dir', type=str, default="analysis_output",
                        help='Output directory for plots')
    parser.add_argument('--run_ilp', action='store_true',
                        help='Run ILP approximation ratio verification')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_ilp:
        run_ilp_approx_verification()
        exit(0)

    # 加载结果
    results_path = os.path.join(args.results_dir, "all_results.json")
    if os.path.exists(results_path):
        results = load_results(results_path)
        print_results_table(results)
        plot_psnr_vs_b(results, os.path.join(args.output_dir, "psnr_vs_b.png"))

        # 检查是否存在 Stage 1 baselines
        stage1_result = os.path.join(
            os.path.dirname(args.results_dir.rstrip('/')),
            "stage1_metrics.json"
        )
        if os.path.exists(stage1_result):
            with open(stage1_result) as f:
                stage1_data = json.load(f)
            print(f"\nStage 1 Baseline: PSNR={stage1_data.get('PSNR', 'N/A')}")
    else:
        print(f"[Warning] Results not found at {results_path}")
        print("Available files:")
        for f in os.listdir(args.results_dir):
            print(f"  {f}")

    # 时序分析
    if args.timing_dir:
        run_timing_analysis(args.timing_dir, os.path.join(args.output_dir, "timing_analysis.png"))
    else:
        # Try default timing location
        if os.path.isdir(args.results_dir):
            run_timing_analysis(args.results_dir, os.path.join(args.output_dir, "timing_analysis.png"))

    print(f"\n[Analysis] Outputs saved to {args.output_dir}/")
