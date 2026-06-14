#!/usr/bin/env python3
"""
make_synthetic_scene.py — 生成合成 COLMAP 格式场景

在没有真实数据集时，用于测试 ExploreGS 流程。生成一个迷你场景：
  - 约 20 张合成训练图（简单几何体）
  - COLMAP 格式的 sparse 目录
  - PLY 点云文件

用法：
  python scripts/make_synthetic_scene.py --output_dir data/synthetic_scene

输出结构：
  data/synthetic_scene/
    ├── images/              # 合成训练图像 (PNG)
    ├── sparse/0/            # COLMAP 格式
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    └── points3d.ply         # 初始化点云
"""

import os
import sys
import argparse
import struct
import numpy as np
from PIL import Image
from pathlib import Path


def create_cameras_bin(filepath, num_cameras=1):
    """创建 COLMAP cameras.bin 文件"""
    with open(filepath, 'wb') as f:
        for cam_id in range(1, num_cameras + 1):
            # Camera: id, model_id, width, height, params
            cam_id_int = cam_id
            model_id = 1  # SIMPLE_PINHOLE
            width = 800
            height = 600
            fx = 500.0
            fy = 500.0
            cx = 400.0
            cy = 300.0

            f.write(struct.pack('Q', cam_id_int))
            f.write(struct.pack('i', model_id))
            f.write(struct.pack('II', width, height))
            # params: fx, fy, cx, cy (4 floats for SIMPLE_PINHOLE)
            f.write(struct.pack('i', 4))  # num_params
            f.write(struct.pack('dddd', fx, fy, cx, cy))


def create_images_bin(filepath, num_images=20, scene_radius=5.0):
    """创建 COLMAP images.bin 文件"""
    np.random.seed(42)
    with open(filepath, 'wb') as f:
        for img_id in range(1, num_images + 1):
            # 相机位置在球面上
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0.3, 1.2)
            r = scene_radius * (0.8 + 0.4 * np.random.random())

            position = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.cos(phi),
                r * np.sin(phi) * np.sin(theta),
            ])

            # 看向原点
            center = np.array([0.0, 0.0, 0.0])
            up = np.array([0.0, 1.0, 0.0])
            forward = center - position
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # COLMAP 的旋转矩阵是 world->camera 的转置
            R = np.vstack([right, up, -forward])
            qvec = rotmat_to_quat(R)
            tvec = -R @ position

            # Image data
            img_id_int = img_id
            f.write(struct.pack('Q', img_id_int))  # image_id

            # qvec
            f.write(struct.pack('dddd', qvec[0], qvec[1], qvec[2], qvec[3]))

            # tvec
            f.write(struct.pack('ddd', tvec[0], tvec[1], tvec[2]))

            # camera_id
            camera_id = 1
            f.write(struct.pack('i', camera_id))

            # image_name
            name = f"img_{img_id:04d}.png\0"
            f.write(name.encode('utf-8'))

            # num_points2D
            num_pts = 0
            f.write(struct.pack('Q', num_pts))


def create_points3d_bin(filepath, num_points=500):
    """创建 COLMAP points3D.bin 文件"""
    np.random.seed(42)
    with open(filepath, 'wb') as f:
        for pt_id in range(1, num_points + 1):
            # 随机点在场景中
            xyz = np.random.uniform(-2, 2, 3)
            rgb = np.random.uniform(0, 255, 3).astype(np.uint8)
            error = np.random.uniform(0.5, 2.0)

            f.write(struct.pack('Q', pt_id))
            f.write(struct.pack('ddd', xyz[0], xyz[1], xyz[2]))
            f.write(struct.pack('BBB', rgb[0], rgb[1], rgb[2]))
            f.write(struct.pack('d', error))
            # track: length 0
            f.write(struct.pack('Q', 0))


def gen_random_ply(filepath, num_points=500):
    """生成随机 PLY 点云"""
    np.random.seed(42)
    points = np.random.uniform(-2, 2, (num_points, 3))
    colors = np.random.uniform(0, 1, (num_points, 3))

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar diffuse_red\n")
        f.write("property uchar diffuse_green\n")
        f.write("property uchar diffuse_blue\n")
        f.write("end_header\n")

        for pt, col in zip(points, colors):
            nx, ny, nz = np.random.uniform(-1, 1, 3)
            n = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= n
            ny /= n
            nz /= n
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                    f"{nx:.6f} {ny:.6f} {nz:.6f} "
                    f"{int(col[0]*255)} {int(col[1]*255)} {int(col[2]*255)}\n")


def gen_images(output_dir, num_images=20):
    """生成简单的合成图像"""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, num_images + 1):
        img = Image.new('RGB', (800, 600), color=(30, 40, 50))
        # 添加一些简单的几何形状
        img_np = np.array(img)
        # 随机圆形
        np.random.seed(i)
        for _ in range(5):
            cx = np.random.randint(100, 700)
            cy = np.random.randint(100, 500)
            r = np.random.randint(20, 80)
            color = tuple(np.random.randint(50, 255, 3).tolist())
            # 画圆（简化：用方块近似）
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(800, cx + r), min(600, cy + r)
            img_np[y1:y2, x1:x2] = color
        img = Image.fromarray(img_np)
        img.save(os.path.join(output_dir, f"img_{i:04d}.png"))


def rotmat_to_quat(R):
    """将旋转矩阵转换为四元数"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s,
                     (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic COLMAP scene")
    parser.add_argument("--output_dir", type=str, default="data/synthetic_scene",
                        help="Output directory")
    parser.add_argument("--num_images", type=int, default=20,
                        help="Number of training images")
    parser.add_argument("--num_points", type=int, default=500,
                        help="Number of 3D points")
    parser.add_argument("--scene_radius", type=float, default=5.0,
                        help="Scene radius (camera distance)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"

    print(f"Generating synthetic scene at {output_dir}")
    print(f"  Images: {args.num_images}, Points: {args.num_points}, Radius: {args.scene_radius}")

    # 创建目录
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 生成 COLMAP binary 文件
    print("  Creating cameras.bin...")
    create_cameras_bin(str(sparse_dir / "cameras.bin"))

    print("  Creating images.bin...")
    create_images_bin(str(sparse_dir / "images.bin"), args.num_images, args.scene_radius)

    print("  Creating points3D.bin...")
    create_points3d_bin(str(sparse_dir / "points3D.bin"), args.num_points)

    # 生成随机点云 PLY
    print("  Generating point cloud...")
    gen_random_ply(str(sparse_dir / "points3D.ply"), args.num_points)

    # 生成合成图像
    print("  Generating images...")
    gen_images(str(images_dir), args.num_images)

    print(f"\nDone! Scene created at {output_dir}")
    print(f"\nTo test, run:")
    print(f"  CUDA_VISIBLE_DEVICES=0 python train_stage1.py \\")
    print(f"    -s {output_dir} --eval -r 1")
    print(f"    -c configs/stage1/default.yaml")
    print(f"    --load Colmap")


if __name__ == "__main__":
    main()
