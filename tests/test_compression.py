#!/usr/bin/env python3
"""
测试checkpoint压缩效果
用于论文中报告实测压缩比
"""

import torch
import gzip
import os
import glob
from pathlib import Path

def test_compression():
    """测试gzip压缩效果"""

    print("="*70)
    print("Checkpoint Compression Test")
    print("="*70)

    # 找到checkpoint目录 - 使用环境变量或默认路径
    ckpt_base = os.environ.get('POL_CHECKPOINTS_DIR', 'experiments/checkpoints')
    ckpt_dir = Path(ckpt_base)

    if not ckpt_dir.exists():
        print(f"Error: Checkpoint directory not found: {ckpt_dir}")
        return

    # 获取所有client_0的checkpoint（测试10个样本）
    ckpt_files = sorted(glob.glob(str(ckpt_dir / 'client_0_iter_*.pt')))[:10]

    if not ckpt_files:
        print(f"Error: No checkpoint files found in {ckpt_dir}")
        return

    print(f"\nTesting {len(ckpt_files)} checkpoint files...")
    print(f"Sample files: {Path(ckpt_files[0]).name} ... {Path(ckpt_files[-1]).name}")
    print()

    total_orig = 0
    total_comp = 0
    ratios = []

    for i, ckpt_file in enumerate(ckpt_files, 1):
        try:
            # 加载checkpoint
            ckpt = torch.load(ckpt_file, map_location='cpu')

            # Gzip压缩
            comp_file = ckpt_file + '.test.gz'
            with gzip.open(comp_file, 'wb', compresslevel=6) as f:
                torch.save(ckpt, f)

            # 计算大小
            size_orig = os.path.getsize(ckpt_file)
            size_comp = os.path.getsize(comp_file)
            ratio = size_orig / size_comp

            total_orig += size_orig
            total_comp += size_comp
            ratios.append(ratio)

            print(f"[{i}/{len(ckpt_files)}] {Path(ckpt_file).name}")
            print(f"  Original: {size_orig/1024:.2f} KB")
            print(f"  Compressed: {size_comp/1024:.2f} KB")
            print(f"  Ratio: {ratio:.2f}x")

            # 清理临时文件
            os.remove(comp_file)

        except Exception as e:
            print(f"Error processing {ckpt_file}: {e}")
            continue

    # 计算总体统计
    avg_ratio = total_orig / total_comp if total_comp > 0 else 0
    min_ratio = min(ratios) if ratios else 0
    max_ratio = max(ratios) if ratios else 0

    print()
    print("="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"Total original size: {total_orig/1024/1024:.2f} MB")
    print(f"Total compressed size: {total_comp/1024/1024:.2f} MB")
    print(f"Average compression ratio: {avg_ratio:.2f}x")
    print(f"Min ratio: {min_ratio:.2f}x")
    print(f"Max ratio: {max_ratio:.2f}x")
    print()

    # 估算全部checkpoint的压缩效果
    print("="*70)
    print("Estimated Total Storage (all checkpoints)")
    print("="*70)

    # 计算总checkpoint数量
    all_ckpts = glob.glob(str(ckpt_dir / 'client_*_iter_*.pt'))
    total_ckpts = len(all_ckpts)

    # 估算总大小
    estimated_total_orig = 1.3  # GB (从ls -lh看到的)
    estimated_total_comp = estimated_total_orig / avg_ratio

    print(f"Total checkpoints: {total_ckpts}")
    print(f"Uncompressed: {estimated_total_orig:.2f} GB")
    print(f"Compressed (gzip level 6): {estimated_total_comp:.2f} GB ({estimated_total_comp*1024:.0f} MB)")
    print(f"Space saved: {estimated_total_orig - estimated_total_comp:.2f} GB ({(1-1/avg_ratio)*100:.1f}%)")
    print()

    # 论文中可以使用的数据
    print("="*70)
    print("Reported Discussion Metrics")
    print("="*70)
    print("The benchmark uses gzip compression (level 6) on SimpleCNN checkpoints and")
    print(f"achieved a compression ratio of {avg_ratio:.1f}×, reducing the {estimated_total_orig:.2f}GB")
    print(f"storage to {estimated_total_comp*1024:.0f}MB. This is comparable to storing 2-3 model")
    print(f"snapshots, which is acceptable for most FL deployments.")
    print()

    return avg_ratio, estimated_total_comp

if __name__ == '__main__':
    test_compression()
