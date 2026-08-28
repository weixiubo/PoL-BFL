#!/usr/bin/env python3
"""
结果迁移工具 - 将历史结果迁入标准目录
默认dry-run模式，需要--execute才真正执行
"""

import os
import shutil
import argparse
import json
from pathlib import Path

def find_previous_layout_results(root_dir):
    """查找需要迁移的历史结果"""
    previous_paths = []

    # 查找可能的历史结果目录
    potential_dirs = [
        'experiments/experiments/results',
        'experiments/rq1_results',
        'experiments/rq2_results',
        'experiments/rq3_results',
        'experiments/rq4_results',
        'pol_data',
        'experiments/pol_data'
    ]

    for previous_dir in potential_dirs:
        full_path = os.path.join(root_dir, previous_dir)
        if os.path.exists(full_path):
            previous_paths.append(full_path)

    return previous_paths

def migrate_directory(src, dst, dry_run=True):
    """迁移目录"""
    if dry_run:
        print(f"[DRY-RUN] Would migrate: {src} -> {dst}")
        if os.path.exists(src):
            # 计算大小
            total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                           for dirpath, dirnames, filenames in os.walk(src)
                           for filename in filenames)
            print(f"  Size: {total_size / (1024*1024*1024):.2f} GB")

            # 计算文件数
            file_count = sum(len(filenames) for _, _, filenames in os.walk(src))
            print(f"  Files: {file_count}")
        return True
    else:
        print(f"[EXECUTE] Migrating: {src} -> {dst}")
        try:
            # 确保目标目录存在
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # 移动目录
            shutil.move(src, dst)
            print(f"  [PASS] Success")
            return True
        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            return False

def create_compatibility_links(root_dir, dry_run=True):
    """创建兼容性软链接"""
    links = [
        ('log', 'experiments/logs'),
        ('data', 'experiments/data'),
    ]

    for target, link_path in links:
        full_link_path = os.path.join(root_dir, link_path)
        full_target_path = os.path.join(root_dir, target)

        if dry_run:
            print(f"[DRY-RUN] Would create symlink: {link_path} -> {target}")
        else:
            try:
                # 删除已存在的链接或目录
                if os.path.exists(full_link_path) or os.path.islink(full_link_path):
                    if os.path.islink(full_link_path):
                        os.unlink(full_link_path)
                    else:
                        print(f"  Warning: {link_path} exists and is not a symlink, skipping")
                        continue

                # 创建相对路径软链接
                rel_target = os.path.relpath(full_target_path, os.path.dirname(full_link_path))
                os.symlink(rel_target, full_link_path)
                print(f"[EXECUTE] Created symlink: {link_path} -> {rel_target}")
            except Exception as e:
                print(f"  [FAIL] Error creating symlink: {e}")

def main():
    parser = argparse.ArgumentParser(description='Migrate previous-layout results to standard directories')
    parser.add_argument('--execute', action='store_true', help='Actually execute the migration (default: dry-run)')
    parser.add_argument('--create-links', action='store_true', help='Create compatibility symlinks')
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.execute:
        print("[ALERT] EXECUTE MODE - Changes will be made.")
    else:
        print("[CHECK] DRY-RUN MODE - No changes will be made")
        print("Use --execute to actually perform migration")

    print(f"Working directory: {root_dir}")
    print()

    # 查找历史结果
    previous_paths = find_previous_layout_results(root_dir)

    if not previous_paths:
        print("[PASS] No previous-layout results found to migrate")
    else:
        print(f"Found {len(previous_paths)} previous-layout result directories:")
        for path in previous_paths:
            rel_path = os.path.relpath(path, root_dir)
            print(f"  [PATH] {rel_path}")
        print()

        # Migrate each detected result directory into the archive layout.
        for previous_path in previous_paths:
            rel_path = os.path.relpath(previous_path, root_dir)
            # 简单示例：将所有内容移到archives/
            archive_path = os.path.join(
                root_dir, 'archives', 'previous_' + rel_path.replace('/', '_')
            )
            migrate_directory(previous_path, archive_path, dry_run=not args.execute)

    # 创建兼容性链接
    if args.create_links:
        print("\n[LINK] Creating compatibility symlinks:")
        create_compatibility_links(root_dir, dry_run=not args.execute)

if __name__ == "__main__":
    main()
