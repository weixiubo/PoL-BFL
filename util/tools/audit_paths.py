#!/usr/bin/env python3
"""
路径审计工具 - 扫描硬编码/遗留路径
用于发现代码中的硬编码路径，确保使用OUTPUT_CONFIG或环境变量
"""

import os
import re
import glob
from pathlib import Path

def scan_hardcoded_paths(root_dir):
    """扫描硬编码路径"""
    issues = []

    # 需要检查的文件类型
    patterns = ['**/*.py', '**/*.sh', '**/*.js']

    # 硬编码路径模式
    hardcoded_patterns = [
        r'experiments/experiments',  # 嵌套路径
        r'experiments/data/',        # 应该用data/
        r'experiments/logs/',        # 应该用log/
        r'pol_data/',               # 应该用checkpoints/
        r'/home/[^/]+/',            # 绝对用户路径
        r'experiments/rq\d+_',      # 旧的实验路径
    ]

    for pattern in patterns:
        for file_path in glob.glob(os.path.join(root_dir, pattern), recursive=True):
            if 'node_modules' in file_path or '__pycache__' in file_path:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                for line_num, line in enumerate(content.split('\n'), 1):
                    for hp_pattern in hardcoded_patterns:
                        if re.search(hp_pattern, line):
                            issues.append({
                                'file': file_path,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': hp_pattern
                            })
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")

    return issues

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Scanning for hardcoded paths in: {root_dir}")

    issues = scan_hardcoded_paths(root_dir)

    if not issues:
        print("[PASS] No hardcoded path issues found.")
        return

    print(f"[ALERT] Found {len(issues)} potential hardcoded path issues:")
    print()

    for issue in issues:
        rel_path = os.path.relpath(issue['file'], root_dir)
        print(f"[PATH] {rel_path}:{issue['line']}")
        print(f"   Pattern: {issue['pattern']}")
        print(f"   Content: {issue['content']}")
        print()

if __name__ == "__main__":
    main()
