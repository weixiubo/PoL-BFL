#!/usr/bin/env python3
"""
重置区块链状态脚本
解决"Client already registered"错误
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "chainfl"))

# 确保当前工作目录正确
import os
os.chdir(str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reset_blockchain():
    """重置区块链状态"""
    try:
        # 导入区块链交互模块
        from chainfl.interact import ChainProxy

        logger.info("正在重置区块链状态...")

        # 创建新的ChainProxy实例（这会重新部署合约）
        chain_proxy = ChainProxy()

        logger.info("区块链状态重置完成")
        return True

    except Exception as e:
        logger.error(f"重置区块链状态失败: {e}")
        return False

def clean_pol_data():
    """清理PoL数据目录"""
    try:
        pol_data_dir = project_root / "experiments" / "results" / "rq1_security" / "pol_data"

        if pol_data_dir.exists():
            import shutil
            shutil.rmtree(pol_data_dir)
            logger.info(f"已清理PoL数据目录: {pol_data_dir}")

        return True

    except Exception as e:
        logger.error(f"清理PoL数据失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始重置实验环境...")

    # 1. 清理PoL数据
    if not clean_pol_data():
        logger.error("清理PoL数据失败")
        return False

    # 2. 重置区块链状态
    if not reset_blockchain():
        logger.error("重置区块链状态失败")
        return False

    logger.info("实验环境重置完成。")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
