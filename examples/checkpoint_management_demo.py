#!/usr/bin/env python3
"""
Checkpoint管理开关演示

这个脚本演示如何使用新的checkpoint管理功能：
1. 磁盘模式 vs 内存模式
2. 自动清理功能
3. 内存限制控制

使用方法：
1. 磁盘模式（默认）：python checkpoint_management_demo.py --mode disk
2. 内存模式：python checkpoint_management_demo.py --mode memory
3. 自动清理模式：python checkpoint_management_demo.py --mode disk --auto-cleanup
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.pol.PoLManager import PoLManager
from config.pol_config import POL_CONFIG


class SimpleModel(nn.Module):
    """简单的演示模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)


def create_demo_data():
    """创建演示数据"""
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)


def demo_checkpoint_management(mode='disk', auto_cleanup=False):
    """
    演示checkpoint管理功能
    
    Args:
        mode: 'disk' 或 'memory'
        auto_cleanup: 是否启用自动清理
    """
    print(f"\n{'='*60}")
    print(f"Checkpoint Management Demo - {mode.upper()} Mode")
    if auto_cleanup:
        print("Auto Cleanup: ENABLED")
    print(f"{'='*60}")
    
    # 配置参数
    save_to_disk = (mode == 'disk')
    memory_limit = 3  # 内存模式下只保留3个checkpoint
    auto_cleanup_interval = 5  # 每5个checkpoint清理一次
    
    # 创建PoL管理器
    pol_manager = PoLManager(
        client_id="demo_client",
        save_dir="demo_pol_data",
        save_freq=2,  # 每2个batch保存一次
        compress=True,
        save_to_disk=save_to_disk,
        memory_limit=memory_limit,
        enable_auto_cleanup=auto_cleanup,
        auto_cleanup_interval=auto_cleanup_interval
    )
    
    # 创建模型和数据
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    dataloader = create_demo_data()
    
    print(f"\nStarting training simulation...")
    print(f"Save to disk: {save_to_disk}")
    if not save_to_disk:
        print(f"Memory limit: {memory_limit}")
    if auto_cleanup:
        print(f"Auto cleanup interval: {auto_cleanup_interval}")
    
    step = 0
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # 模拟训练
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            # 保存checkpoint
            if step % pol_manager.save_freq == 0:
                checkpoint_data = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'loss': loss.item()
                }
                
                ckpt_hash = pol_manager.save_checkpoint(step, checkpoint_data)
                
                # 显示状态
                if save_to_disk:
                    print(f"  Step {step}: Saved to disk, hash: {ckpt_hash[:8]}...")
                else:
                    memory_count = pol_manager.get_memory_checkpoint_count()
                    print(f"  Step {step}: Saved to memory ({memory_count}/{memory_limit}), hash: {ckpt_hash[:8]}...")
    
    # 显示最终状态
    print(f"\n{'='*40}")
    print("Final Status:")
    print(f"Total checkpoints created: {pol_manager.get_checkpoint_count()}")
    
    if save_to_disk:
        # 检查磁盘上的文件
        checkpoint_dir = os.path.join(pol_manager.save_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            files = [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_")]
            print(f"Files on disk: {len(files)}")
            if auto_cleanup:
                print("Note: Some files may have been auto-cleaned")
        else:
            print("No checkpoint directory found")
    else:
        memory_count = pol_manager.get_memory_checkpoint_count()
        print(f"Checkpoints in memory: {memory_count}")
        print("Memory checkpoint steps:", list(pol_manager.memory_checkpoints.keys()))
    
    # 清理演示数据
    if save_to_disk and os.path.exists("demo_pol_data"):
        import shutil
        shutil.rmtree("demo_pol_data")
        print("\nCleaned up demo data directory")


def main():
    parser = argparse.ArgumentParser(description='Checkpoint Management Demo')
    parser.add_argument('--mode', choices=['disk', 'memory'], default='disk',
                        help='Checkpoint storage mode')
    parser.add_argument('--auto-cleanup', action='store_true',
                        help='Enable auto cleanup')
    
    args = parser.parse_args()
    
    # 运行演示
    demo_checkpoint_management(args.mode, args.auto_cleanup)
    
    print(f"\n{'='*60}")
    print("Demo completed!")
    print("\nTo use in your code:")
    print("1. Set 'save_checkpoints_to_disk': False in config/pol_config.py")
    print("2. Adjust 'memory_checkpoint_limit' as needed")
    print("3. Enable 'enable_auto_cleanup' for automatic cleanup")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
