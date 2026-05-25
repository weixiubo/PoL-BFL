# RQ2 Ablation Study - 实验配置记录

**目的**: 记录RQ2实验的所有配置版本，确保不会忘记原始配置

---

## 📋 配置版本历史

### Version 1: 原始完整配置（论文最终版）

**文件**: `run_rq2_ablation.py`  
**用途**: 论文最终实验，完整数据

```python
RQ2_ABLATION_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 20,
    'clients_per_round': 10,
    'num_rounds': 100,  # ⭐ 原始配置：100轮
    'data_distribution': 'NonIID_Dirichlet',
    
    # Attack scenario (fixed for ablation)
    'attack_type': 'byzantine_random_noise',
    'malicious_ratio': 0.2,
    'noise_scale': 1.0,
    
    # Variants to test
    'variants': [
        'vanilla_fl',
        'pol_only',
        'pol_zkp',
        'pol_incentive',
        'pol_zkp_incentive'
    ],
    
    # Number of repetitions for statistical significance
    'num_repetitions': 3,
}
```

**预计时间**: ~15小时（5个变体 × 3次重复 × 100轮）  
**状态**: ⏸️ 待执行（改进设计验证后）

---

### Version 2: Phase 3小规模测试

**文件**: `run_rq2_ablation.py`（临时修改）  
**用途**: 快速验证代码正确性

```python
RQ2_ABLATION_CONFIG = {
    'dataset': 'MNIST',  # 改为MNIST（更快）
    'model': 'SimpleCNN',
    'num_clients': 20,
    'clients_per_round': 10,
    'num_rounds': 5,  # ⭐ 小规模：5轮
    'data_distribution': 'NonIID_Dirichlet',
    
    # 其他配置同上
    'num_repetitions': 3,
}
```

**预计时间**: ~45分钟  
**状态**: ✅ 已完成（2025-10-23，发现检测指标问题）

---

### Version 3: 中等规模测试（当前）

**文件**: `run_rq2_ablation.py`（当前版本）  
**用途**: 验证改进设计，快速获得有意义的数据

```python
RQ2_ABLATION_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 20,
    'clients_per_round': 10,
    'num_rounds': 20,  # ⭐ 中等规模：20轮（原100轮的1/5）
    'data_distribution': 'NonIID_Dirichlet',
    
    # Attack scenario (fixed for ablation)
    'attack_type': 'byzantine_random_noise',
    'malicious_ratio': 0.2,
    'noise_scale': 1.0,
    
    # Variants to test
    'variants': [
        'vanilla_fl',
        'pol_only',
        'pol_zkp',
        'pol_incentive',
        'pol_zkp_incentive'
    ],
    
    # Number of repetitions for statistical significance
    'num_repetitions': 3,
}
```

**预计时间**: ~3小时（5个变体 × 3次重复 × 20轮）  
**状态**: 🔄 准备执行（2025-10-23，改进设计后）

**优势**:
- ✅ 快5倍（3小时 vs 15小时）
- ✅ 数据足够有意义（20轮足以看出趋势）
- ✅ 可以快速验证改进设计是否有效
- ✅ 如果结果好，晚上再启动完整100轮实验

---

## 🔄 配置切换指南

### 切换到完整配置（100轮）

```bash
# 编辑 run_rq2_ablation.py
# 修改第53行：
'num_rounds': 100,  # 从20改回100
```

### 切换到中等配置（20轮）

```bash
# 编辑 run_rq2_ablation.py
# 修改第53行：
'num_rounds': 20,  # 从100改为20
```

### 切换到小规模测试（5轮，MNIST）

```bash
# 编辑 run_rq2_ablation.py
# 修改第49-53行：
'dataset': 'MNIST',
'model': 'SimpleCNN',
'num_rounds': 5,
```

---

## 📊 各配置对比

| 配置 | 数据集 | 轮数 | 重复 | 预计时间 | 用途 | 状态 |
|------|--------|------|------|---------|------|------|
| **Version 1** | CIFAR-10 | 100 | 3 | ~15小时 | 论文最终版 | ⏸️ 待执行 |
| **Version 2** | MNIST | 5 | 3 | ~45分钟 | 代码验证 | ✅ 已完成 |
| **Version 3** | CIFAR-10 | 20 | 3 | ~3小时 | 改进设计验证 | 🔄 当前 |

---

## 🎯 执行计划

### 当前阶段（2025-10-23）

1. **立即执行**: Version 3（20轮，3小时）
   - 验证改进设计（所有节点启用PoL + 最终模型验证）
   - 获得初步数据，评估TPR/FPR

2. **如果结果好**: 晚上启动Version 1（100轮，15小时）
   - 获得完整数据用于论文
   - 后台运行，明天早上查看结果

3. **如果结果不好**: 调试并重新测试
   - 分析问题，调整参数
   - 重新运行Version 3验证

---

## 📝 重要提醒

**⚠️ 在修改配置前，务必检查此文件，确保知道原始配置！**

**原始配置（论文最终版）**:
- Dataset: CIFAR-10
- Model: ResNet18
- Rounds: **100**
- Repetitions: 3
- Variants: 5个（vanilla_fl, pol_only, pol_zkp, pol_incentive, pol_zkp_incentive）

**当前配置（快速验证）**:
- Dataset: CIFAR-10
- Model: ResNet18
- Rounds: **20**（临时，用于快速验证）
- Repetitions: 3
- Variants: 5个（同上）

---

**创建时间**: 2025-10-23 21:10  
**最后更新**: 2025-10-23 21:10  
**维护者**: AI Assistant

