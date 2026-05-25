# Experiments Scripts Guide

**Last Updated**: 2025-10-22  
**Purpose**: 说明experiments目录中各实验脚本的用途和使用方法

---

## 📋 目录

1. [RQ1: Security Evaluation](#rq1-security-evaluation)
2. [RQ2: System Overhead](#rq2-system-overhead)
3. [RQ3: Scalability](#rq3-scalability)
4. [RQ4: Incentive Mechanism](#rq4-incentive-mechanism)
5. [CIFAR-10 Experiments](#cifar-10-experiments)
6. [Utility Scripts](#utility-scripts)

---

## RQ1: Security Evaluation

### 核心脚本

#### 1. `run_rq1_security.py` ⭐ **主脚本**
**用途**: RQ1安全性评估的主要脚本  
**特点**:
- 测试多种攻击场景（Byzantine、Free-riding）
- 对比多种防御基线（Vanilla FL、Krum、Trimmed Mean）
- 使用MNIST + SimpleCNN（快速测试）
- 输出准确率和收敛性指标

**配置**:
```python
RQ1_SIMPLE_CONFIG = {
    'dataset': 'MNIST',
    'model': 'SimpleCNN',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 10,
    'attacks': {
        'no_attack': {...},
        'byzantine_random_noise': {...},
        'free_riding_no_training': {...}
    },
    'baselines': ['Vanilla_FL', 'Krum', 'Trimmed_Mean']
}
```

**运行**:
```bash
python experiments/run_rq1_security.py
```

**输出**: `experiments/results/rq1_security/rq1_results.json`

---

#### 2. `run_rq1_with_detection.py` ⭐ **Detection版本**
**用途**: RQ1 + PoL Detection Metrics  
**特点**:
- 在`run_rq1_security.py`基础上添加PoL验证
- 使用PoLClient + PoLTrainer + PoLVerifyAggregator
- 记录Detection Rate (TPR, FPR, Precision, Recall, F1)
- 更多攻击场景（包括label flipping、lazy training）

**关键差异**:
```python
# 使用PoL组件
from client.PoLClient import PoLClient
from client.trainer.PoLTrainer import PoLTrainer
from server.aggregation_alg.PoLVerifyAggregator import PoLVerifyAggregator

# 记录detection metrics
detection_metrics = compute_detection_metrics(
    verification_results,
    malicious_clients,
    all_clients
)
```

**运行**:
```bash
python experiments/run_rq1_with_detection.py
```

**输出**: `experiments/results/rq1_security/rq1_with_detection.json`

---

#### 3. `run_rq1_pol_detection.py` 🔬 **Detection专用**
**用途**: 专门测量PoL Detection Rate  
**特点**:
- 最小化实验（1轮，4个客户端）
- 专注于TPR/FPR测量
- 禁用链上/ZKP/激励机制（避免依赖）
- 快速验证PoL检测功能

**使用场景**:
- 快速测试PoL检测功能
- 调试PoL验证逻辑
- 生成detection rate数据

**运行**:
```bash
python experiments/run_rq1_pol_detection.py
```

**输出**: `experiments/results/rq1_security/pol_detection.json`

**状态**: ⚠️ 功能已被`run_rq1_with_detection.py`包含，可考虑归档

---

#### 4. `run_rq1_resnet18.py` 🎯 **ResNet-18版本**
**用途**: 使用ResNet-18 + CIFAR-10进行RQ1实验  
**特点**:
- 更复杂的模型和数据集
- 满足CVPR审稿人对复杂模型的期望
- 基于`run_rq1_security.py`的配置

**配置**:
```python
RQ1_RESNET18_CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'ResNet18',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 5,  # 快速测试，完整实验用50轮
    ...
}
```

**运行**:
```bash
python experiments/run_rq1_resnet18.py
```

**输出**: `experiments/results/rq1_security/rq1_resnet18_results.json`

---

### RQ1脚本对比总结

| 脚本 | 数据集 | 模型 | PoL | Detection | 用途 |
|------|--------|------|-----|-----------|------|
| `run_rq1_security.py` | MNIST | SimpleCNN | ❌ | ❌ | 基线对比 |
| `run_rq1_with_detection.py` | MNIST | SimpleCNN | ✅ | ✅ | PoL + Detection |
| `run_rq1_pol_detection.py` | MNIST | SimpleCNN | ✅ | ✅ | 快速Detection测试 |
| `run_rq1_resnet18.py` | CIFAR-10 | ResNet-18 | ❌ | ❌ | 复杂模型评估 |

**推荐使用**:
- 论文实验: `run_rq1_with_detection.py` (完整数据)
- 快速测试: `run_rq1_pol_detection.py` (验证功能)
- 复杂模型: `run_rq1_resnet18.py` (CVPR要求)

---

## RQ2: System Overhead

### `run_rq2_overhead.py` ⭐
**用途**: 测量PoL-FL的系统开销  
**测量指标**:
- 训练时间
- Checkpoint保存时间
- ZKP生成时间
- 模型上传大小
- ZKP证明大小
- Gas成本

**运行**:
```bash
python experiments/run_rq2_overhead.py
```

**输出**: `experiments/results/rq2_overhead/rq2_results.json`

---

## RQ3: Scalability

### `run_rq3_scalability.py` ⭐
**用途**: 测试系统可扩展性  
**测试维度**:
- 客户端数量 (10, 20, 50, 100)
- 模型大小
- 数据集大小

**运行**:
```bash
python experiments/run_rq3_scalability.py
```

**输出**: `experiments/results/rq3_scalability/rq3_results.json`

---

## RQ4: Incentive Mechanism

### `run_rq4_incentive.py` ⭐
**用途**: 评估经济激励机制  
**测试内容**:
- 质押机制
- 奖励分配
- 声誉系统
- 博弈论分析

**运行**:
```bash
python experiments/run_rq4_incentive.py
```

**输出**: `experiments/results/rq4_incentive/rq4_results.json`

---

## CIFAR-10 Experiments

### 1. `run_cifar10_paper.py` ⭐ **论文实验**
**用途**: 论文中的CIFAR-10完整实验
**特点**:
- 50轮训练，5个local epochs
- 测试Vanilla FL和Trimmed Mean
- 包含无攻击、Byzantine、Free-riding场景
- 支持`--only-no-attack`参数快速测试基线

**配置**:
```python
CONFIG = {
    'dataset': 'CIFAR10',
    'model': 'SimpleCNN',
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 50,
    'local_epochs': 5,
    'batch_size': 64,
    'learning_rate': 0.01,
    'alpha': 0.5,
}
```

**运行**:
```bash
# 完整实验
python experiments/run_cifar10_paper.py

# 仅运行无攻击基线（快速测试）
python experiments/run_cifar10_paper.py --only-no-attack
```

**输出**: `experiments/results/cifar10_paper/results.json`

---

### 2. `run_cifar10_ablation.py` 🔬 **消融实验**
**用途**: CIFAR-10消融研究，测试不同noise scale的影响
**特点**:
- 支持命令行参数配置
- 测试多个Byzantine noise scale (0.3, 0.5等)
- 对比Vanilla FL和Trimmed Mean的鲁棒性
- 快速实验（10轮，2个local epochs）

**配置**:
```bash
# 默认配置
--num-rounds 10
--local-epochs 2
--batch-size 64
--malicious-ratio 0.2
--noise-scales 0.3,0.5
```

**运行**:
```bash
# 默认配置
python experiments/run_cifar10_ablation.py

# 自定义noise scales
python experiments/run_cifar10_ablation.py --noise-scales 0.1,0.3,0.5,1.0

# 更多轮次
python experiments/run_cifar10_ablation.py --num-rounds 20 --local-epochs 3
```

**输出**: `experiments/results/cifar10_ablation/results.json`

---

### 3. `run_cifar10_trimmed_beta_ablation.py` 🔬 **Beta参数消融**
**用途**: 专门测试Trimmed Mean的Beta参数影响
**特点**:
- 测试Beta值: [0.1, 0.2, 0.3]
- 固定Byzantine攻击场景（20%恶意，noise=0.1）
- 快速实验（10轮，2个local epochs）
- 依赖`run_cifar10_paper.py`的工具函数

**配置**:
```python
CONFIG = {
    'num_clients': 10,
    'clients_per_round': 5,
    'num_rounds': 10,
    'local_epochs': 2,
    'batch_size': 64,
    'learning_rate': 0.01,
    'alpha': 0.5,
}
BETAS = [0.1, 0.2, 0.3]
```

**运行**:
```bash
python experiments/run_cifar10_trimmed_beta_ablation.py
```

**输出**: `experiments/results/ablation/beta_results.json`

---

### 4. `run_cifar10_quick.py` ⚠️ **快速测试（已废弃）**
**用途**: 快速验证CIFAR-10实验代码
**特点**:
- 20轮，1个local epoch
- 简化配置
- 仅用于调试

**状态**: ❌ **建议删除** - 功能已被`run_cifar10_paper.py --only-no-attack`替代

---

### CIFAR-10脚本对比总结

| 脚本 | 轮次 | Local Epochs | 用途 | 状态 |
|------|------|--------------|------|------|
| `run_cifar10_paper.py` | 50 | 5 | 论文完整实验 | ✅ 保留 |
| `run_cifar10_ablation.py` | 10 | 2 | Noise scale消融 | ✅ 保留 |
| `run_cifar10_trimmed_beta_ablation.py` | 10 | 2 | Beta参数消融 | ✅ 保留 |
| `run_cifar10_quick.py` | 20 | 1 | 快速调试 | ❌ 删除 |

**推荐使用**:
- 论文数据: `run_cifar10_paper.py`
- 消融研究: `run_cifar10_ablation.py` (noise scale), `run_cifar10_trimmed_beta_ablation.py` (beta)
- 快速测试: `run_cifar10_paper.py --only-no-attack`

---

## Utility Scripts

### 1. `test_infrastructure.py` 🛠️
**用途**: 测试实验基础设施  
**测试内容**:
- 数据加载
- 模型创建
- 聚合器
- 攻击实现
- 指标计算

**运行**:
```bash
python experiments/test_infrastructure.py
```

### 2. `test_rq1_mini.py` 🛠️
**用途**: RQ1最小化测试  
**特点**: 3轮，3客户端，快速验证

---

## 脚本使用建议

### 论文投稿前
1. ✅ 运行`run_rq1_with_detection.py` - 获取完整RQ1数据
2. ✅ 运行`run_rq2_overhead.py` - 获取开销数据
3. ✅ 运行`run_rq3_scalability.py` - 获取可扩展性数据
4. ✅ 运行`run_rq4_incentive.py` - 获取激励机制数据
5. ⚠️ 可选: `run_rq1_resnet18.py` - 如果审稿人要求复杂模型

### 快速验证
1. `test_infrastructure.py` - 验证基础设施
2. `test_rq1_mini.py` - 快速RQ1测试
3. `run_rq1_pol_detection.py` - 快速Detection测试

### 调试
1. `run_cifar10_quick.py` - CIFAR-10快速测试
2. `test_rq1_mini.py` - 最小化RQ1测试

---

## 脚本清理建议

### 保留（核心功能）
- ✅ `run_rq1_security.py`
- ✅ `run_rq1_with_detection.py`
- ✅ `run_rq1_resnet18.py`
- ✅ `run_rq2_overhead.py`
- ✅ `run_rq3_scalability.py`
- ✅ `run_rq4_incentive.py`
- ✅ `run_cifar10_paper.py`
- ✅ `run_cifar10_ablation.py`
- ✅ `run_cifar10_trimmed_beta_ablation.py`
- ✅ `test_infrastructure.py`
- ✅ `test_rq1_mini.py`

### 可归档（功能重复）
- ⚠️ `run_rq1_pol_detection.py` - 功能已被`run_rq1_with_detection.py`包含

### 可删除（调试用）
- ❌ `run_cifar10_quick.py` - 仅用于快速测试

---

## 配置文件

所有脚本使用统一的配置文件：
- `experiment_config.py` - FL配置、PoL配置、输出配置
- `data_utils.py` - 数据加载和处理
- `models.py` - 模型创建
- `metrics.py` - 指标计算
- `baselines.py` - 基线聚合器

---

**维护者**: AI Assistant  
**最后更新**: 2025-10-22

