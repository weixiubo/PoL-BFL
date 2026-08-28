"""
PoL配置文件
包含PoL相关的所有配置参数
"""
import os


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


# ========== 基础PoL配置 ==========

POL_CONFIG = {
    # 是否启用PoL
    'enable': True,

    # Checkpoint保存频率（每N个batch保存一次）
    'save_freq': 10,

    # PoL数据保存目录（允许通过环境变量覆盖）
    'save_dir': os.getenv('POL_SAVE_DIR') or os.getenv('POL_CHECKPOINTS_DIR') or 'pol_data',

    # 是否压缩checkpoint
    'compress': _env_bool('POL_COMPRESS_CHECKPOINTS', True),

    # 是否使用独立I/O线程异步保存
    'async_save': False,

    # ========== 验证配置 ==========

    # 验证比例（0-1，表示验证多少比例的客户端）
    'verification_rate': 0.3,

    # 参数距离阈值
    'delta': 100.0,

    # 距离度量方式 ('l1', 'l2', 'linf', 'cosine')
    'distance_metric': 'l2',

    # 是否使用Top-Q验证策略
    'use_top_q': False,

    # Top-Q的Q值（只验证Q个计算量最大的步骤）
    'top_q': 5,

    # ========== 存储优化配置 ==========

    # 是否保存checkpoint到磁盘（False=仅内存，True=保存到磁盘）
    'save_checkpoints_to_disk': _env_bool('POL_SAVE_CHECKPOINTS_TO_DISK', _env_bool('POL_SAVE_TO_DISK', True)),

    # 内存模式下保留的checkpoint数量（仅当save_checkpoints_to_disk=False时有效）
    'memory_checkpoint_limit': _env_int('POL_MEMORY_CHECKPOINT_LIMIT', 5),

    # 是否启用自动清理（定期删除旧checkpoint）
    'enable_auto_cleanup': True,  # Coordinated with the verification lock.

    # 自动清理间隔（每N个checkpoint清理一次）
    'auto_cleanup_interval': 50,

    # Checkpoint清理策略（保留每N个checkpoint）
    'cleanup_keep_every_n': 10,

    # 是否启用增量存储（只保存参数变化）
    'use_delta_encoding': False,

    # 是否启用量化压缩
    'use_quantization': False,

    # 量化位数
    'quantization_bits': 8,

    # ========== ZKP configuration ==========

    # 是否启用ZKP
    'use_zkp': False,

    # ZKP电路路径
    'zkp_circuit_path': 'circuits/sgd_update.circom',

    # ZKP证明密钥路径
    'zkp_proving_key': 'circuits/proving_key.json',

    # ZKP验证密钥路径
    'zkp_verification_key': 'circuits/verification_key.json',

    # ========== Economic incentive configuration ==========

    # 是否启用质押机制
    'enable_staking': False,

    # 最小质押金额
    'min_stake': 1.0,

    # 是否启用动态奖励
    'enable_dynamic_reward': False,

    # 基础奖励
    'base_reward': 1.0,

    # 计算奖励系数
    'computation_reward_coef': 0.5,

    # 贡献度奖励系数
    'contribution_reward_coef': 0.3,

    # 声誉奖励系数
    'reputation_reward_coef': 0.2,

    # 是否启用声誉系统
    'enable_reputation': False,

    # 初始声誉分数
    'initial_reputation': 500,

    # 验证通过奖励分数
    'verification_pass_score': 10,

    # 验证失败惩罚分数
    'verification_fail_score': -50,

    # 是否启用女巫攻击防御
    'enable_sybil_defense': False,

    # 女巫检测阈值
    'sybil_detection_threshold': 0.9,
}


# ========== 实验配置 ==========

EXPERIMENT_CONFIG = {
    # 数据集
    'dataset': 'MNIST',  # 'MNIST', 'CIFAR10', 'CIFAR100'

    # 模型
    'model': 'SimpleCNN',  # 'SimpleCNN', 'ResNet18', 'VGG16'

    # 客户端数量
    'num_clients': 10,

    # 每轮参与的客户端数量
    'clients_per_round': 5,

    # 训练轮数
    'num_rounds': 50,

    # 本地训练epoch数
    'local_epochs': 5,

    # 批次大小
    'batch_size': 32,

    # 学习率
    'learning_rate': 0.01,

    # 优化器
    'optimizer': 'SGD',  # 'SGD', 'Adam'

    # 权重衰减
    'weight_decay': 1e-4,

    # 设备
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',

    # 数据分布
    'data_distribution': 'iid',  # 'iid', 'non-iid'

    # Non-IID参数（Dirichlet分布的alpha）
    'non_iid_alpha': 0.5,
}


# ========== 攻击场景配置（用于实验）==========

ATTACK_CONFIG = {
    # 是否启用攻击
    'enable_attack': False,

    # 攻击类型 ('byzantine', 'free-riding', 'model-poisoning')
    'attack_type': 'byzantine',

    # 恶意客户端比例
    'malicious_ratio': 0.2,

    # Byzantine攻击参数
    'byzantine_scale': 10.0,

    # Free-riding攻击参数（不训练，直接提交初始模型）
    'free_riding_prob': 1.0,
}


# ========== 日志配置 ==========

LOGGING_CONFIG = {
    # 日志级别
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'

    # 日志文件路径
    'log_file': 'logs/pol_veryfl.log',

    # 是否输出到控制台
    'console': True,

    # 日志格式
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}


# ========== 辅助函数 ==========

def get_pol_config():
    """获取PoL配置"""
    return POL_CONFIG.copy()


def get_experiment_config():
    """获取实验配置"""
    return EXPERIMENT_CONFIG.copy()


def get_attack_config():
    """获取攻击配置"""
    return ATTACK_CONFIG.copy()


def get_logging_config():
    """获取日志配置"""
    return LOGGING_CONFIG.copy()


def merge_configs(*configs):
    """合并多个配置字典"""
    merged = {}
    for config in configs:
        merged.update(config)
    return merged


def get_full_config():
    """获取完整配置"""
    return merge_configs(
        POL_CONFIG,
        EXPERIMENT_CONFIG,
        ATTACK_CONFIG,
        LOGGING_CONFIG
    )


# ========== 配置验证 ==========

def validate_config(config):
    """
    验证配置的有效性

    Args:
        config: 配置字典

    Returns:
        is_valid: 是否有效
        errors: 错误列表
    """
    errors = []

    # 验证verification_rate
    if 'verification_rate' in config:
        if not 0 <= config['verification_rate'] <= 1:
            errors.append("verification_rate must be between 0 and 1")

    # 验证delta
    if 'delta' in config:
        if config['delta'] <= 0:
            errors.append("delta must be positive")

    # 验证save_freq
    if 'save_freq' in config:
        if config['save_freq'] <= 0:
            errors.append("save_freq must be positive")

    # 验证top_q
    if 'use_top_q' in config and config['use_top_q']:
        if 'top_q' not in config or config['top_q'] <= 0:
            errors.append("top_q must be positive when use_top_q is True")

    # 验证ZKP配置
    if 'use_zkp' in config and config['use_zkp']:
        required_zkp_keys = ['zkp_circuit_path', 'zkp_proving_key', 'zkp_verification_key']
        for key in required_zkp_keys:
            if key not in config:
                errors.append(f"{key} is required when use_zkp is True")

    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    # 测试配置
    print("=" * 60)
    print("PoL Configuration Test")
    print("=" * 60)

    print("\nPoL Config:")
    for key, value in POL_CONFIG.items():
        print(f"  {key}: {value}")

    print("\nExperiment Config:")
    for key, value in EXPERIMENT_CONFIG.items():
        print(f"  {key}: {value}")

    print("\nValidating configuration...")
    full_config = get_full_config()
    is_valid, errors = validate_config(full_config)

    if is_valid:
        print("[PASS] Configuration is valid")
    else:
        print("[FAIL] Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
