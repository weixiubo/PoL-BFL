pragma circom 2.0.0;

/*
 * 参数更新零知识证明电路
 *
 * 目标：证明参数更新的合法性，同时保护训练数据隐私
 *
 * 公开输入：
 *   - W_t_hash: 当前参数的哈希
 *   - W_t1_hash: 更新后参数的哈希
 *   - data_hash: 使用的数据批次哈希
 *   - max_distance: 允许的最大参数变化距离
 *
 * 私有输入：
 *   - W_t: 当前参数（保密）
 *   - W_t1: 更新后参数（保密）
 *   - data_indices: 使用的数据索引（保密）
 *
 * 验证内容：
 *   1. W_t的哈希确实是W_t_hash
 *   2. W_t1的哈希确实是W_t1_hash
 *   3. data_indices的哈希确实是data_hash
 *   4. ||W_t1 - W_t||^2 <= max_distance
 */

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";

template ParameterUpdateProof(param_size, batch_size) {
    // ========== 公开输入 ==========
    signal input W_t_hash;
    signal input W_t1_hash;
    signal input data_hash;
    signal input max_distance;

    // ========== 私有输入 ==========
    signal input W_t[param_size];
    signal input W_t1[param_size];
    signal input data_indices[batch_size];

    // ========== 约束1: 验证W_t的哈希（折叠Poseidon，t=2） ==========
    signal wt_acc[param_size + 1];
    wt_acc[0] <== 0;
    component pose_wt[param_size];
    for (var i = 0; i < param_size; i++) {
        pose_wt[i] = Poseidon(2);
        pose_wt[i].inputs[0] <== wt_acc[i];
        pose_wt[i].inputs[1] <== W_t[i];
        wt_acc[i + 1] <== pose_wt[i].out;
    }
    wt_acc[param_size] === W_t_hash;

    // ========== 约束2: 验证W_{t+1}的哈希（折叠Poseidon，t=2） ==========
    signal wt1_acc[param_size + 1];
    wt1_acc[0] <== 0;
    component pose_wt1[param_size];
    for (var i = 0; i < param_size; i++) {
        pose_wt1[i] = Poseidon(2);
        pose_wt1[i].inputs[0] <== wt1_acc[i];
        pose_wt1[i].inputs[1] <== W_t1[i];
        wt1_acc[i + 1] <== pose_wt1[i].out;
    }
    wt1_acc[param_size] === W_t1_hash;

    // ========== 约束3: 验证数据索引哈希（折叠Poseidon，t=2） ==========
    signal data_acc[batch_size + 1];
    data_acc[0] <== 0;
    component pose_data[batch_size];
    for (var i = 0; i < batch_size; i++) {
        pose_data[i] = Poseidon(2);
        pose_data[i].inputs[0] <== data_acc[i];
        pose_data[i].inputs[1] <== data_indices[i];
        data_acc[i + 1] <== pose_data[i].out;
    }
    data_acc[batch_size] === data_hash;

    // ========== 约束4: 验证参数变化在合理范围内 ==========
    signal distance_squared;
    signal partial_sum[param_size];
    signal diffs[param_size];

    // 计算L2距离的平方
    for (var i = 0; i < param_size; i++) {
        diffs[i] <== W_t1[i] - W_t[i];
        if (i == 0) {
            partial_sum[i] <== diffs[i] * diffs[i];
        } else {
            partial_sum[i] <== partial_sum[i-1] + diffs[i] * diffs[i];
        }
    }

    distance_squared <== partial_sum[param_size - 1];

    // 距离约束：distance_squared <= max_distance
    component less_than = LessThan(252);
    less_than.in[0] <== distance_squared;
    less_than.in[1] <== max_distance;
    less_than.out === 1;
}

// 主组件：实例化电路
// Reference parameter capacity: 100
// 批次大小：32
component main {public [W_t_hash, W_t1_hash, data_hash, max_distance]} = ParameterUpdateProof(100, 32);


/*
 * 使用说明：
 *
 * 1. 编译电路：
 *    circom parameter_update.circom --r1cs --wasm --sym
 *
 * 2. 生成witness：
 *    node parameter_update_js/generate_witness.js parameter_update_js/parameter_update.wasm input.json witness.wtns
 *
 * 3. 生成proving key（需要Powers of Tau ceremony）：
 *    snarkjs groth16 setup parameter_update.r1cs pot12_final.ptau parameter_update_0000.zkey
 *
 * 4. 生成证明：
 *    snarkjs groth16 prove parameter_update_0000.zkey witness.wtns proof.json public.json
 *
 * 5. 验证证明：
 *    snarkjs groth16 verify verification_key.json public.json proof.json
 *
 * 6. 生成Solidity验证器：
 *    snarkjs zkey export solidityverifier parameter_update_0000.zkey verifier.sol
 */


/*
 * 电路复杂度分析：
 *
 * 约束数量估算：
 * - Poseidon哈希（3次）: ~150 * 3 = 450约束
 * - L2距离计算: param_size * 2 = 200约束
 * - LessThan比较: ~252约束
 * 总计: ~900约束
 *
 * 证明生成时间（估算）：
 * - 在普通笔记本上: ~1-2秒
 * - 在服务器上: <1秒
 *
 * 证明大小：
 * - Groth16证明: 固定128字节
 * - 公开输入: 4 * 32字节 = 128字节
 * 总计: ~256字节
 *
 * 验证Gas成本（估算）：
 * - 链上验证: ~250,000 Gas
 * - 约$5-10（取决于Gas价格）
 */


/*
 * 安全性分析：
 *
 * 1. 隐私保护：
 *    - W_t, W_t1, data_indices完全保密
 *    - 只暴露哈希值，无法反推原始数据
 *
 * 2. 防作弊：
 *    - 无法伪造满足哈希约束的参数
 *    - 距离约束防止参数异常变化
 *
 * 3. 局限性：
 *    - 不验证训练逻辑的正确性
 *    - 依赖max_distance参数的合理设置
 *    - 可能存在满足约束但非真实训练的参数
 *
 * 4. 缓解策略：
 *    - 结合随机抽样明文验证
 *    - 长期观察客户端行为
 *    - 声誉系统动态调整验证频率
 */


/*
 * 扩展方向：
 *
 * 1. 增加梯度约束：
 *    - 验证梯度的范数在合理范围内
 *    - 需要额外的私有输入和约束
 *
 * 2. 多步验证：
 *    - 一次证明多个连续的更新步骤
 *    - 增加电路复杂度但减少证明次数
 *
 * 3. 模型特定优化：
 *    - 针对特定模型结构（如CNN）优化电路
 *    - 利用模型稀疏性减少约束
 *
 * 4. 递归证明：
 *    - 使用递归SNARK压缩多个证明
 *    - 需要更高级的ZKP技术
 */
