// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title ZKPVerifier
 * @dev 零知识证明验证合约
 * 
 * 功能:
 * 1. 验证SGD更新步骤的ZKP证明
 * 2. 支持链上验证
 * 3. 记录验证结果
 */
contract ZKPVerifier {
    
    // ========== 数据结构 ==========
    
    /**
     * @dev ZKP证明结构
     */
    struct ZKProof {
        bytes32 W_t_hash;           // 初始权重哈希
        bytes32 W_t_plus_1_hash;    // 更新后权重哈希
        bytes32 D_hash;             // 数据哈希
        uint256 learning_rate;      // 学习率
        uint256 batch_size;         // 批次大小
        uint256 step_number;        // 训练步数
        uint256 l2_error;           // L2误差
        bool is_valid;              // 是否有效
        uint256 timestamp;          // 时间戳
    }
    
    /**
     * @dev 验证结果结构
     */
    struct VerificationResult {
        address prover;             // 证明者地址
        bool is_valid;              // 验证结果
        uint256 timestamp;          // 验证时间
        string reason;              // 失败原因
    }
    
    // ========== 状态变量 ==========
    
    address public owner;
    
    mapping(bytes32 => ZKProof) public proofs;                  // 证明ID => 证明
    mapping(bytes32 => VerificationResult[]) public verificationHistory;  // 验证历史
    
    bytes32[] public proofIds;                                  // 所有证明ID列表
    
    uint256 public totalProofs;                                 // 总证明数
    uint256 public totalVerifications;                          // 总验证数
    uint256 public successfulVerifications;                     // 成功验证数
    
    // 验证参数
    uint256 public tolerance = 100;                             // 容许误差（0.01 * 10000）
    uint256 public maxL2Error = 1000;                           // 最大L2误差
    
    // ========== 事件 ==========
    
    event ProofSubmitted(
        bytes32 indexed proofId,
        address indexed prover,
        bytes32 W_t_hash,
        bytes32 W_t_plus_1_hash,
        uint256 timestamp
    );
    
    event ProofVerified(
        bytes32 indexed proofId,
        address indexed verifier,
        bool is_valid,
        uint256 l2_error,
        uint256 timestamp
    );
    
    event ToleranceUpdated(uint256 new_tolerance);
    
    // ========== 修饰符 ==========
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }
    
    // ========== 构造函数 ==========
    
    constructor() {
        owner = msg.sender;
    }
    
    // ========== 核心函数 ==========
    
    /**
     * @dev 提交ZKP证明
     */
    function submitProof(
        bytes32 W_t_hash,
        bytes32 W_t_plus_1_hash,
        bytes32 D_hash,
        uint256 learning_rate,
        uint256 batch_size,
        uint256 step_number,
        uint256 l2_error
    ) external returns (bytes32) {
        // 生成证明ID
        bytes32 proofId = keccak256(abi.encodePacked(
            msg.sender,
            W_t_hash,
            W_t_plus_1_hash,
            block.timestamp
        ));
        
        // 检查证明是否已存在
        require(proofs[proofId].timestamp == 0, "Proof already exists");
        
        // 创建证明对象
        ZKProof memory proof = ZKProof({
            W_t_hash: W_t_hash,
            W_t_plus_1_hash: W_t_plus_1_hash,
            D_hash: D_hash,
            learning_rate: learning_rate,
            batch_size: batch_size,
            step_number: step_number,
            l2_error: l2_error,
            is_valid: false,
            timestamp: block.timestamp
        });
        
        // 存储证明
        proofs[proofId] = proof;
        proofIds.push(proofId);
        totalProofs++;
        
        // 发出事件
        emit ProofSubmitted(
            proofId,
            msg.sender,
            W_t_hash,
            W_t_plus_1_hash,
            block.timestamp
        );
        
        return proofId;
    }
    
    /**
     * @dev 验证ZKP证明
     */
    function verifyProof(bytes32 proofId) external returns (bool) {
        // 获取证明
        ZKProof storage proof = proofs[proofId];
        require(proof.timestamp != 0, "Proof not found");
        
        // 验证L2误差
        bool is_valid = proof.l2_error <= tolerance;
        
        // 更新证明状态
        proof.is_valid = is_valid;
        
        // 记录验证结果
        VerificationResult memory result = VerificationResult({
            prover: msg.sender,
            is_valid: is_valid,
            timestamp: block.timestamp,
            reason: is_valid ? "Verification passed" : "L2 error exceeds tolerance"
        });
        
        verificationHistory[proofId].push(result);
        totalVerifications++;
        
        if (is_valid) {
            successfulVerifications++;
        }
        
        // 发出事件
        emit ProofVerified(
            proofId,
            msg.sender,
            is_valid,
            proof.l2_error,
            block.timestamp
        );
        
        return is_valid;
    }
    
    /**
     * @dev 批量验证证明
     */
    function batchVerifyProofs(bytes32[] calldata proofIds_) external returns (uint256) {
        uint256 successCount = 0;
        
        for (uint256 i = 0; i < proofIds_.length; i++) {
            if (this.verifyProof(proofIds_[i])) {
                successCount++;
            }
        }
        
        return successCount;
    }
    
    // ========== 查询函数 ==========
    
    /**
     * @dev 获取证明信息
     */
    function getProof(bytes32 proofId) external view returns (ZKProof memory) {
        return proofs[proofId];
    }
    
    /**
     * @dev 获取验证历史
     */
    function getVerificationHistory(bytes32 proofId) 
        external 
        view 
        returns (VerificationResult[] memory) 
    {
        return verificationHistory[proofId];
    }
    
    /**
     * @dev 获取所有证明ID
     */
    function getAllProofIds() external view returns (bytes32[] memory) {
        return proofIds;
    }
    
    /**
     * @dev 获取验证统计
     */
    function getVerificationStats() external view returns (
        uint256 total,
        uint256 successful,
        uint256 failed
    ) {
        return (
            totalVerifications,
            successfulVerifications,
            totalVerifications - successfulVerifications
        );
    }
    
    // ========== 管理函数 ==========
    
    /**
     * @dev 更新容许误差
     */
    function setTolerance(uint256 new_tolerance) external onlyOwner {
        tolerance = new_tolerance;
        emit ToleranceUpdated(new_tolerance);
    }
    
    /**
     * @dev 更新最大L2误差
     */
    function setMaxL2Error(uint256 new_max_error) external onlyOwner {
        maxL2Error = new_max_error;
    }
}

