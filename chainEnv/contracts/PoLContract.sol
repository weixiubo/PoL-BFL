// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;


interface IZKVerifier {
    function verifyProof(
        uint256[2] memory a,
        uint256[2][2] memory b,
        uint256[2] memory c,
        uint256[4] memory input
    ) external view returns (bool r);
}

/**
 * @title PoLContract
 * @dev Proof-of-Learning智能合约
 *
 * 核心功能:
 * 1. 提交PoL证明（commitment, data_hash等）
 * 2. 挑战机制（服务器挑战客户端）
 * 3. 记录验证结果
 * 4. Stake and reward mechanisms
 */
contract PoLContract {

    // ========== 数据结构 ==========

    /**
     * @dev PoL证明结构
     */
    struct PoLProof {
        bytes32 commitment;      // Merkle root
        bytes32 dataHash;        // 数据哈希
        uint256 numCheckpoints;  // checkpoint数量
        uint256 totalSteps;      // 总训练步数
        uint256 timestamp;       // 提交时间
        bool verified;           // 是否已验证
        bool isValid;            // 验证结果
    }

    /**
     * @dev 验证记录结构
     */
    struct VerificationRecord {
        address verifier;        // 验证者地址
        bool isValid;            // 验证结果
        uint256 timestamp;       // 验证时间
        string details;          // 验证详情（可选）
    }

    // ========== 状态变量 ==========

    address public owner;                                    // 合约所有者（服务器）

    mapping(address => PoLProof) public proofs;             // 客户端地址 => PoL证明
    mapping(address => VerificationRecord[]) public verificationHistory;  // 验证历史
    mapping(address => bool) public registeredClients;      // 已注册的客户端

    address[] public clientList;                            // 客户端列表

    uint256 public totalProofs;                             // 总证明数
    uint256 public totalVerifications;                      // 总验证数

    // ========== 经济激励相关状态变量 ==========

    mapping(address => uint256) public stakes;              // 客户端质押
    mapping(address => uint256) public lockedStakes;        // 锁定的质押
    mapping(address => uint256) public reputations;         // 声誉分数 (0-1000, 表示0.000-1.000)
    mapping(address => uint256) public totalRewards;        // 累计奖励

    uint256 public rewardPool;                              // 奖励池
    uint256 public penaltyPool;                             // 惩罚池（用于再分配）
    uint256 public minStake = 100 ether;                    // 最小质押要求

    uint256 public constant REPUTATION_SCALE = 1000;        // 声誉缩放因子

    // ========== ZKP Verifier integration (optional) ==========
    address public zkpVerifier;
    bool public useOnchainVerifier = false;


    // ========== 挑战机制状态 ==========
    struct Challenge {
        address client;
        uint256 idx0;
        uint256 idx1;
        uint256 issuedAt;
        uint256 deadline;
        bool resolved;
        bool success;
        string reason;
        uint256 W_t_hash;
        uint256 W_t1_hash;
        uint256 data_hash;
    }

    mapping(bytes32 => Challenge) public challenges;

    // ========== 事件 ==========

    event ProofSubmitted(
        address indexed client,
        bytes32 commitment,
        bytes32 dataHash,
        uint256 numCheckpoints,
        uint256 totalSteps,
        uint256 timestamp
    );

    event VerificationRecorded(
        address indexed client,
        address indexed verifier,
        bool isValid,
        uint256 timestamp
    );

    event ClientRegistered(
        address indexed client,
        uint256 timestamp
    );

    // 经济激励事件
    event Staked(
        address indexed client,
        uint256 amount,
        uint256 timestamp
    );

    event Unstaked(
        address indexed client,
        uint256 amount,
        uint256 timestamp
    );

    event Penalized(
        address indexed client,
        uint256 amount,
        string reason,
        uint256 timestamp
    );

    // 挑战事件
    event ChallengeIssued(
        bytes32 indexed challengeId,
        address indexed client,
        uint256 idx0,
        uint256 idx1,
        uint256 deadline,
        uint256 timestamp
    );

    event ChallengeResolved(
        bytes32 indexed challengeId,
        address indexed client,
        bool success,
        string reason,
        uint256 timestamp
    );

    event RewardDistributed(
        address indexed client,
        uint256 amount,
        uint256 timestamp
    );

    event ReputationUpdated(
        address indexed client,
        uint256 oldReputation,
        uint256 newReputation,
        uint256 timestamp
    );

    // ========== 修饰符 ==========

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    modifier onlyRegistered() {
        require(registeredClients[msg.sender], "Client not registered");
        _;
    }

    // ========== 构造函数 ==========

    constructor() {
        owner = msg.sender;
    }

    // ========== 核心功能 ==========

    /**
     * @dev 客户端注册
     */

    // ========== Admin: set verifier and toggle ==========
    function setVerifier(address _verifier, bool _use) external onlyOwner {
        zkpVerifier = _verifier;
        useOnchainVerifier = _use;
    }

    function registerClient() external {
        require(!registeredClients[msg.sender], "Client already registered");

        registeredClients[msg.sender] = true;
        clientList.push(msg.sender);

        emit ClientRegistered(msg.sender, block.timestamp);
    }

    /**
     * @dev 提交PoL证明
     * @param _commitment Merkle root
     * @param _dataHash 数据哈希
     * @param _numCheckpoints checkpoint数量
     * @param _totalSteps 总训练步数
     */
    function submitProof(
        bytes32 _commitment,
        bytes32 _dataHash,
        uint256 _numCheckpoints,
        uint256 _totalSteps
    ) external onlyRegistered {
        require(_commitment != bytes32(0), "Invalid commitment");
        require(_numCheckpoints > 0, "Invalid checkpoint count");
        require(_totalSteps > 0, "Invalid total steps");

        // 创建PoL证明
        proofs[msg.sender] = PoLProof({
            commitment: _commitment,
            dataHash: _dataHash,
            numCheckpoints: _numCheckpoints,
            totalSteps: _totalSteps,
            timestamp: block.timestamp,
            verified: false,
            isValid: false
        });

        totalProofs++;

        emit ProofSubmitted(
            msg.sender,
            _commitment,
            _dataHash,
            _numCheckpoints,
            _totalSteps,
            block.timestamp
        );
    }

    /**
     * @dev 记录验证结果（仅owner可调用）
     * @param _client 客户端地址
     * @param _isValid 验证结果
     */
    function recordVerification(
        address _client,
        bool _isValid
    ) external onlyOwner {
        require(registeredClients[_client], "Client not registered");
        require(proofs[_client].commitment != bytes32(0), "No proof submitted");

        // 更新证明状态
        proofs[_client].verified = true;
        proofs[_client].isValid = _isValid;

        // 记录验证历史
        verificationHistory[_client].push(VerificationRecord({
            verifier: msg.sender,
            isValid: _isValid,
            timestamp: block.timestamp,
            details: ""
        }));

        totalVerifications++;

        emit VerificationRecorded(
            _client,
            msg.sender,
            _isValid,
            block.timestamp
        );
    }

    /**
    // ========== 挑战机制 ==========
    /**
     * @dev 发起挑战（仅owner）
     */
    function issueChallenge(address _client, uint256 _idx0, uint256 _idx1, uint256 _deadline)
        external onlyOwner returns (bytes32)
    {
        require(registeredClients[_client], "Client not registered");
        require(_idx1 >= _idx0, "Invalid indices");
        require(_deadline > block.timestamp, "Invalid deadline");

        bytes32 challengeId = keccak256(abi.encodePacked(_client, _idx0, _idx1, block.number, block.timestamp));
        require(challenges[challengeId].issuedAt == 0, "Challenge exists");

        challenges[challengeId] = Challenge({
            client: _client,
            idx0: _idx0,
            idx1: _idx1,
            issuedAt: block.timestamp,
            deadline: _deadline,
            resolved: false,
            success: false,
            reason: "",
            W_t_hash: 0,
            W_t1_hash: 0,
            data_hash: 0
        });

        emit ChallengeIssued(challengeId, _client, _idx0, _idx1, _deadline, block.timestamp);
        return challengeId;
    }

    /**
     * @dev 提交挑战的ZKP结果（仅owner）
     * @dev Records public signals and results for proofs verified off chain.
     */
    function challengeProof(
        bytes32 challengeId,
        uint256 W_t_hash,
        uint256 W_t1_hash,
        uint256 data_hash,
        bool verified,
        string calldata reason
    ) external onlyOwner {
        Challenge storage ch = challenges[challengeId];
        require(ch.issuedAt != 0, "No such challenge");
        require(!ch.resolved, "Already resolved");
        require(block.timestamp <= ch.deadline, "Challenge expired");

        ch.resolved = true;
        ch.success = verified;
        ch.reason = reason;
        ch.W_t_hash = W_t_hash;
        ch.W_t1_hash = W_t1_hash;
        ch.data_hash = data_hash;

        emit ChallengeResolved(challengeId, ch.client, verified, reason, block.timestamp);
    }


    /**
     * @dev 提交挑战并在合约内联验证ZKP（仅owner）
     * 要求已通过 setVerifier 设置 verifier 地址并开启 useOnchainVerifier
     */
    function challengeProofOnchainVerify(
        bytes32 challengeId,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[4] calldata input,
        string calldata reason
    ) external onlyOwner {
        require(useOnchainVerifier, "On-chain verifier disabled");
        require(zkpVerifier != address(0), "Verifier not set");
        Challenge storage ch = challenges[challengeId];
        require(ch.issuedAt != 0, "No such challenge");
        require(!ch.resolved, "Already resolved");
        require(block.timestamp <= ch.deadline, "Challenge expired");

        bool ok = IZKVerifier(zkpVerifier).verifyProof(a, b, c, input);

        ch.resolved = true;
        ch.success = ok;
        ch.reason = reason;
        ch.W_t_hash = input[0];
        ch.W_t1_hash = input[1];
        ch.data_hash = input[2];

        emit ChallengeResolved(challengeId, ch.client, ok, reason, block.timestamp);
    }

    /**
     * @dev 查询挑战详情
     */
    function getChallenge(bytes32 challengeId) external view returns (
        address client,
        uint256 idx0,
        uint256 idx1,
        uint256 issuedAt,
        uint256 deadline,
        bool resolved,
        bool success,
        string memory reason,
        uint256 _W_t_hash,
        uint256 _W_t1_hash,
        uint256 _data_hash
    ) {
        Challenge storage ch = challenges[challengeId];
        require(ch.issuedAt != 0, "No such challenge");
        return (
            ch.client,
            ch.idx0,
            ch.idx1,
            ch.issuedAt,
            ch.deadline,
            ch.resolved,
            ch.success,
            ch.reason,
            ch.W_t_hash,
            ch.W_t1_hash,
            ch.data_hash
        );
    }


    function batchRecordVerification(
        address[] calldata _clients,
        bool[] calldata _results
    ) external onlyOwner {
        require(_clients.length == _results.length, "Array length mismatch");

        for (uint256 i = 0; i < _clients.length; i++) {
            if (registeredClients[_clients[i]] &&
                proofs[_clients[i]].commitment != bytes32(0)) {

                proofs[_clients[i]].verified = true;
                proofs[_clients[i]].isValid = _results[i];

                verificationHistory[_clients[i]].push(VerificationRecord({
                    verifier: msg.sender,
                    isValid: _results[i],
                    timestamp: block.timestamp,
                    details: ""
                }));

                totalVerifications++;

                emit VerificationRecorded(
                    _clients[i],
                    msg.sender,
                    _results[i],
                    block.timestamp
                );
            }
        }
    }

    // ========== 查询功能 ==========

    /**
     * @dev 获取客户端的PoL证明
     * @param _client 客户端地址
     */
    function getProof(address _client) external view returns (
        bytes32 commitment,
        bytes32 dataHash,
        uint256 numCheckpoints,
        uint256 totalSteps,
        uint256 timestamp,
        bool verified,
        bool isValid
    ) {
        PoLProof memory proof = proofs[_client];
        return (
            proof.commitment,
            proof.dataHash,
            proof.numCheckpoints,
            proof.totalSteps,
            proof.timestamp,
            proof.verified,
            proof.isValid
        );
    }

    /**
     * @dev 获取验证历史记录数量
     * @param _client 客户端地址
     */
    function getVerificationCount(address _client) external view returns (uint256) {
        return verificationHistory[_client].length;
    }

    /**
     * @dev 获取特定的验证记录
     * @param _client 客户端地址
     * @param _index 记录索引
     */
    function getVerificationRecord(address _client, uint256 _index) external view returns (
        address verifier,
        bool isValid,
        uint256 timestamp,
        string memory details
    ) {
        require(_index < verificationHistory[_client].length, "Index out of bounds");
        VerificationRecord memory record = verificationHistory[_client][_index];
        return (
            record.verifier,
            record.isValid,
            record.timestamp,
            record.details
        );
    }

    /**
     * @dev 获取已注册客户端数量
     */
    function getClientCount() external view returns (uint256) {
        return clientList.length;
    }

    /**
     * @dev 检查客户端是否已注册
     * @param _client 客户端地址
     */
    function isClientRegistered(address _client) external view returns (bool) {
        return registeredClients[_client];
    }

    /**
     * @dev 检查客户端是否已提交证明
     * @param _client 客户端地址
     */
    function hasProof(address _client) external view returns (bool) {
        return proofs[_client].commitment != bytes32(0);
    }

    /**
     * @dev 获取合约统计信息
     */
    function getStats() external view returns (
        uint256 _totalProofs,
        uint256 _totalVerifications,
        uint256 _totalClients
    ) {
        return (
            totalProofs,
            totalVerifications,
            clientList.length
        );
    }

    // ========== 经济激励功能 ==========

    /**
     * @dev 质押代币
     */
    function stake() external payable {
        require(msg.value > 0, "Stake amount must be positive");

        stakes[msg.sender] += msg.value;

        emit Staked(msg.sender, msg.value, block.timestamp);
    }

    /**
     * @dev 解质押代币
     * @param _amount 解质押数量
     */
    function unstake(uint256 _amount) external {
        require(_amount > 0, "Unstake amount must be positive");
        require(stakes[msg.sender] >= _amount, "Insufficient stake");
        require(stakes[msg.sender] - lockedStakes[msg.sender] >= _amount, "Stake is locked");

        stakes[msg.sender] -= _amount;

        // 转账给客户端
        payable(msg.sender).transfer(_amount);

        emit Unstaked(msg.sender, _amount, block.timestamp);
    }

    /**
     * @dev 锁定质押（仅owner可调用）
     * @param _client 客户端地址
     * @param _amount 锁定数量
     */
    function lockStake(address _client, uint256 _amount) external onlyOwner {
        require(stakes[_client] >= _amount, "Insufficient stake");
        require(stakes[_client] - lockedStakes[_client] >= _amount, "Already locked");

        lockedStakes[_client] += _amount;
    }

    /**
     * @dev 解锁质押（仅owner可调用）
     * @param _client 客户端地址
     * @param _amount 解锁数量
     */
    function unlockStake(address _client, uint256 _amount) external onlyOwner {
        require(lockedStakes[_client] >= _amount, "Insufficient locked stake");

        lockedStakes[_client] -= _amount;
    }

    /**
     * @dev 惩罚客户端（仅owner可调用）
     * @param _client 客户端地址
     * @param _amount 惩罚数量
     * @param _reason 惩罚原因
     */
    function penalize(address _client, uint256 _amount, string calldata _reason) external onlyOwner {
        require(stakes[_client] >= _amount, "Insufficient stake for penalty");

        stakes[_client] -= _amount;

        // 50%进入惩罚池（用于再分配）
        // 50%销毁（发送到0地址）
        uint256 redistribution = _amount / 2;
        uint256 burned = _amount - redistribution;

        penaltyPool += redistribution;
        payable(address(0)).transfer(burned);

        emit Penalized(_client, _amount, _reason, block.timestamp);
    }

    /**
     * @dev 分配奖励（仅owner可调用）
     * @param _client 客户端地址
     * @param _amount 奖励数量
     */
    function distributeReward(address _client, uint256 _amount) external onlyOwner {
        require(rewardPool >= _amount, "Insufficient reward pool");

        rewardPool -= _amount;
        totalRewards[_client] += _amount;

        // 转账给客户端
        payable(_client).transfer(_amount);

        emit RewardDistributed(_client, _amount, block.timestamp);
    }

    /**
     * @dev 批量分配奖励（仅owner可调用）
     * @param _clients 客户端地址数组
     * @param _amounts 奖励数量数组
     */
    function batchDistributeRewards(
        address[] calldata _clients,
        uint256[] calldata _amounts
    ) external onlyOwner {
        require(_clients.length == _amounts.length, "Array length mismatch");

        uint256 totalAmount = 0;
        for (uint256 i = 0; i < _amounts.length; i++) {
            totalAmount += _amounts[i];
        }

        require(rewardPool >= totalAmount, "Insufficient reward pool");

        for (uint256 i = 0; i < _clients.length; i++) {
            rewardPool -= _amounts[i];
            totalRewards[_clients[i]] += _amounts[i];

            payable(_clients[i]).transfer(_amounts[i]);

            emit RewardDistributed(_clients[i], _amounts[i], block.timestamp);
        }
    }

    /**
     * @dev 更新声誉（仅owner可调用）
     * @param _client 客户端地址
     * @param _newReputation 新声誉分数 (0-1000)
     */
    function updateReputation(address _client, uint256 _newReputation) external onlyOwner {
        require(_newReputation <= REPUTATION_SCALE, "Reputation out of range");

        uint256 oldReputation = reputations[_client];
        reputations[_client] = _newReputation;

        emit ReputationUpdated(_client, oldReputation, _newReputation, block.timestamp);
    }

    /**
     * @dev 充值奖励池（任何人可调用）
     */
    function fundRewardPool() external payable {
        require(msg.value > 0, "Fund amount must be positive");
        rewardPool += msg.value;
    }

    /**
     * @dev 获取客户端质押信息
     * @param _client 客户端地址
     */
    function getStakeInfo(address _client) external view returns (
        uint256 totalStake,
        uint256 locked,
        uint256 available
    ) {
        totalStake = stakes[_client];
        locked = lockedStakes[_client];
        available = totalStake - locked;
        return (totalStake, locked, available);
    }

    /**
     * @dev 获取客户端声誉
     * @param _client 客户端地址
     */
    function getReputation(address _client) external view returns (uint256) {
        return reputations[_client];
    }

    /**
     * @dev 获取经济激励统计信息
     */
    function getIncentiveStats() external view returns (
        uint256 _rewardPool,
        uint256 _penaltyPool,
        uint256 _totalStaked,
        uint256 _minStake
    ) {
        uint256 totalStaked = 0;
        for (uint256 i = 0; i < clientList.length; i++) {
            totalStaked += stakes[clientList[i]];
        }

        return (
            rewardPool,
            penaltyPool,
            totalStaked,
            minStake
        );
    }
}
