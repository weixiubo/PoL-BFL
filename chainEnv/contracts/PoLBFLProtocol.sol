// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IAuthenticatedRandomness {
    function verifyRandomness(bytes32 roundId, bytes32 output) external view returns (bool);
}

/// @title PoL-BFL paper protocol settlement
/// @notice Commit-before-challenge, VRF-derived 3-of-5 verification, full
/// slashing, reputation rewards, and strict separation of Layer-1 and Layer-2.
contract PoLBFLProtocol {
    uint256 public constant WAD = 1e18;
    uint256 public constant BPS = 10_000;
    uint256 public constant COMMITTEE_SIZE = 5;
    uint256 public constant COMMITTEE_THRESHOLD = 3;
    bytes32 public constant AUDIT_TICKET_DOMAIN =
        0x57af282f15188622133c51f7bb649311273ff151f9eb9eeb9a788f70831290e3;
    uint256 private constant SECP256K1_HALF_ORDER =
        0x7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0;
    bytes32 private constant RECEIPT_TYPEHASH = keccak256(
        "PoLBFLReceipt(address protocol,uint256 chainId,bytes32 roundId,address client,bytes32 proofSetDigest,bool valid,address verifier)"
    );

    enum Role { None, Client, Verifier }
    enum Vote { None, Accept, Reject }

    struct Account {
        uint256 stake;
        uint256 reputation;
        uint256 claimableReward;
        uint64 locks;
        Role role;
        bool active;
    }

    struct Round {
        address aggregator;
        uint64 commitDeadline;
        uint64 auditDeadline;
        uint64 expectedSteps;
        bytes32 randomness;
        bool auditActive;
        bool finalized;
        uint32 submittedClients;
        uint32 settledClients;
    }

    struct Submission {
        bytes32 commitmentRoot;
    }

    struct Audit {
        bytes32 proofSetDigest;
        uint8 accepts;
        uint8 rejects;
        bool resolved;
        bool passed;
        bool settled;
    }

    address public governance;
    IAuthenticatedRandomness public randomnessOracle;
    address public gasOracle;

    uint256 public baseMinimumStake = 0.05 ether;
    uint256 public minimumClientStake = 0.05 ether;
    uint256 public minimumVerifierStake = 0.05 ether;
    uint256 public slashingRatio = WAD;
    uint256 public reputationAlpha = 9e17;
    uint256 public challengeProbabilityBps = 2_000;
    uint256 public detectionProbability = 965e15;
    uint256 public baseReward;
    uint256 public betaWork;
    uint256 public betaReputation;
    uint256 public verifierReward;
    uint256 public rewardPool;

    mapping(address => Account) public accounts;
    mapping(address => bool) public authorizedAggregators;
    address[] public verifierList;
    mapping(bytes32 => Round) public rounds;
    mapping(bytes32 => address[]) private roundCommittee;
    mapping(bytes32 => mapping(address => bool)) public isCommitteeMember;
    mapping(bytes32 => mapping(address => Submission)) public submissions;
    mapping(bytes32 => mapping(address => Audit)) public audits;
    mapping(bytes32 => mapping(address => mapping(address => Vote))) public votes;
    bool private entered;

    event Registered(address indexed participant, Role role, uint256 stake);
    event StakeAdded(address indexed participant, uint256 amount);
    event RoundCreated(bytes32 indexed roundId, address indexed aggregator, uint64 commitDeadline, uint64 auditDeadline);
    event CommitmentSubmitted(bytes32 indexed roundId, address indexed client, bytes32 commitmentRoot, bytes32 updateDigest);
    event AuditActivated(bytes32 indexed roundId, bytes32 randomness, address[5] committee);
    event ReceiptRecorded(bytes32 indexed roundId, address indexed client, address indexed verifier, bytes32 proofSetDigest, bool valid);
    event AuditResolved(bytes32 indexed roundId, address indexed client, bool passed, bool timedOut);
    event ClientSettled(bytes32 indexed roundId, address indexed client, bool eligible, uint256 reward, uint256 reputation);
    event Slashed(address indexed participant, bytes32 indexed roundId, uint256 amount, bytes32 reason);
    event RoundFinalized(bytes32 indexed roundId);
    event MinimumStakeUpdated(uint256 oldValue, uint256 newValue, uint256 gasPriceWei);
    event AggregatorAuthorized(address indexed aggregator, bool authorized);
    event VerifierEquivocationProved(bytes32 indexed roundId, address indexed client, address indexed verifier);

    modifier onlyGovernance() {
        require(msg.sender == governance, "governance only");
        _;
    }

    modifier nonReentrant() {
        require(!entered, "reentrant call");
        entered = true;
        _;
        entered = false;
    }

    constructor(address randomnessOracle_, address gasOracle_) {
        require(randomnessOracle_ != address(0) && gasOracle_ != address(0), "oracle required");
        governance = msg.sender;
        randomnessOracle = IAuthenticatedRandomness(randomnessOracle_);
        gasOracle = gasOracle_;
        authorizedAggregators[msg.sender] = true;
    }

    function authorizeAggregator(address aggregator, bool authorized) external onlyGovernance {
        require(aggregator != address(0) && accounts[aggregator].role != Role.Verifier, "invalid aggregator");
        authorizedAggregators[aggregator] = authorized;
        emit AggregatorAuthorized(aggregator, authorized);
    }

    function configureEconomics(
        uint256 baseReward_,
        uint256 betaWork_,
        uint256 betaReputation_,
        uint256 verifierReward_
    ) external onlyGovernance {
        require(betaWork_ <= 10 * WAD && betaReputation_ <= 10 * WAD, "beta out of range");
        baseReward = baseReward_;
        betaWork = betaWork_;
        betaReputation = betaReputation_;
        verifierReward = verifierReward_;
    }

    function registerClient() external payable {
        _register(Role.Client, minimumClientStake);
    }

    function registerVerifier() external payable {
        _register(Role.Verifier, minimumVerifierStake);
        verifierList.push(msg.sender);
    }

    function _register(Role role, uint256 minimum) internal {
        require(accounts[msg.sender].role == Role.None, "already registered");
        require(msg.value >= minimum, "stake below minimum");
        accounts[msg.sender] = Account(msg.value, WAD / 2, 0, 0, role, true);
        emit Registered(msg.sender, role, msg.value);
    }

    function addStake() external payable {
        Account storage account = accounts[msg.sender];
        require(account.role != Role.None && msg.value > 0, "invalid stake deposit");
        account.stake += msg.value;
        account.active = true;
        emit StakeAdded(msg.sender, msg.value);
    }

    function withdrawStake(uint256 amount) external nonReentrant {
        Account storage account = accounts[msg.sender];
        require(account.locks == 0 && amount > 0 && account.stake >= amount, "stake unavailable");
        account.stake -= amount;
        uint256 minimum = account.role == Role.Client ? minimumClientStake : minimumVerifierStake;
        account.active = account.stake >= minimum;
        (bool ok,) = payable(msg.sender).call{value: amount}("");
        require(ok, "stake transfer failed");
    }

    function fundRewards() external payable {
        require(msg.value > 0, "empty reward funding");
        rewardPool += msg.value;
    }

    function claimReward() external nonReentrant {
        uint256 amount = accounts[msg.sender].claimableReward;
        require(amount > 0, "no reward");
        accounts[msg.sender].claimableReward = 0;
        (bool ok,) = payable(msg.sender).call{value: amount}("");
        require(ok, "reward transfer failed");
    }

    function createRound(bytes32 roundId, uint64 commitDeadline, uint64 auditDeadline, uint64 expectedSteps) external {
        require(roundId != bytes32(0) && rounds[roundId].aggregator == address(0), "round exists");
        require(commitDeadline > block.timestamp && auditDeadline > commitDeadline && expectedSteps > 0, "invalid round parameters");
        require(authorizedAggregators[msg.sender] && accounts[msg.sender].role != Role.Verifier, "aggregator unauthorized");
        rounds[roundId] = Round(msg.sender, commitDeadline, auditDeadline, expectedSteps, bytes32(0), false, false, 0, 0);
        emit RoundCreated(roundId, msg.sender, commitDeadline, auditDeadline);
    }

    function submitCommitment(
        bytes32 roundId,
        bytes32 commitmentRoot,
        bytes32 updateDigest,
        uint64 totalSteps
    ) external {
        Round storage round = rounds[roundId];
        Account storage account = accounts[msg.sender];
        require(round.aggregator != address(0) && block.timestamp <= round.commitDeadline, "commit phase closed");
        require(account.role == Role.Client && account.active && account.stake >= minimumClientStake, "client ineligible");
        require(submissions[roundId][msg.sender].commitmentRoot == bytes32(0), "duplicate commitment");
        require(commitmentRoot != bytes32(0) && updateDigest != bytes32(0), "empty commitment");
        require(totalSteps == round.expectedSteps, "invalid trace bounds");
        submissions[roundId][msg.sender] = Submission(commitmentRoot);
        round.submittedClients += 1;
        account.locks += 1;
        emit CommitmentSubmitted(roundId, msg.sender, commitmentRoot, updateDigest);
    }

    function activateAudit(bytes32 roundId, bytes32 randomness) external {
        Round storage round = rounds[roundId];
        require(msg.sender == round.aggregator && !round.auditActive, "aggregator or phase mismatch");
        require(block.timestamp > round.commitDeadline && block.timestamp < round.auditDeadline, "outside audit activation window");
        require(randomnessOracle.verifyRandomness(roundId, randomness), "invalid VRF output");
        address[5] memory committee = _selectCommittee(round.aggregator, randomness);
        round.randomness = randomness;
        round.auditActive = true;
        for (uint256 index = 0; index < COMMITTEE_SIZE; index++) {
            roundCommittee[roundId].push(committee[index]);
            isCommitteeMember[roundId][committee[index]] = true;
            accounts[committee[index]].locks += 1;
        }
        emit AuditActivated(roundId, randomness, committee);
    }

    function _selectCommittee(address aggregator, bytes32 randomness) internal view returns (address[5] memory selected) {
        uint256 totalStake;
        uint256 eligibleCount;
        for (uint256 index = 0; index < verifierList.length; index++) {
            Account storage account = accounts[verifierList[index]];
            if (verifierList[index] != aggregator && account.active && account.stake >= minimumVerifierStake && account.reputation > 0) {
                totalStake += account.stake;
                eligibleCount++;
            }
        }
        require(eligibleCount >= COMMITTEE_SIZE && totalStake > 0, "insufficient verifier pool");
        uint256[5] memory scores;
        for (uint256 index = 0; index < verifierList.length; index++) {
            address candidate = verifierList[index];
            Account storage account = accounts[candidate];
            if (candidate == aggregator || !account.active || account.stake < minimumVerifierStake || account.reputation == 0) continue;
            uint256 ticket = uint256(keccak256(abi.encodePacked(randomness, candidate))) >> 128;
            uint256 score = ((ticket * account.stake) / totalStake * account.reputation) / WAD;
            for (uint256 position = 0; position < COMMITTEE_SIZE; position++) {
                if (score > scores[position] || (score == scores[position] && candidate < selected[position])) {
                    for (uint256 shift = COMMITTEE_SIZE - 1; shift > position; shift--) {
                        scores[shift] = scores[shift - 1];
                        selected[shift] = selected[shift - 1];
                    }
                    scores[position] = score;
                    selected[position] = candidate;
                    break;
                }
            }
        }
        require(selected[COMMITTEE_SIZE - 1] != address(0), "committee selection failed");
    }

    function auditTicket(bytes32 roundId, address client) public view returns (bytes32) {
        Round storage round = rounds[roundId];
        bytes32 commitmentRoot = submissions[roundId][client].commitmentRoot;
        require(round.auditActive && commitmentRoot != bytes32(0), "audit not active");
        return sha256(abi.encodePacked(
            AUDIT_TICKET_DOMAIN,
            round.randomness,
            roundId,
            commitmentRoot
        ));
    }

    function isAudited(bytes32 roundId, address client) public view returns (bool) {
        return uint256(auditTicket(roundId, client)) % BPS < challengeProbabilityBps;
    }

    function submitReceipt(bytes32 roundId, address client, bytes32 proofSetDigest, bool valid) external {
        _recordReceipt(roundId, client, proofSetDigest, valid, msg.sender);
    }

    function submitReceiptBySignature(
        bytes32 roundId,
        address client,
        bytes32 proofSetDigest,
        bool valid,
        address verifier,
        bytes calldata signature
    ) external {
        require(_recoverReceipt(roundId, client, proofSetDigest, valid, verifier, signature) == verifier, "invalid receipt signature");
        _recordReceipt(roundId, client, proofSetDigest, valid, verifier);
    }

    function submitQuorumBySignatures(
        bytes32 roundId,
        address client,
        bytes32 proofSetDigest,
        bool valid,
        address[] calldata verifiers,
        bytes[] calldata signatures
    ) external {
        Round storage round = rounds[roundId];
        Audit storage audit = audits[roundId][client];
        require(round.auditActive && block.timestamp <= round.auditDeadline, "audit closed");
        require(isAudited(roundId, client) && !audit.resolved, "quorum unavailable");
        require(proofSetDigest != bytes32(0) && verifiers.length == signatures.length, "invalid quorum payload");
        require(verifiers.length >= COMMITTEE_THRESHOLD && verifiers.length <= COMMITTEE_SIZE, "invalid quorum size");
        for (uint256 index = 0; index < verifiers.length; index++) {
            address verifier = verifiers[index];
            require(isCommitteeMember[roundId][verifier], "non-committee signer");
            for (uint256 prior = 0; prior < index; prior++) require(verifiers[prior] != verifier, "duplicate signer");
            require(
                _recoverReceipt(roundId, client, proofSetDigest, valid, verifier, signatures[index]) == verifier,
                "invalid quorum signature"
            );
        }
        audit.proofSetDigest = proofSetDigest;
        audit.accepts = valid ? uint8(verifiers.length) : 0;
        audit.rejects = valid ? 0 : uint8(verifiers.length);
        audit.resolved = true;
        audit.passed = valid;
        emit AuditResolved(roundId, client, valid, false);
    }

    function executeRejectedAudit(bytes32 roundId, address client) external {
        Audit storage audit = audits[roundId][client];
        require(audit.resolved && !audit.passed && !audit.settled, "rejection unavailable");
        _slashClient(roundId, client, keccak256("PROOF_REJECTED"));
    }

    function _recordReceipt(bytes32 roundId, address client, bytes32 proofSetDigest, bool valid, address verifier) internal {
        Round storage round = rounds[roundId];
        require(round.auditActive && block.timestamp <= round.auditDeadline, "audit closed");
        require(isCommitteeMember[roundId][verifier] && isAudited(roundId, client), "receipt unauthorized");
        Audit storage audit = audits[roundId][client];
        require(!audit.resolved && votes[roundId][client][verifier] == Vote.None, "receipt already final");
        require(proofSetDigest != bytes32(0), "empty proof digest");
        if (audit.proofSetDigest == bytes32(0)) audit.proofSetDigest = proofSetDigest;
        require(audit.proofSetDigest == proofSetDigest, "proof set mismatch");
        votes[roundId][client][verifier] = valid ? Vote.Accept : Vote.Reject;
        if (valid) audit.accepts += 1; else audit.rejects += 1;
        emit ReceiptRecorded(roundId, client, verifier, proofSetDigest, valid);
        if (audit.accepts >= COMMITTEE_THRESHOLD) _resolveAudit(roundId, client, true, false);
        else if (audit.rejects >= COMMITTEE_THRESHOLD) _resolveAudit(roundId, client, false, false);
    }

    function receiptMessage(
        bytes32 roundId,
        address client,
        bytes32 proofSetDigest,
        bool valid,
        address verifier
    ) public view returns (bytes32) {
        return keccak256(abi.encode(
            RECEIPT_TYPEHASH,
            address(this),
            block.chainid,
            roundId,
            client,
            proofSetDigest,
            valid,
            verifier
        ));
    }

    function _recoverReceipt(
        bytes32 roundId,
        address client,
        bytes32 proofSetDigest,
        bool valid,
        address verifier,
        bytes calldata signature
    ) internal view returns (address) {
        if (signature.length != 65) return address(0);
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := calldataload(signature.offset)
            s := calldataload(add(signature.offset, 32))
            v := byte(0, calldataload(add(signature.offset, 64)))
        }
        if (uint256(s) > SECP256K1_HALF_ORDER || (v != 27 && v != 28)) return address(0);
        bytes32 digest = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", receiptMessage(
            roundId, client, proofSetDigest, valid, verifier
        )));
        return ecrecover(digest, v, r, s);
    }

    function proveVerifierEquivocation(
        bytes32 roundId,
        address client,
        bytes32 firstProofDigest,
        bool firstValid,
        bytes calldata firstSignature,
        bytes32 secondProofDigest,
        bool secondValid,
        bytes calldata secondSignature,
        address verifier
    ) external {
        require(isCommitteeMember[roundId][verifier], "not a round verifier");
        require(firstProofDigest != secondProofDigest || firstValid != secondValid, "receipts do not conflict");
        require(_recoverReceipt(roundId, client, firstProofDigest, firstValid, verifier, firstSignature) == verifier, "invalid first signature");
        require(_recoverReceipt(roundId, client, secondProofDigest, secondValid, verifier, secondSignature) == verifier, "invalid second signature");
        Account storage account = accounts[verifier];
        uint256 penalty = account.stake * slashingRatio / WAD;
        require(penalty > 0, "verifier already slashed");
        account.stake -= penalty;
        account.active = account.stake >= minimumVerifierStake;
        account.reputation = reputationAlpha * account.reputation / WAD;
        emit Slashed(verifier, roundId, penalty, keccak256("VERIFIER_EQUIVOCATION"));
        emit VerifierEquivocationProved(roundId, client, verifier);
    }

    function finalizeAuditTimeout(bytes32 roundId, address client) external {
        Round storage round = rounds[roundId];
        require(round.auditActive && block.timestamp > round.auditDeadline, "deadline not reached");
        require(isAudited(roundId, client) && !audits[roundId][client].resolved, "timeout unavailable");
        _resolveAudit(roundId, client, false, true);
    }

    function _resolveAudit(bytes32 roundId, address client, bool passed, bool timedOut) internal {
        Audit storage audit = audits[roundId][client];
        require(!audit.resolved, "audit resolved");
        audit.resolved = true;
        audit.passed = passed;
        _settleVerifierService(roundId, client, passed);
        if (!passed) _slashClient(roundId, client, timedOut ? keccak256("AUDIT_TIMEOUT") : keccak256("PROOF_REJECTED"));
        emit AuditResolved(roundId, client, passed, timedOut);
    }

    function _settleVerifierService(bytes32 roundId, address client, bool result) internal {
        address[] storage committee = roundCommittee[roundId];
        uint256 correctVotes;
        for (uint256 index = 0; index < committee.length; index++) {
            Vote vote = votes[roundId][client][committee[index]];
            if ((result && vote == Vote.Accept) || (!result && vote == Vote.Reject)) correctVotes++;
        }
        uint256 totalReward = correctVotes * verifierReward;
        require(rewardPool >= totalReward, "verifier reward pool insufficient");
        rewardPool -= totalReward;
        for (uint256 index = 0; index < committee.length; index++) {
            Account storage verifier = accounts[committee[index]];
            Vote vote = votes[roundId][client][committee[index]];
            if ((result && vote == Vote.Accept) || (!result && vote == Vote.Reject)) {
                verifier.claimableReward += verifierReward;
                verifier.reputation = reputationAlpha * verifier.reputation / WAD + (WAD - reputationAlpha);
            } else if (vote != Vote.None) {
                verifier.reputation = reputationAlpha * verifier.reputation / WAD;
            }
        }
    }

    function settleClient(
        bytes32 roundId,
        address client,
        uint256 normalizedWork,
        bool sybilFlagged,
        bool statisticallyAccepted
    ) external {
        Round storage round = rounds[roundId];
        require(msg.sender == round.aggregator && round.auditActive && !round.finalized, "settlement unauthorized");
        require(submissions[roundId][client].commitmentRoot != bytes32(0) && !audits[roundId][client].settled && normalizedWork <= WAD, "invalid client settlement");
        bool audited = isAudited(roundId, client);
        if (audited) require(audits[roundId][client].resolved && audits[roundId][client].passed, "Layer-1 not passed");
        bool eligible = !sybilFlagged && statisticallyAccepted;
        _settleEligibleClient(roundId, client, normalizedWork, eligible);
    }

    function _settleEligibleClient(bytes32 roundId, address client, uint256 work, bool eligible) internal {
        Round storage round = rounds[roundId];
        Account storage account = accounts[client];
        uint256 reward;
        if (eligible) {
            reward = baseReward + (baseReward * betaWork / WAD * work / WAD)
                + (baseReward * betaReputation / WAD * account.reputation / WAD);
            require(rewardPool >= reward, "reward pool insufficient");
            rewardPool -= reward;
            account.claimableReward += reward;
        }
        account.reputation = reputationAlpha * account.reputation / WAD
            + (WAD - reputationAlpha) * (eligible ? WAD : 0) / WAD;
        audits[roundId][client].settled = true;
        if (account.locks > 0) account.locks -= 1;
        round.settledClients += 1;
        emit ClientSettled(roundId, client, eligible, reward, account.reputation);
    }

    function _slashClient(bytes32 roundId, address client, bytes32 reason) internal {
        Account storage account = accounts[client];
        uint256 penalty = account.stake * slashingRatio / WAD;
        account.stake -= penalty;
        account.active = account.stake >= minimumClientStake;
        account.reputation = reputationAlpha * account.reputation / WAD;
        if (!audits[roundId][client].settled) {
            audits[roundId][client].settled = true;
            if (account.locks > 0) account.locks -= 1;
            rounds[roundId].settledClients += 1;
        }
        emit Slashed(client, roundId, penalty, reason);
    }

    function finalizeExpiredClients(bytes32 roundId, address[] calldata clients) external {
        Round storage round = rounds[roundId];
        require(round.auditActive && !round.finalized && block.timestamp > round.auditDeadline, "round not finalizable");
        for (uint256 index = 0; index < clients.length; index++) {
            address client = clients[index];
            require(submissions[roundId][client].commitmentRoot != bytes32(0), "unknown round client");
            if (audits[roundId][client].settled) continue;
            if (isAudited(roundId, client) && !audits[roundId][client].resolved) {
                _resolveAudit(roundId, client, false, true);
            } else if (isAudited(roundId, client) && !audits[roundId][client].passed) {
                _slashClient(roundId, client, keccak256("PROOF_REJECTED"));
            } else {
                _settleEligibleClient(roundId, client, 0, false);
            }
        }
    }

    function finalizeRound(bytes32 roundId) external {
        Round storage round = rounds[roundId];
        require(round.auditActive && !round.finalized && block.timestamp > round.auditDeadline, "round not finalizable");
        require(round.settledClients == round.submittedClients, "unsettled clients remain");
        address[] storage committee = roundCommittee[roundId];
        for (uint256 index = 0; index < committee.length; index++) {
            Account storage verifier = accounts[committee[index]];
            if (verifier.locks > 0) verifier.locks -= 1;
        }
        round.finalized = true;
        emit RoundFinalized(roundId);
    }

    function updateMinimumStake(uint256 gasPriceWei, uint256 operationsGas, uint256 marginWei) external {
        require(msg.sender == gasOracle && operationsGas > 0, "gas oracle only");
        uint256 detection = challengeProbabilityBps * WAD / BPS * detectionProbability / WAD;
        uint256 denominator = slashingRatio * detection / WAD;
        require(denominator > 0, "invalid economic denominator");
        uint256 responsive = gasPriceWei * operationsGas * WAD / denominator + marginWei;
        uint256 old = minimumClientStake;
        minimumClientStake = responsive > baseMinimumStake ? responsive : baseMinimumStake;
        emit MinimumStakeUpdated(old, minimumClientStake, gasPriceWei);
    }

    function getRoundCommittee(bytes32 roundId) external view returns (address[] memory) {
        return roundCommittee[roundId];
    }
}
