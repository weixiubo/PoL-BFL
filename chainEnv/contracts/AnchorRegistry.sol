// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AnchorRegistry
 * @dev Lightweight on-chain anchoring for PoL verification rounds
 * 
 * Purpose:
 * - Record verification round digests on-chain for auditability
 * - Minimal gas cost (~50-80k per anchor)
 * - Decoupled from main PoLContract for modularity
 * 
 * Use case:
 * - Aggregator anchors (round_id, commit_hash, sigset_hash) after each round
 * - commit_hash = SHA256({round_id, {client_commitments}})
 * - sigset_hash = SHA256(sorted verifier addresses)
 * - Provides tamper-proof audit trail
 * 
 * Security:
 * - Only authorized anchors (aggregators) can submit
 * - Immutable once anchored (no updates)
 * - Events for off-chain indexing
 */
contract AnchorRegistry {
    
    // ========== Structs ==========
    
    struct RoundAnchor {
        bytes32 roundId;         // Round identifier (e.g., hash of round number + timestamp)
        bytes32 commitHash;      // Hash of all client commitments in this round
        bytes32 sigsetHash;      // Hash of verifier signature set
        address aggregator;      // Who anchored this round
        uint256 timestamp;       // When anchored
        uint256 blockNumber;     // Block number
    }
    
    // ========== State Variables ==========
    
    address public owner;
    
    // Round anchors: roundId => RoundAnchor
    mapping(bytes32 => RoundAnchor) public anchors;
    
    // Authorized aggregators (can anchor rounds)
    mapping(address => bool) public authorizedAggregators;
    
    // Statistics
    uint256 public totalAnchors;
    
    // ========== Events ==========
    
    event RoundAnchored(
        bytes32 indexed roundId,
        bytes32 commitHash,
        bytes32 sigsetHash,
        address indexed aggregator,
        uint256 timestamp,
        uint256 blockNumber
    );
    
    event AggregatorAuthorized(
        address indexed aggregator,
        bool authorized,
        uint256 timestamp
    );
    
    // ========== Modifiers ==========
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyAuthorized() {
        require(authorizedAggregators[msg.sender], "Not authorized aggregator");
        _;
    }
    
    // ========== Constructor ==========
    
    constructor() {
        owner = msg.sender;
        authorizedAggregators[msg.sender] = true;
    }
    
    // ========== Admin Functions ==========
    
    /**
     * @dev Authorize or revoke aggregator
     */
    function authorizeAggregator(address aggregator, bool authorized) external onlyOwner {
        require(aggregator != address(0), "Invalid address");
        authorizedAggregators[aggregator] = authorized;
        emit AggregatorAuthorized(aggregator, authorized, block.timestamp);
    }
    
    /**
     * @dev Transfer ownership
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
    
    // ========== Core Functions ==========
    
    /**
     * @dev Anchor a verification round
     * 
     * @param roundId Round identifier (unique)
     * @param commitHash Hash of all client commitments
     * @param sigsetHash Hash of verifier signature set
     * 
     * Gas cost: ~50-80k
     * 
     * Example:
     *   roundId = keccak256(abi.encodePacked(roundNumber, timestamp))
     *   commitHash = sha256({client1_commit, client2_commit, ...})
     *   sigsetHash = sha256(sorted([verifier1_addr, verifier2_addr, ...]))
     */
    function anchorRound(
        bytes32 roundId,
        bytes32 commitHash,
        bytes32 sigsetHash
    ) external onlyAuthorized returns (bool) {
        require(roundId != bytes32(0), "Invalid round ID");
        require(anchors[roundId].timestamp == 0, "Round already anchored");
        
        anchors[roundId] = RoundAnchor({
            roundId: roundId,
            commitHash: commitHash,
            sigsetHash: sigsetHash,
            aggregator: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number
        });
        
        totalAnchors++;
        
        emit RoundAnchored(
            roundId,
            commitHash,
            sigsetHash,
            msg.sender,
            block.timestamp,
            block.number
        );
        
        return true;
    }
    
    /**
     * @dev Get anchor information
     */
    function getAnchor(bytes32 roundId) external view returns (
        bytes32 commitHash,
        bytes32 sigsetHash,
        address aggregator,
        uint256 timestamp,
        uint256 blockNumber
    ) {
        RoundAnchor storage anchor = anchors[roundId];
        require(anchor.timestamp != 0, "Round not anchored");
        
        return (
            anchor.commitHash,
            anchor.sigsetHash,
            anchor.aggregator,
            anchor.timestamp,
            anchor.blockNumber
        );
    }
    
    /**
     * @dev Check if round is anchored
     */
    function isAnchored(bytes32 roundId) external view returns (bool) {
        return anchors[roundId].timestamp != 0;
    }
    
    /**
     * @dev Verify anchor matches expected values
     */
    function verifyAnchor(
        bytes32 roundId,
        bytes32 expectedCommitHash,
        bytes32 expectedSigsetHash
    ) external view returns (bool) {
        RoundAnchor storage anchor = anchors[roundId];
        if (anchor.timestamp == 0) {
            return false;
        }
        
        return anchor.commitHash == expectedCommitHash 
            && anchor.sigsetHash == expectedSigsetHash;
    }
    
    /**
     * @dev Get statistics
     */
    function getStats() external view returns (
        uint256 total,
        uint256 uniqueAggregators
    ) {
        total = totalAnchors;
        // Note: uniqueAggregators would require additional tracking
        // For simplicity, we only return total anchors
        uniqueAggregators = 0;
    }
}


/**
 * Usage Example:
 * 
 * Deploy:
 *   registry = AnchorRegistry.deploy({'from': accounts[0]})
 *   registry.authorizeAggregator(aggregator_address, True)
 * 
 * Anchor a round:
 *   round_id = web3.keccak(text=f"round_{round_num}_{timestamp}")
 *   commit_hash = sha256({client commitments})
 *   sigset_hash = sha256(sorted verifier addresses)
 *   tx = registry.anchorRound(round_id, commit_hash, sigset_hash, {'from': aggregator})
 * 
 * Query:
 *   anchor = registry.getAnchor(round_id)
 *   is_valid = registry.verifyAnchor(round_id, expected_commit, expected_sigset)
 * 
 * Gas costs:
 *   - anchorRound: ~50-80k gas
 *   - getAnchor: free (view function)
 *   - verifyAnchor: free (view function)
 * 
 * Benefits:
 *   - Tamper-proof audit trail
 *   - Minimal gas cost
 *   - Decoupled from main PoL logic
 *   - Easy to query and verify
 */

