// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Verifier Committee Election (Phase A minimal)
/// @notice MVP uses simple filtering; randomness/VRF can be added later.
interface IRegistry {
    struct NodeInfo {
        address addr;
        string endpoint;
        string role;
        uint256 stake;
        uint256 reputation;
        bool active;
    }
    function nodes(address) external view returns (NodeInfo memory);
    function getAll() external view returns (NodeInfo[] memory);
}

contract CommitteeElection {
    IRegistry public registry;
    uint256 public minStake;
    uint256 public minReputation;

    event CommitteeSelected(address[] committee, uint256 epoch);

    constructor(address registryAddr, uint256 _minStake, uint256 _minRep) {
        registry = IRegistry(registryAddr);
        minStake = _minStake; minReputation = _minRep;
    }

    function selectVerifierCommittee(uint256 epoch, uint256 maxSize) external view returns (address[] memory members) {
        IRegistry.NodeInfo[] memory allNodes = registry.getAll();
        uint count = 0;
        // First pass: count
        for (uint i=0; i<allNodes.length; i++) {
            if (allNodes[i].active && keccak256(bytes(allNodes[i].role)) == keccak256(bytes("verifier"))
                && allNodes[i].stake >= minStake && allNodes[i].reputation >= minReputation) {
                count++;
            }
        }
        if (count > maxSize) count = maxSize;
        members = new address[](count);
        uint idx = 0;
        for (uint i=0; i<allNodes.length && idx<count; i++) {
            if (allNodes[i].active && keccak256(bytes(allNodes[i].role)) == keccak256(bytes("verifier"))
                && allNodes[i].stake >= minStake && allNodes[i].reputation >= minReputation) {
                members[idx++] = allNodes[i].addr;
            }
        }
    }
}

