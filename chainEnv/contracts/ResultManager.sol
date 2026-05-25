// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Result Manager for PoL verification summaries (Phase A minimal)
contract ResultManager {
    struct VerificationRecord {
        bytes32 commitmentRoot;
        bool success;
        address reporter; // committee leader or submitter
        uint256 timestamp;
    }

    mapping(bytes32 => VerificationRecord) public records; // keyed by commitmentRoot

    event VerificationSubmitted(bytes32 indexed commitmentRoot, bool success, address indexed reporter);

    function submit(bytes32 commitmentRoot, bool success) external {
        records[commitmentRoot] = VerificationRecord({
            commitmentRoot: commitmentRoot,
            success: success,
            reporter: msg.sender,
            timestamp: block.timestamp
        });
        emit VerificationSubmitted(commitmentRoot, success, msg.sender);
    }

    function get(bytes32 commitmentRoot) external view returns (VerificationRecord memory) {
        return records[commitmentRoot];
    }
}

