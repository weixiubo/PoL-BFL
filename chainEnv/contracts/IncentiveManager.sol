// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Incentive Manager (Phase A minimal)
contract IncentiveManager {
    event RewardsRecorded(address indexed submitter, address[] clients, uint256[] amounts);

    function recordRewards(address[] calldata clients, uint256[] calldata amounts) external {
        require(clients.length == amounts.length, "length mismatch");
        emit RewardsRecorded(msg.sender, clients, amounts);
    }
}

