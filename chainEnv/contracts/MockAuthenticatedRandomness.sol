// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract MockAuthenticatedRandomness {
    mapping(bytes32 => bytes32) public outputs;

    function setOutput(bytes32 roundId, bytes32 output) external {
        outputs[roundId] = output;
    }

    function verifyRandomness(bytes32 roundId, bytes32 output) external view returns (bool) {
        return output != bytes32(0) && outputs[roundId] == output;
    }
}
