// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title Node/Client Registry (Phase A minimal)
contract Registry {
    struct NodeInfo {
        address addr;
        string endpoint; // e.g., http(s) URL
        string role;     // "verifier" | "aggregator" | "client"
        uint256 stake;
        uint256 reputation;
        bool active;
    }

    mapping(address => NodeInfo) public nodes;
    address[] public nodeList;

    event Registered(address indexed addr, string role, string endpoint);
    event Updated(address indexed addr, string role, string endpoint, uint256 stake, uint256 reputation, bool active);

    function register(string calldata role, string calldata endpoint) external {
        require(bytes(role).length > 0, "role required");
        NodeInfo storage n = nodes[msg.sender];
        if (n.addr == address(0)) {
            n.addr = msg.sender;
            nodeList.push(msg.sender);
        }
        n.role = role;
        n.endpoint = endpoint;
        n.active = true;
        emit Registered(msg.sender, role, endpoint);
    }

    function update(string calldata role, string calldata endpoint, uint256 stake, uint256 reputation, bool active) external {
        NodeInfo storage n = nodes[msg.sender];
        require(n.addr != address(0), "not registered");
        n.role = role;
        n.endpoint = endpoint;
        n.stake = stake;
        n.reputation = reputation;
        n.active = active;
        emit Updated(msg.sender, role, endpoint, stake, reputation, active);
    }

    function getAll() external view returns (NodeInfo[] memory out) {
        out = new NodeInfo[](nodeList.length);
        for (uint i=0; i<nodeList.length; i++) {
            out[i] = nodes[nodeList[i]];
        }
    }
}

