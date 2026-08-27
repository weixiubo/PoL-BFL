# ERC-20 Brownie Test Fixture

## Purpose

This directory contains an ERC-20 contract fixture used by the Brownie-based
contract tests. The fixture originates from the Brownie token
mix and is distributed under its accompanying MIT License.

The PoL-BFL settlement protocol is implemented separately in
`chainEnv/contracts/PoLBFLProtocol.sol`.

## Dependencies

The fixture requires an Ethereum Brownie release in the range declared by
`requirements.txt` and a Brownie-compatible local test network.

```bash
python -m pip install -r chainEnv/token/requirements.txt
```

## Test execution

```bash
cd chainEnv/token
brownie test
```

The tests deploy `contracts/Token.sol` to an isolated local chain and verify
the contract behavior defined by the fixture test suite.

## Upstream references

- [ERC-20 specification](https://eips.ethereum.org/EIPS/eip-20)
- [Ethereum Brownie documentation](https://eth-brownie.readthedocs.io/en/stable/)
- [Brownie mix organization](https://github.com/brownie-mix/)

## License

The fixture retains the copyright and permission notice in
[`LICENSE`](LICENSE).
