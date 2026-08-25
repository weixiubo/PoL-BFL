const fs = require("node:fs");
const assert = require("node:assert/strict");
const ganache = require("ganache");
const solc = require("solc");
const { ethers } = require("ethers");

function argumentsFrom(argv) {
  const values = {};
  for (let index = 2; index < argv.length; index += 2) {
    values[argv[index]] = argv[index + 1];
  }
  if (!values["--verifier"] || !values["--proof"]) {
    throw new Error("--verifier and --proof are required");
  }
  return values;
}

function compile(source) {
  const output = JSON.parse(
    solc.compile(
      JSON.stringify({
        language: "Solidity",
        sources: {"verifier.sol": {content: source}},
        settings: {
          optimizer: {enabled: true, runs: 200},
          outputSelection: {
            "*": {"*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"]},
          },
        },
      }),
    ),
  );
  const errors = (output.errors || []).filter((item) => item.severity === "error");
  if (errors.length) {
    throw new Error(errors.map((item) => item.formattedMessage).join("\n"));
  }
  return output.contracts["verifier.sol"].Verifier;
}

async function main() {
  const args = argumentsFrom(process.argv);
  const artifact = compile(fs.readFileSync(args["--verifier"], "utf8"));
  const proof = JSON.parse(fs.readFileSync(args["--proof"], "utf8"));
  const providerHandle = ganache.provider({
    logging: {quiet: true},
    wallet: {totalAccounts: 2, defaultBalance: 1000},
    chain: {hardfork: "shanghai"},
  });
  const provider = new ethers.BrowserProvider(providerHandle);
  const signer = await provider.getSigner(0);
  const factory = new ethers.ContractFactory(
    artifact.abi,
    artifact.evm.bytecode.object,
    signer,
  );
  const verifier = await factory.deploy();
  await verifier.waitForDeployment();
  const values = proof.proof;
  const solidityProof = {
    a: {X: BigInt(values.a[0]), Y: BigInt(values.a[1])},
    b: {
      X: [BigInt(values.b[0][0]), BigInt(values.b[0][1])],
      Y: [BigInt(values.b[1][0]), BigInt(values.b[1][1])],
    },
    c: {X: BigInt(values.c[0]), Y: BigInt(values.c[1])},
  };
  const inputs = proof.inputs.map((value) => BigInt(value));
  assert.equal(await verifier.verifyTx(solidityProof, inputs), true);
  const populated = await verifier.verifyTx.populateTransaction(
    solidityProof,
    inputs,
  );
  const transaction = await signer.sendTransaction({
    to: await verifier.getAddress(),
    data: populated.data,
    gasLimit: 5_000_000,
  });
  const receipt = await transaction.wait();
  assert.equal(receipt.status, 1);
  const output = {
    passed: true,
    verification_gas: receipt.gasUsed.toString(),
    transaction_hash: receipt.hash,
    runtime_bytes: artifact.evm.deployedBytecode.object.length / 2,
    contract_address: await verifier.getAddress(),
    chain_id: (await provider.getNetwork()).chainId.toString(),
    tool_versions: {
      node: process.version,
      solc: solc.version(),
      ethers: ethers.version,
      ganache: ganache.__experimental_info().version,
    },
  };
  process.stdout.write(JSON.stringify(output) + "\n");
  if (typeof providerHandle.disconnect === "function") {
    providerHandle.disconnect();
  }
}

main().catch((error) => {
  process.stderr.write(String(error.stack || error) + "\n");
  process.exitCode = 1;
});
