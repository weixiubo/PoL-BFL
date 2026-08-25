const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const ganache = require("ganache");
const solc = require("solc");
const { ethers } = require("ethers");

const root = path.resolve(__dirname, "..", "..");
const auditTicketDomain = ethers.sha256(
  ethers.toUtf8Bytes("POLBFL_CLIENT_AUDIT_TICKET_V2"),
);

function compile() {
  const files = ["PoLBFLProtocol.sol", "MockAuthenticatedRandomness.sol"];
  const sources = Object.fromEntries(
    files.map((file) => [file, { content: fs.readFileSync(path.join(root, "chainEnv", "contracts", file), "utf8") }]),
  );
  const output = JSON.parse(solc.compile(JSON.stringify({
    language: "Solidity",
    sources,
    settings: {
      optimizer: { enabled: true, runs: 200 },
      outputSelection: { "*": { "*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"] } },
    },
  })));
  const errors = (output.errors || []).filter((item) => item.severity === "error");
  if (errors.length) throw new Error(errors.map((item) => item.formattedMessage).join("\n"));
  return {
    protocol: output.contracts["PoLBFLProtocol.sol"].PoLBFLProtocol,
    randomness: output.contracts["MockAuthenticatedRandomness.sol"].MockAuthenticatedRandomness,
  };
}

async function deploy(factorySigner, artifact, args = []) {
  const factory = new ethers.ContractFactory(artifact.abi, artifact.evm.bytecode.object, factorySigner);
  const contract = await factory.deploy(...args);
  await contract.waitForDeployment();
  return contract;
}

function auditTicket(seed, roundId, commitmentRoot) {
  return ethers.sha256(ethers.concat([auditTicketDomain, seed, roundId, commitmentRoot]));
}

function findAuditingSeed(roundId, commitmentRoots) {
  for (let counter = 1n; counter < 2_000_000n; counter += 1n) {
    const seed = ethers.zeroPadValue(ethers.toBeHex(counter), 32);
    const allAudited = commitmentRoots.every(
      (rootValue) => BigInt(auditTicket(seed, roundId, rootValue)) % 10_000n < 2_000n,
    );
    if (allAudited) return seed;
  }
  throw new Error("could not find deterministic audit seed");
}

async function main() {
  const artifacts = compile();
  const eip170Bytes = artifacts.protocol.evm.deployedBytecode.object.length / 2;
  assert.ok(eip170Bytes < 24_576, `optimized runtime exceeds EIP-170: ${eip170Bytes}`);

  const ganacheProvider = ganache.provider({
    logging: { quiet: true },
    wallet: { totalAccounts: 12, defaultBalance: 1_000 },
    chain: { hardfork: "shanghai" },
  });
  const provider = new ethers.BrowserProvider(ganacheProvider);
  const initialAccounts = ganacheProvider.getInitialAccounts();
  const signers = await Promise.all(Array.from({ length: 10 }, (_, index) => provider.getSigner(index)));
  const addresses = await Promise.all(signers.map((signer) => signer.getAddress()));
  const governance = signers[0];
  const oracle = await deploy(governance, artifacts.randomness);
  const protocol = await deploy(governance, artifacts.protocol, [await oracle.getAddress(), addresses[0]]);
  await (await protocol.configureEconomics(ethers.parseEther("0.001"), 0, 0, 0)).wait();
  await (await protocol.fundRewards({ value: ethers.parseEther("1") })).wait();

  const stake = ethers.parseEther("0.05");
  const clientSigners = signers.slice(1, 4);
  const clientAddresses = addresses.slice(1, 4);
  const verifierSigners = signers.slice(5, 10);
  for (const signer of clientSigners) await (await protocol.connect(signer).registerClient({ value: stake })).wait();
  for (const signer of verifierSigners) await (await protocol.connect(signer).registerVerifier({ value: stake })).wait();

  const roundId = ethers.keccak256(ethers.toUtf8Bytes("paper-round-contract-e2e"));
  const latest = await provider.getBlock("latest");
  const commitDeadline = BigInt(latest.timestamp + 1_000);
  const auditDeadline = BigInt(latest.timestamp + 2_000);
  await (await protocol.createRound(roundId, commitDeadline, auditDeadline, 25)).wait();
  const commitmentRoots = clientSigners.map((_, index) =>
    ethers.keccak256(ethers.toUtf8Bytes(`commitment-${index}`)),
  );
  const updateDigests = clientSigners.map((_, index) =>
    ethers.keccak256(ethers.toUtf8Bytes(`update-${index}`)),
  );
  const commitmentGas = [];
  for (let index = 0; index < clientSigners.length; index += 1) {
    const receipt = await (await protocol.connect(clientSigners[index]).submitCommitment(
      roundId,
      commitmentRoots[index],
      updateDigests[index],
      25,
    )).wait();
    commitmentGas.push(receipt.gasUsed);
  }

  const seed = findAuditingSeed(roundId, commitmentRoots);
  await (await oracle.setOutput(roundId, seed)).wait();
  await provider.send("evm_increaseTime", [1_001]);
  await provider.send("evm_mine", []);
  await (await protocol.activateAudit(roundId, seed)).wait();
  for (let index = 0; index < clientAddresses.length; index += 1) {
    const client = clientAddresses[index];
    assert.equal(await protocol.auditTicket(roundId, client), auditTicket(seed, roundId, commitmentRoots[index]));
    assert.equal(await protocol.isAudited(roundId, client), true);
  }

  const committee = Array.from(await protocol.getRoundCommittee(roundId));
  assert.equal(committee.length, 5);
  assert.equal(new Set(committee.map((item) => item.toLowerCase())).size, 5);
  const messageWalletByAddress = new Map(
    Object.entries(initialAccounts).map(([address, details]) => [address.toLowerCase(), new ethers.Wallet(details.secretKey)]),
  );
  const proofAccept = ethers.keccak256(ethers.toUtf8Bytes("proof-set-accept"));
  const receiptGas = [];
  const acceptVerifiers = committee.slice(0, 3);
  const acceptSignatures = [];
  for (const verifier of acceptVerifiers) {
    const message = await protocol.receiptMessage(roundId, clientAddresses[0], proofAccept, true, verifier);
    acceptSignatures.push(await messageWalletByAddress.get(verifier.toLowerCase()).signMessage(ethers.getBytes(message)));
  }
  const acceptReceipt = await (await protocol.submitQuorumBySignatures(
    roundId, clientAddresses[0], proofAccept, true, acceptVerifiers, acceptSignatures,
  )).wait();
  receiptGas.push(acceptReceipt.gasUsed);
  const acceptedAudit = await protocol.audits(roundId, clientAddresses[0]);
  assert.equal(acceptedAudit.resolved, true);
  assert.equal(acceptedAudit.passed, true);

  const proofReject = ethers.keccak256(ethers.toUtf8Bytes("proof-set-reject"));
  const rejectVerifiers = committee.slice(0, 3);
  const rejectSignatures = [];
  for (const verifier of rejectVerifiers) {
    const message = await protocol.receiptMessage(roundId, clientAddresses[1], proofReject, false, verifier);
    rejectSignatures.push(await messageWalletByAddress.get(verifier.toLowerCase()).signMessage(ethers.getBytes(message)));
  }
  await (await protocol.submitQuorumBySignatures(
    roundId, clientAddresses[1], proofReject, false, rejectVerifiers, rejectSignatures,
  )).wait();
  const slashReceipt = await (await protocol.executeRejectedAudit(roundId, clientAddresses[1])).wait();
  assert.equal((await protocol.accounts(clientAddresses[1])).stake, 0n);

  await (await protocol.settleClient(roundId, clientAddresses[0], ethers.parseEther("1"), false, true)).wait();
  const acceptedAccount = await protocol.accounts(clientAddresses[0]);
  assert.equal(acceptedAccount.stake, stake);
  assert.equal(acceptedAccount.reputation, ethers.parseEther("0.55"));
  assert.equal(acceptedAccount.claimableReward, ethers.parseEther("0.001"));
  const rewardClaimReceipt = await (await protocol.connect(clientSigners[0]).claimReward()).wait();

  await provider.send("evm_increaseTime", [1_100]);
  await provider.send("evm_mine", []);
  await (await protocol.finalizeAuditTimeout(roundId, clientAddresses[2])).wait();
  assert.equal((await protocol.accounts(clientAddresses[2])).stake, 0n);
  await (await protocol.finalizeRound(roundId)).wait();
  assert.equal((await protocol.rounds(roundId)).finalized, true);
  for (const verifier of committee) assert.equal((await protocol.accounts(verifier)).locks, 0n);

  assert.ok(
    commitmentGas.every((value) => value <= 85_000n),
    `commitment gas exceeds paper target: ${commitmentGas.map(String).join(",")}`,
  );
  assert.ok(receiptGas.every((value) => value <= 120_000n), "receipt gas exceeds paper target");
  assert.ok(slashReceipt.gasUsed <= 65_000n, "slashing gas exceeds paper target");
  assert.ok(rewardClaimReceipt.gasUsed <= 45_000n, "reward claim gas exceeds paper target");

  process.stdout.write(`${JSON.stringify({
    runtime_bytes: eip170Bytes,
    commitment_gas: commitmentGas.map(String),
    receipt_gas: receiptGas.map(String),
    slash_gas: slashReceipt.gasUsed.toString(),
    reward_claim_gas: rewardClaimReceipt.gasUsed.toString(),
    slashed_clients: 2,
  })}\n`);
  if (typeof ganacheProvider.disconnect === "function") ganacheProvider.disconnect();
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
