const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const ganache = require("ganache");
const solc = require("solc");
const { ethers } = require("ethers");

const root = path.resolve(__dirname, "..");
const WAD = 10n ** 18n;
const CLIENT_STAKE = ethers.parseEther("0.05");
const PROVIDER_POLLING_INTERVAL_MS = 10;
const COMMIT_WINDOW_SECONDS = 3_600n;
const AUDIT_WINDOW_SECONDS = 3_600n;
const AUDIT_TICKET_DOMAIN = ethers.sha256(
  ethers.toUtf8Bytes("POLBFL_CLIENT_AUDIT_TICKET_V2"),
);

function parseArguments(argv) {
  const values = {};
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key || !key.startsWith("--") || value === undefined) {
      throw new Error("arguments must be supplied as --name value pairs");
    }
    values[key.slice(2)] = value;
  }
  for (const required of ["rounds-jsonl", "seed", "num-clients", "expected-rounds"]) {
    if (!(required in values)) throw new Error("missing --" + required);
  }
  return {
    roundsPath: path.resolve(values["rounds-jsonl"]),
    seed: Number(values.seed),
    numClients: Number(values["num-clients"]),
    expectedRounds: Number(values["expected-rounds"]),
  };
}

function compile() {
  const files = ["PoLBFLProtocol.sol", "MockAuthenticatedRandomness.sol"];
  const sources = Object.fromEntries(
    files.map((file) => [
      file,
      {
        content: fs.readFileSync(
          path.join(root, "chainEnv", "contracts", file),
          "utf8",
        ),
      },
    ]),
  );
  const output = JSON.parse(
    solc.compile(
      JSON.stringify({
        language: "Solidity",
        sources,
        settings: {
          optimizer: { enabled: true, runs: 200 },
          outputSelection: {
            "*": {
              "*": [
                "abi",
                "evm.bytecode.object",
                "evm.deployedBytecode.object",
              ],
            },
          },
        },
      }),
    ),
  );
  const errors = (output.errors || []).filter((item) => item.severity === "error");
  if (errors.length) {
    throw new Error(errors.map((item) => item.formattedMessage).join("\n"));
  }
  return {
    protocol: output.contracts["PoLBFLProtocol.sol"].PoLBFLProtocol,
    randomness:
      output.contracts["MockAuthenticatedRandomness.sol"]
        .MockAuthenticatedRandomness,
  };
}

async function deploy(signer, artifact, argumentsList = []) {
  const factory = new ethers.ContractFactory(
    artifact.abi,
    artifact.evm.bytecode.object,
    signer,
  );
  const contract = await factory.deploy(...argumentsList);
  await contract.waitForDeployment();
  return contract;
}

function normalizeDigest(value, label) {
  const text = String(value || "").replace(/^0x/, "").toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(text) || /^0+$/.test(text)) {
    throw new Error(label + " must be a nonzero SHA-256-sized digest");
  }
  return "0x" + text;
}

function protocolRoundId(roundNumber) {
  return ethers.sha256(ethers.toUtf8Bytes("round-" + roundNumber));
}

function auditSeed(seed, roundNumber) {
  return (
    "0x" +
    crypto
      .createHash("sha256")
      .update("audit:" + seed + ":" + roundNumber)
      .digest("hex")
  );
}

function clientIndex(clientId, numClients) {
  const match = /^client-([0-9]+)$/.exec(String(clientId));
  if (!match) throw new Error("invalid client identifier: " + clientId);
  const index = Number(match[1]);
  if (!Number.isInteger(index) || index < 0 || index >= numClients) {
    throw new Error("client identifier is outside the registered population");
  }
  return index;
}

function sorted(values) {
  return Array.from(values).map(String).sort();
}

function approximatelyWad(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    throw new Error("invalid expected account value");
  }
  return BigInt(Math.round(number * 1e9)) * 1_000_000_000n;
}

function absBigInt(value) {
  return value < 0n ? -value : value;
}

function receiptEvidence(kind, clientId, receipt) {
  assert.equal(receipt.status, 1);
  return {
    kind,
    client_id: clientId,
    transaction_hash: receipt.hash,
    block_number: Number(receipt.blockNumber),
    gas_used: receipt.gasUsed.toString(),
  };
}

function digestObject(value) {
  return crypto
    .createHash("sha256")
    .update(JSON.stringify(value))
    .digest("hex");
}

async function advancePast(provider, deadline) {
  const latest = await provider.getBlock("latest");
  const seconds = deadline - BigInt(latest.timestamp) + 1n;
  if (seconds > 0n) {
    await provider.send("evm_increaseTime", [Number(seconds)]);
  }
  await provider.send("evm_mine", []);
  const advanced = await provider.getBlock("latest");
  assert.ok(
    BigInt(advanced.timestamp) > deadline,
    "chain time did not advance past the protocol deadline",
  );
}
async function main() {
  const args = parseArguments(process.argv);
  if (
    !Number.isSafeInteger(args.seed) ||
    !Number.isInteger(args.numClients) ||
    args.numClients <= 0 ||
    !Number.isInteger(args.expectedRounds) ||
    args.expectedRounds <= 0
  ) {
    throw new Error("seed, client count, and round count are invalid");
  }
  const rows = fs
    .readFileSync(args.roundsPath, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line))
    .sort((left, right) => Number(left.round) - Number(right.round));
  assert.equal(rows.length, args.expectedRounds, "round evidence is incomplete");
  assert.deepEqual(
    rows.map((row) => Number(row.round)),
    Array.from({ length: args.expectedRounds }, (_, index) => index),
    "round evidence is not contiguous",
  );

  const artifacts = compile();
  const runtimeBytes =
    artifacts.protocol.evm.deployedBytecode.object.length / 2;
  assert.ok(runtimeBytes < 24_576, "optimized runtime exceeds EIP-170");
  const accountCount = args.numClients + 6;
  const ganacheProvider = ganache.provider({
    logging: { quiet: true },
    wallet: { totalAccounts: accountCount, defaultBalance: 1_000 },
    chain: { hardfork: "shanghai" },
  });
  const provider = new ethers.BrowserProvider(
    ganacheProvider,
    undefined,
    { pollingInterval: PROVIDER_POLLING_INTERVAL_MS, cacheTimeout: -1 },
  );
  const signers = await Promise.all(
    Array.from({ length: accountCount }, (_, index) => provider.getSigner(index)),
  );
  const addresses = await Promise.all(
    signers.map((signer) => signer.getAddress()),
  );
  const governance = signers[0];
  const clientSigners = signers.slice(1, 1 + args.numClients);
  const clientAddresses = addresses.slice(1, 1 + args.numClients);
  const verifierSigners = signers.slice(1 + args.numClients);
  assert.equal(verifierSigners.length, 5);
  const initialAccounts = ganacheProvider.getInitialAccounts();
  const messageWalletByAddress = new Map(
    Object.entries(initialAccounts).map(([address, details]) => [
      address.toLowerCase(),
      new ethers.Wallet(details.secretKey),
    ]),
  );

  const oracle = await deploy(governance, artifacts.randomness);
  const protocol = await deploy(governance, artifacts.protocol, [
    await oracle.getAddress(),
    addresses[0],
  ]);
  await (
    await protocol.configureEconomics(ethers.parseEther("0.001"), 0, 0, 0)
  ).wait();
  await (
    await protocol.fundRewards({ value: ethers.parseEther("100") })
  ).wait();
  for (const signer of clientSigners) {
    await (
      await protocol.connect(signer).registerClient({ value: CLIENT_STAKE })
    ).wait();
  }
  for (const signer of verifierSigners) {
    await (
      await protocol.connect(signer).registerVerifier({ value: CLIENT_STAKE })
    ).wait();
  }

  const roundEvidence = [];
  let transactionCount = 0;
  const startedTransitions = process.hrtime.bigint();
  let totalGas = 0n;
  for (const row of rows) {
    const startedRound = process.hrtime.bigint();
    const roundNumber = Number(row.round);
    const roundId = protocolRoundId(roundNumber);
    const commitments = row.trace_commitments;
    const participants = sorted(row.participating_clients || []);
    if (
      !commitments ||
      participants.length !== Number(row.active_clients) ||
      sorted(Object.keys(commitments)).join(",") !== participants.join(",")
    ) {
      throw new Error("round commitments do not cover every participant");
    }
    const proofOutcomes = row.proof_outcomes || {};
    if (sorted(Object.keys(proofOutcomes)).join(",") !== participants.join(",")) {
      throw new Error("round proof outcomes do not cover every participant");
    }
    const stepCounts = new Set(
      participants.map((clientId) => {
        const commitment = commitments[clientId];
        return Number(commitment.final_step) - Number(commitment.first_step);
      }),
    );
    if (stepCounts.size !== 1 || Array.from(stepCounts)[0] <= 0) {
      throw new Error("round commitments disagree on positive trace bounds");
    }
    const expectedSteps = Array.from(stepCounts)[0];
    const latest = await provider.getBlock("latest");
    const commitDeadline = BigInt(latest.timestamp) + COMMIT_WINDOW_SECONDS;
    const auditDeadline = commitDeadline + AUDIT_WINDOW_SECONDS;
    const transactions = [];
    let receipt = await (
      await protocol.createRound(
        roundId,
        commitDeadline,
        auditDeadline,
        expectedSteps,
      )
    ).wait();
    transactions.push(receiptEvidence("create_round", null, receipt));

    for (const clientId of participants) {
      const index = clientIndex(clientId, args.numClients);
      const commitment = commitments[clientId];
      receipt = await (
        await protocol.connect(clientSigners[index]).submitCommitment(
          roundId,
          normalizeDigest(commitment.merkle_root, "commitment root"),
          normalizeDigest(
            commitment.final_model_digest,
            "final model digest",
          ),
          expectedSteps,
        )
      ).wait();
      transactions.push(
        receiptEvidence("submit_commitment", clientId, receipt),
      );
    }

    const randomness = auditSeed(args.seed, roundNumber);
    receipt = await (await oracle.setOutput(roundId, randomness)).wait();
    transactions.push(receiptEvidence("publish_randomness", null, receipt));
    await advancePast(provider, commitDeadline);
    receipt = await (await protocol.activateAudit(roundId, randomness)).wait();
    transactions.push(receiptEvidence("activate_audit", null, receipt));

    const auditedOnChain = [];
    for (const clientId of participants) {
      const index = clientIndex(clientId, args.numClients);
      const expectedTicket = ethers.sha256(
        ethers.concat([
          AUDIT_TICKET_DOMAIN,
          randomness,
          roundId,
          normalizeDigest(commitments[clientId].merkle_root, "commitment root"),
        ]),
      );
      assert.equal(
        await protocol.auditTicket(roundId, clientAddresses[index]),
        expectedTicket,
      );
      if (await protocol.isAudited(roundId, clientAddresses[index])) {
        auditedOnChain.push(clientId);
      }
    }
    assert.deepEqual(
      sorted(auditedOnChain),
      sorted(row.audited_clients || []),
      "Python and Solidity selected different audit clients",
    );

    const committee = Array.from(
      await protocol.getRoundCommittee(roundId),
    ).map(String);
    assert.equal(new Set(committee.map((value) => value.toLowerCase())).size, 5);
    const timeoutClients = [];
    for (const clientId of auditedOnChain) {
      const outcome = String(proofOutcomes[clientId]);
      const index = clientIndex(clientId, args.numClients);
      if (outcome === "timeout") {
        timeoutClients.push(clientId);
        continue;
      }
      if (outcome !== "accept" && outcome !== "reject") {
        throw new Error("audited client has an invalid proof outcome");
      }
      const auditEvidence = (row.audit_evidence || {})[clientId];
      const proofDigest = normalizeDigest(
        auditEvidence && auditEvidence.proof_set_digest,
        "proof-set digest",
      );
      const valid = outcome === "accept";
      const verifiers = committee.slice(0, 3);
      const signatures = [];
      for (const verifier of verifiers) {
        const message = await protocol.receiptMessage(
          roundId,
          clientAddresses[index],
          proofDigest,
          valid,
          verifier,
        );
        signatures.push(
          await messageWalletByAddress
            .get(verifier.toLowerCase())
            .signMessage(ethers.getBytes(message)),
        );
      }
      receipt = await (
        await protocol.submitQuorumBySignatures(
          roundId,
          clientAddresses[index],
          proofDigest,
          valid,
          verifiers,
          signatures,
        )
      ).wait();
      transactions.push(receiptEvidence("submit_quorum", clientId, receipt));
      if (!valid) {
        receipt = await (
          await protocol.executeRejectedAudit(
            roundId,
            clientAddresses[index],
          )
        ).wait();
        transactions.push(
          receiptEvidence("execute_rejection", clientId, receipt),
        );
      }
    }

    const sybil = new Set((row.sybil_flagged_clients || []).map(String));
    const statistical = new Set(
      (row.statistically_rejected_clients || []).map(String),
    );
    for (const clientId of participants) {
      const outcome = String(proofOutcomes[clientId]);
      if (outcome === "reject" || outcome === "timeout") continue;
      const index = clientIndex(clientId, args.numClients);
      receipt = await (
        await protocol.settleClient(
          roundId,
          clientAddresses[index],
          WAD,
          sybil.has(clientId),
          !statistical.has(clientId),
        )
      ).wait();
      transactions.push(receiptEvidence("settle_client", clientId, receipt));
    }

    await advancePast(provider, auditDeadline);
    for (const clientId of timeoutClients) {
      const index = clientIndex(clientId, args.numClients);
      receipt = await (
        await protocol.finalizeAuditTimeout(
          roundId,
          clientAddresses[index],
        )
      ).wait();
      transactions.push(
        receiptEvidence("finalize_timeout", clientId, receipt),
      );
    }
    receipt = await (await protocol.finalizeRound(roundId)).wait();
    transactions.push(receiptEvidence("finalize_round", null, receipt));
    const contractRound = await protocol.rounds(roundId);
    assert.equal(contractRound.finalized, true);
    assert.equal(
      Number(contractRound.settledClients),
      participants.length,
      "on-chain round has unsettled clients",
    );

    const accountState = {};
    for (let index = 0; index < args.numClients; index += 1) {
      const clientId = "client-" + index;
      const account = await protocol.accounts(clientAddresses[index]);
      const expectedStake = approximatelyWad(row.stake_by_client[clientId]);
      const expectedReputation = approximatelyWad(
        row.reputation_by_client[clientId],
      );
      assert.ok(
        absBigInt(account.stake - expectedStake) <= 2_000_000_000n,
        "stake state differs for " + clientId,
      );
      assert.ok(
        absBigInt(account.reputation - expectedReputation) <= 2_000_000_000n,
        "reputation state differs for " + clientId,
      );
      accountState[clientId] = {
        stake: account.stake.toString(),
        reputation: account.reputation.toString(),
        active: Boolean(account.active),
      };
    }
    for (const item of transactions) {
      transactionCount += 1;
      totalGas += BigInt(item.gas_used);
    }
    roundEvidence.push({
      round: roundNumber,
      round_id: roundId,
      randomness,
      audited_clients: sorted(auditedOnChain),
      committee,
      transaction_count: transactions.length,
      transactions,
      python_settlement_digest: String(row.settlement_digest),
      runtime_seconds:
        Number(process.hrtime.bigint() - startedRound) / 1_000_000_000,
      account_state_digest: digestObject(accountState),
    });
  }

  const network = await provider.getNetwork();
  const output = {
    schema_version: 1,
    passed: true,
    real_contract_transitions: true,
    contract_rounds: rows.length,
    num_clients: args.numClients,
    chain_id: network.chainId.toString(),
    protocol_address: await protocol.getAddress(),
    randomness_oracle_address: await oracle.getAddress(),
    runtime_bytes: runtimeBytes,
    transition_runtime_seconds:
      Number(process.hrtime.bigint() - startedTransitions) / 1_000_000_000,
    protocol_creation_bytecode_sha256: crypto
      .createHash("sha256")
      .update(Buffer.from(artifacts.protocol.evm.bytecode.object, "hex"))
      .digest("hex"),
    protocol_runtime_bytecode_sha256: crypto
      .createHash("sha256")
      .update(
        Buffer.from(artifacts.protocol.evm.deployedBytecode.object, "hex"),
      )
      .digest("hex"),
    transaction_count: transactionCount,
    total_gas: totalGas.toString(),
    rounds: roundEvidence,
    tool_versions: {
      node: process.version,
      solc: solc.version(),
      ethers: ethers.version,
      ganache: ganache.__experimental_info().version,
    },
  };
  output.transition_digest = digestObject(output);
  process.stdout.write(JSON.stringify(output) + "\n");
  if (typeof ganacheProvider.disconnect === "function") {
    ganacheProvider.disconnect();
  }
}

module.exports = { advancePast };

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(String(error.stack || error) + "\n");
    process.exitCode = 1;
  });
}
