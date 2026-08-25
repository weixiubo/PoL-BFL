const assert = require("node:assert/strict");
const { advancePast } = require("../../scripts/contract_round_replay.cjs");

async function main() {
  let timestamp = 100;
  const calls = [];
  const provider = {
    async getBlock() {
      return { timestamp };
    },
    async send(method, argumentsList) {
      calls.push([method, argumentsList]);
      if (method === "evm_increaseTime") {
        timestamp += Number(argumentsList[0]);
      } else if (method === "evm_mine") {
        timestamp += 1;
      } else {
        throw new Error("unexpected provider method: " + method);
      }
    },
  };

  await advancePast(provider, 200n);
  assert.equal(timestamp, 202);
  assert.deepEqual(calls, [
    ["evm_increaseTime", [101]],
    ["evm_mine", []],
  ]);

  calls.length = 0;
  timestamp = 250;
  await advancePast(provider, 200n);
  assert.equal(timestamp, 251);
  assert.deepEqual(calls, [["evm_mine", []]]);

  process.stdout.write(
    JSON.stringify({ advanced_timestamp: 202, already_past_timestamp: 251 }) + "\n",
  );
}

main().catch((error) => {
  process.stderr.write(String(error.stack || error) + "\n");
  process.exitCode = 1;
});
