const fs = require("node:fs");
const circomlib = require("circomlibjs");

let poseidon;

function fold2(values, initial) {
  let acc = BigInt(initial);
  for (const value of values) acc = poseidon([acc, BigInt(value)]);
  return acc;
}

function fold3(rows, initial) {
  let acc = BigInt(initial);
  for (const row of rows) {
    if (!Array.isArray(row) || row.length !== 2) throw new Error("fold3 rows require two values");
    acc = poseidon([acc, BigInt(row[0]), BigInt(row[1])]);
  }
  return acc;
}

function foldPairChunks(rows, pairsPerChunk, initial) {
  if (!Number.isInteger(pairsPerChunk) || pairsPerChunk <= 0) {
    throw new Error("pairsPerChunk must be positive");
  }
  if (rows.length % pairsPerChunk !== 0) {
    throw new Error("pair rows must divide exactly into commitment chunks");
  }
  let acc = BigInt(initial);
  for (let offset = 0; offset < rows.length; offset += pairsPerChunk) {
    const inputs = [acc];
    for (const row of rows.slice(offset, offset + pairsPerChunk)) {
      if (!Array.isArray(row) || row.length !== 2) throw new Error("pair row requires two values");
      inputs.push(BigInt(row[0]), BigInt(row[1]));
    }
    acc = poseidon(inputs);
  }
  return acc;
}

function handle(request) {
  if (!Array.isArray(request.operations)) throw new Error("operations array is required");
  const results = request.operations.map((operation) => {
    const initial = operation.initial === undefined ? 0n : BigInt(operation.initial);
    if (operation.kind === "fold2") return fold2(operation.values, initial).toString();
    if (operation.kind === "fold3") return fold3(operation.rows, initial).toString();
    if (operation.kind === "fold_pair_chunks") {
      return foldPairChunks(operation.rows, Number(operation.pairs_per_chunk), initial).toString();
    }
    throw new Error(`unsupported Poseidon operation: ${operation.kind}`);
  });
  return { results };
}

async function main() {
  const implementation = await circomlib.buildPoseidon();
  poseidon = (inputs) => implementation.F.toObject(implementation(inputs.map(BigInt)));
  if (process.argv.includes("--stream")) {
    const readline = require("node:readline");
    const lines = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
    lines.on("line", (line) => {
      try {
        process.stdout.write(`${JSON.stringify(handle(JSON.parse(line)))}\n`);
      } catch (error) {
        process.stdout.write(`${JSON.stringify({ error: String(error.message || error) })}\n`);
      }
    });
  } else {
    const request = JSON.parse(fs.readFileSync(0, "utf8"));
    process.stdout.write(`${JSON.stringify(handle(request))}\n`);
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
