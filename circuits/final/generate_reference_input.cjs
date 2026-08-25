const fs = require("node:fs");
const circomlib = require("circomlibjs");

let poseidon;

function fold2(values, initial = 0n) {
  let acc = BigInt(initial);
  for (const value of values) acc = poseidon([acc, BigInt(value)]);
  return acc;
}

function fold3(rows, initial = 0n) {
  let acc = BigInt(initial);
  for (const [left, right] of rows) {
    acc = poseidon([acc, BigInt(left), BigInt(right)]);
  }
  return acc;
}

function foldPairChunks(rows, pairsPerChunk, initial = 0n) {
  if (rows.length % pairsPerChunk !== 0) {
    throw new Error("pair rows must divide exactly into commitment chunks");
  }
  let acc = BigInt(initial);
  for (let offset = 0; offset < rows.length; offset += pairsPerChunk) {
    const inputs = [acc];
    for (const [left, right] of rows.slice(offset, offset + pairsPerChunk)) {
      inputs.push(BigInt(left), BigInt(right));
    }
    acc = poseidon(inputs);
  }
  return acc;
}

function unsigned(values) {
  return values.map((value) => {
    const number = BigInt(value);
    if (number < 0n) return { magnitude: (-number).toString(), sign: "1" };
    return { magnitude: number.toString(), sign: "0" };
  });
}

async function main() {
const implementation = await circomlib.buildPoseidon();
poseidon = (inputs) => implementation.F.toObject(implementation(inputs.map(BigInt)));

const sampleCount = 14;
const steps = 5;
const batchTerms = 32;
const scale = 1000n;
const learningRate = 10n;
const contextHash = 123n;
const commitmentRootHash = 456n;
const challengeHash = 789n;
const pairIndex = 3n;
const batchCommitmentHash = 321n;
const protocolBindingHash = contextHash
  + 2n * commitmentRootHash
  + 4n * challengeHash
  + 8n * pairIndex
  + 16n * batchCommitmentHash;

const sampleIndices = Array.from({ length: sampleCount }, (_, index) => index);
const weights = Array.from({ length: steps + 1 }, (_, step) =>
  Array.from({ length: sampleCount }, (_, index) => 10000 + index - step),
);
const gradients = Array.from({ length: steps }, () =>
  Array.from({ length: sampleCount }, () => 100),
);
const dataIndices = Array.from({ length: steps }, (_, step) =>
  Array.from({ length: batchTerms }, (_, index) => step * batchTerms + index),
);
const activations = gradients.map((step) =>
  step.map(() => Array.from({ length: batchTerms }, (_, term) => (term === 0 ? Number(scale) : 0))),
);
const errors = gradients.map((step) =>
  step.map((gradient) => Array.from({ length: batchTerms }, (_, term) => (term === 0 ? gradient : 0))),
);

const weightSigned = weights.map(unsigned);
const gradientSigned = gradients.map(unsigned);
const roundingSigned = gradients.map((step) => unsigned(step.map(() => 0)));
const activationSigned = activations.map((step) => step.map(unsigned));
const errorSigned = errors.map((step) => step.map(unsigned));

const startWeightsHash = fold3(sampleIndices.map((index, i) => [index, weights[0][i]]), contextHash);
const endWeightsHash = fold3(sampleIndices.map((index, i) => [index, weights[steps][i]]), contextHash);
const gradientsHash = fold2(gradients.flat(), contextHash);
const dataIndicesHash = fold2(dataIndices.flat(), contextHash);
const samplePlanHash = fold2(sampleIndices);
const auxiliaryRows = [];
for (let step = 0; step < gradients.length; step += 1) {
  for (let i = 0; i < sampleIndices.length; i += 1) {
    for (let b = 0; b < dataIndices[step].length; b += 1) {
      auxiliaryRows.push([activations[step][i][b], errors[step][i][b]]);
    }
  }
}
const auxiliaryHash = foldPairChunks(auxiliaryRows, 4, contextHash);

const input = {
  contextHash: contextHash.toString(),
  commitmentRootHash: commitmentRootHash.toString(),
  challengeHash: challengeHash.toString(),
  pairIndex: pairIndex.toString(),
  batchCommitmentHash: batchCommitmentHash.toString(),
  protocolBindingHash: protocolBindingHash.toString(),
  activeStepCount: String(steps),
  startWeightsHash: startWeightsHash.toString(),
  endWeightsHash: endWeightsHash.toString(),
  gradientsHash: gradientsHash.toString(),
  dataIndicesHash: dataIndicesHash.toString(),
  samplePlanHash: samplePlanHash.toString(),
  auxiliaryHash: auxiliaryHash.toString(),
  scale: scale.toString(),
  learningRate: learningRate.toString(),
  maxDistanceSquared: String(sampleCount * steps * steps),
  maxRoundingError: "0",
  maxCumulativeRoundingErrorSquared: "0",
  sampleIndices: sampleIndices.map(String),
  dataIndices: dataIndices.map((row) => row.map(String)),
  stepActive: Array.from({ length: steps }, () => "1"),
  weightMagnitude: weightSigned.map((row) => row.map((item) => item.magnitude)),
  weightSign: weightSigned.map((row) => row.map((item) => item.sign)),
  gradientMagnitude: gradientSigned.map((row) => row.map((item) => item.magnitude)),
  gradientSign: gradientSigned.map((row) => row.map((item) => item.sign)),
  roundingMagnitude: roundingSigned.map((row) => row.map((item) => item.magnitude)),
  roundingSign: roundingSigned.map((row) => row.map((item) => item.sign)),
  activationMagnitude: activationSigned.map((step) => step.map((row) => row.map((item) => item.magnitude))),
  activationSign: activationSigned.map((step) => step.map((row) => row.map((item) => item.sign))),
  errorMagnitude: errorSigned.map((step) => step.map((row) => row.map((item) => item.magnitude))),
  errorSign: errorSigned.map((step) => step.map((row) => row.map((item) => item.sign))),
};

const output = process.argv[2];
if (!output) throw new Error("usage: node generate_reference_input.cjs OUTPUT.json");
fs.writeFileSync(output, `${JSON.stringify(input)}\n`, "utf8");
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
