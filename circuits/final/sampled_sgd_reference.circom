pragma circom 2.1.8;

include "sampled_sgd_transition.circom";

// The reference circuit targets a five-step checkpoint interval, batch size 32,
// and fourteen deterministic coordinate checks drawn from the committed 1%
// gradient sample. Its exact constraint count and proving profile are enforced
// by the circuit benchmark gate.
component main {public [
    contextHash,
    commitmentRootHash,
    challengeHash,
    pairIndex,
    batchCommitmentHash,
    protocolBindingHash,
    activeStepCount,
    startWeightsHash,
    endWeightsHash,
    gradientsHash,
    dataIndicesHash,
    samplePlanHash,
    auxiliaryHash,
    scale,
    learningRate,
    maxDistanceSquared,
    maxRoundingError,
    maxCumulativeRoundingErrorSquared
]} = SampledSGDTransition(14, 5, 32, 48, 8, 4);
