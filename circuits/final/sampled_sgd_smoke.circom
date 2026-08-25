pragma circom 2.1.8;

include "sampled_sgd_transition.circom";

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
]} = SampledSGDTransition(4, 2, 2, 32, 1, 2);
