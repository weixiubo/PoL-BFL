pragma circom 2.1.8;

include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/poseidon.circom";

// Signed field value represented as a constrained sign and magnitude.
template SignedValue(bits) {
    signal input magnitude;
    signal input sign;
    signal output value;

    sign * (sign - 1) === 0;
    component range = Num2Bits(bits);
    range.in <== magnitude;
    value <== magnitude * (1 - 2 * sign);
}

// The private witness proves a sampled SGD relation for one checkpoint interval.
// SHA-256 checkpoint/Merkle membership is verified by the outer protocol using
// the same authenticated checkpoint openings. The public Poseidon commitments
// below are included in those checkpoint auxiliary records.
template SampledSGDTransition(sampleCount, steps, batchTerms, valueBits, auxiliaryChunks, auxiliaryPairsPerChunk) {
    assert(auxiliaryChunks * auxiliaryPairsPerChunk == batchTerms);
    // Public protocol binding and commitments.
    signal input contextHash;
    signal input commitmentRootHash;
    signal input challengeHash;
    signal input pairIndex;
    signal input batchCommitmentHash;
    signal input protocolBindingHash;
    signal input activeStepCount;
    signal input startWeightsHash;
    signal input endWeightsHash;
    signal input gradientsHash;
    signal input dataIndicesHash;
    signal input samplePlanHash;
    signal input auxiliaryHash;
    signal input scale;
    signal input learningRate;
    signal input maxDistanceSquared;
    signal input maxRoundingError;
    signal input maxCumulativeRoundingErrorSquared;

    // Private sampled coordinates and SGD witnesses.
    signal input sampleIndices[sampleCount];
    signal input dataIndices[steps][batchTerms];
    signal input stepActive[steps];

    signal input weightMagnitude[steps + 1][sampleCount];
    signal input weightSign[steps + 1][sampleCount];
    signal input gradientMagnitude[steps][sampleCount];
    signal input gradientSign[steps][sampleCount];
    signal input roundingMagnitude[steps][sampleCount];
    signal input roundingSign[steps][sampleCount];

    // Each sampled gradient is a sum of committed activation/error products.
    signal input activationMagnitude[steps][sampleCount][batchTerms];
    signal input activationSign[steps][sampleCount][batchTerms];
    signal input errorMagnitude[steps][sampleCount][batchTerms];
    signal input errorSign[steps][sampleCount][batchTerms];

    component scaleRange = Num2Bits(32);
    component learningRateRange = Num2Bits(32);
    component pairIndexRange = Num2Bits(32);
    scaleRange.in <== scale;
    learningRateRange.in <== learningRate;
    pairIndexRange.in <== pairIndex;

    signal activeStepAcc[steps + 1];
    activeStepAcc[0] <== 0;
    for (var s = 0; s < steps; s++) {
        stepActive[s] * (stepActive[s] - 1) === 0;
        if (s > 0) {
            stepActive[s] * (1 - stepActive[s - 1]) === 0;
        }
        activeStepAcc[s + 1] <== activeStepAcc[s] + stepActive[s];
    }
    activeStepAcc[steps] === activeStepCount;

    // Domain-separated linear field binding. SHA-256 digests are mapped into
    // the BN254 scalar field by the protocol adapter; all terms are public and
    // recomputed independently by every verifier.
    protocolBindingHash === contextHash
        + 2 * commitmentRootHash
        + 4 * challengeHash
        + 8 * pairIndex
        + 16 * batchCommitmentHash;

    component sampleIndexRange[sampleCount];
    for (var i = 0; i < sampleCount; i++) {
        sampleIndexRange[i] = Num2Bits(32);
        sampleIndexRange[i].in <== sampleIndices[i];
    }
    component sampleIndexOrder[sampleCount - 1];
    for (var i = 0; i < sampleCount - 1; i++) {
        sampleIndexOrder[i] = LessThan(32);
        sampleIndexOrder[i].in[0] <== sampleIndices[i];
        sampleIndexOrder[i].in[1] <== sampleIndices[i + 1];
        sampleIndexOrder[i].out === 1;
    }

    component dataIndexRange[steps][batchTerms];
    for (var s = 0; s < steps; s++) {
        for (var b = 0; b < batchTerms; b++) {
            dataIndexRange[s][b] = Num2Bits(32);
            dataIndexRange[s][b].in <== dataIndices[s][b];
            (1 - stepActive[s]) * (dataIndices[s][b] - 4294967295) === 0;
        }
    }

    component weights[steps + 1][sampleCount];
    component gradients[steps][sampleCount];
    component rounding[steps][sampleCount];
    component activations[steps][sampleCount][batchTerms];
    component errors[steps][sampleCount][batchTerms];

    signal gradientTermSum[steps][sampleCount][batchTerms + 1];
    signal scaledWeightDelta[steps][sampleCount];
    signal scaledGradientStep[steps][sampleCount];
    signal totalDistance[sampleCount + 1];
    signal finalDifference[sampleCount];
    signal cumulativeRounding[sampleCount][steps + 1];
    signal totalCumulativeRoundingSquared[sampleCount + 1];
    totalDistance[0] <== 0;
    totalCumulativeRoundingSquared[0] <== 0;

    component roundingBound[steps][sampleCount];

    for (var i = 0; i < sampleCount; i++) {
        cumulativeRounding[i][0] <== 0;
        for (var s = 0; s < steps + 1; s++) {
            weights[s][i] = SignedValue(valueBits);
            weights[s][i].magnitude <== weightMagnitude[s][i];
            weights[s][i].sign <== weightSign[s][i];
        }

        for (var s = 0; s < steps; s++) {
            gradients[s][i] = SignedValue(valueBits);
            gradients[s][i].magnitude <== gradientMagnitude[s][i];
            gradients[s][i].sign <== gradientSign[s][i];
            (1 - stepActive[s]) * gradients[s][i].value === 0;

            rounding[s][i] = SignedValue(valueBits);
            rounding[s][i].magnitude <== roundingMagnitude[s][i];
            rounding[s][i].sign <== roundingSign[s][i];
            (1 - stepActive[s]) * rounding[s][i].value === 0;
            roundingBound[s][i] = LessThan(valueBits + 1);
            roundingBound[s][i].in[0] <== roundingMagnitude[s][i];
            roundingBound[s][i].in[1] <== maxRoundingError + 1;
            roundingBound[s][i].out === 1;

            gradientTermSum[s][i][0] <== 0;
            for (var b = 0; b < batchTerms; b++) {
                activations[s][i][b] = SignedValue(valueBits);
                activations[s][i][b].magnitude <== activationMagnitude[s][i][b];
                activations[s][i][b].sign <== activationSign[s][i][b];
                errors[s][i][b] = SignedValue(valueBits);
                errors[s][i][b].magnitude <== errorMagnitude[s][i][b];
                errors[s][i][b].sign <== errorSign[s][i][b];
                (1 - stepActive[s]) * activations[s][i][b].value === 0;
                (1 - stepActive[s]) * errors[s][i][b].value === 0;
                gradientTermSum[s][i][b + 1] <==
                    gradientTermSum[s][i][b]
                    + activations[s][i][b].value * errors[s][i][b].value;
            }

            // Fixed-point sampled gradient and SGD update with bounded rounding.
            gradients[s][i].value * scale === gradientTermSum[s][i][batchTerms];
            scaledWeightDelta[s][i] <==
                scale * (weights[s + 1][i].value - weights[s][i].value);
            scaledGradientStep[s][i] <== learningRate * gradients[s][i].value;
            scaledWeightDelta[s][i] + scaledGradientStep[s][i]
                === rounding[s][i].value;
            cumulativeRounding[i][s + 1] <==
                cumulativeRounding[i][s] + rounding[s][i].value;
        }

        finalDifference[i] <== weights[steps][i].value - weights[0][i].value;
        totalDistance[i + 1] <== totalDistance[i] + finalDifference[i] * finalDifference[i];
        totalCumulativeRoundingSquared[i + 1] <==
            totalCumulativeRoundingSquared[i]
            + cumulativeRounding[i][steps] * cumulativeRounding[i][steps];
    }

    component distanceBound = LessThan(252);
    distanceBound.in[0] <== totalDistance[sampleCount];
    distanceBound.in[1] <== maxDistanceSquared + 1;
    distanceBound.out === 1;

    component cumulativeRoundingBound = LessThan(252);
    cumulativeRoundingBound.in[0] <== totalCumulativeRoundingSquared[sampleCount];
    cumulativeRoundingBound.in[1] <== maxCumulativeRoundingErrorSquared + 1;
    cumulativeRoundingBound.out === 1;

    // Public sample-plan commitment.
    signal samplePlanAcc[sampleCount + 1];
    samplePlanAcc[0] <== 0;
    component samplePlanPoseidon[sampleCount];
    for (var i = 0; i < sampleCount; i++) {
        samplePlanPoseidon[i] = Poseidon(2);
        samplePlanPoseidon[i].inputs[0] <== samplePlanAcc[i];
        samplePlanPoseidon[i].inputs[1] <== sampleIndices[i];
        samplePlanAcc[i + 1] <== samplePlanPoseidon[i].out;
    }
    samplePlanAcc[sampleCount] === samplePlanHash;

    // Endpoint weight commitments include context and sampled coordinate IDs.
    signal startAcc[sampleCount + 1];
    signal endAcc[sampleCount + 1];
    startAcc[0] <== contextHash;
    endAcc[0] <== contextHash;
    component startPoseidon[sampleCount];
    component endPoseidon[sampleCount];
    for (var i = 0; i < sampleCount; i++) {
        startPoseidon[i] = Poseidon(3);
        startPoseidon[i].inputs[0] <== startAcc[i];
        startPoseidon[i].inputs[1] <== sampleIndices[i];
        startPoseidon[i].inputs[2] <== weights[0][i].value;
        startAcc[i + 1] <== startPoseidon[i].out;

        endPoseidon[i] = Poseidon(3);
        endPoseidon[i].inputs[0] <== endAcc[i];
        endPoseidon[i].inputs[1] <== sampleIndices[i];
        endPoseidon[i].inputs[2] <== weights[steps][i].value;
        endAcc[i + 1] <== endPoseidon[i].out;
    }
    startAcc[sampleCount] === startWeightsHash;
    endAcc[sampleCount] === endWeightsHash;

    signal gradientAcc[steps * sampleCount + 1];
    gradientAcc[0] <== contextHash;
    component gradientPoseidon[steps * sampleCount];
    for (var s = 0; s < steps; s++) {
        for (var i = 0; i < sampleCount; i++) {
            var position = s * sampleCount + i;
            gradientPoseidon[position] = Poseidon(2);
            gradientPoseidon[position].inputs[0] <== gradientAcc[position];
            gradientPoseidon[position].inputs[1] <== gradients[s][i].value;
            gradientAcc[position + 1] <== gradientPoseidon[position].out;
        }
    }
    gradientAcc[steps * sampleCount] === gradientsHash;

    signal dataAcc[steps * batchTerms + 1];
    dataAcc[0] <== contextHash;
    component dataPoseidon[steps * batchTerms];
    for (var s = 0; s < steps; s++) {
        for (var b = 0; b < batchTerms; b++) {
            var position = s * batchTerms + b;
            dataPoseidon[position] = Poseidon(2);
            dataPoseidon[position].inputs[0] <== dataAcc[position];
            dataPoseidon[position].inputs[1] <== dataIndices[s][b];
            dataAcc[position + 1] <== dataPoseidon[position].out;
        }
    }
    dataAcc[steps * batchTerms] === dataIndicesHash;

    // Pack several activation/error pairs into each Poseidon permutation. This
    // preserves collision-resistant witness binding while avoiding one costly
    // permutation per scalar pair.
    signal auxiliaryAcc[steps * sampleCount * auxiliaryChunks + 1];
    auxiliaryAcc[0] <== contextHash;
    component auxiliaryPoseidon[steps * sampleCount * auxiliaryChunks];
    for (var s = 0; s < steps; s++) {
        for (var i = 0; i < sampleCount; i++) {
            for (var chunk = 0; chunk < auxiliaryChunks; chunk++) {
                var position = (s * sampleCount + i) * auxiliaryChunks + chunk;
                auxiliaryPoseidon[position] = Poseidon(1 + 2 * auxiliaryPairsPerChunk);
                auxiliaryPoseidon[position].inputs[0] <== auxiliaryAcc[position];
                for (var pair = 0; pair < auxiliaryPairsPerChunk; pair++) {
                    var b = chunk * auxiliaryPairsPerChunk + pair;
                    auxiliaryPoseidon[position].inputs[1 + 2 * pair] <== activations[s][i][b].value;
                    auxiliaryPoseidon[position].inputs[2 + 2 * pair] <== errors[s][i][b].value;
                }
                auxiliaryAcc[position + 1] <== auxiliaryPoseidon[position].out;
            }
        }
    }
    auxiliaryAcc[steps * sampleCount * auxiliaryChunks] === auxiliaryHash;
}
