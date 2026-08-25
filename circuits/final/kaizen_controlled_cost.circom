pragma circom 2.1.8;

// Table 12 describes a controlled Kaizen-style full-verification baseline,
// not a wire-compatible implementation of the Kaizen protocol. The paper
// reports approximately 4.5M constraints. This circuit fixes a reproducible
// 4.0M sequential training-arithmetic workload below the official power-22
// transcript capacity and measures the same Groth16 witness/prove/verify path.
template ControlledTrainingCost(steps) {
    signal input seed;
    signal output terminal;
    signal state[steps + 1];

    state[0] <== seed;
    for (var index = 0; index < steps; index++) {
        state[index + 1] <== state[index] * state[index] + index + 1;
    }
    terminal <== state[steps];
}

component main {public [seed]} = ControlledTrainingCost(4000000);
