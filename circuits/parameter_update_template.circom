pragma circom 2.0.0;

// Template-only version of the parameter update circuit (no main component here)
// so that we can instantiate different param_size values via wrapper files.

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";

// Parameter update proof: hashes (Poseidon fold) and L2 distance constraint
// Public: W_t_hash, W_t1_hash, data_hash, max_distance
// Private: W_t[param_size], W_t1[param_size], data_indices[batch_size]
template ParameterUpdateProof(param_size, batch_size) {
    // Public inputs
    signal input W_t_hash;
    signal input W_t1_hash;
    signal input data_hash;
    signal input max_distance;

    // Private inputs
    signal input W_t[param_size];
    signal input W_t1[param_size];
    signal input data_indices[batch_size];

    // Hash W_t with Poseidon(2) folding
    signal wt_acc[param_size + 1];
    wt_acc[0] <== 0;
    component pose_wt[param_size];
    for (var i = 0; i < param_size; i++) {
        pose_wt[i] = Poseidon(2);
        pose_wt[i].inputs[0] <== wt_acc[i];
        pose_wt[i].inputs[1] <== W_t[i];
        wt_acc[i + 1] <== pose_wt[i].out;
    }
    wt_acc[param_size] === W_t_hash;

    // Hash W_t1 with Poseidon(2) folding
    signal wt1_acc[param_size + 1];
    wt1_acc[0] <== 0;
    component pose_wt1[param_size];
    for (var i = 0; i < param_size; i++) {
        pose_wt1[i] = Poseidon(2);
        pose_wt1[i].inputs[0] <== wt1_acc[i];
        pose_wt1[i].inputs[1] <== W_t1[i];
        wt1_acc[i + 1] <== pose_wt1[i].out;
    }
    wt1_acc[param_size] === W_t1_hash;

    // Hash data indices with Poseidon(2) folding
    signal data_acc[batch_size + 1];
    data_acc[0] <== 0;
    component pose_data[batch_size];
    for (var i = 0; i < batch_size; i++) {
        pose_data[i] = Poseidon(2);
        pose_data[i].inputs[0] <== data_acc[i];
        pose_data[i].inputs[1] <== data_indices[i];
        data_acc[i + 1] <== pose_data[i].out;
    }
    data_acc[batch_size] === data_hash;

    // L2 distance squared between W_t1 and W_t
    signal distance_squared;
    signal partial_sum[param_size];
    signal diffs[param_size];

    for (var j = 0; j < param_size; j++) {
        diffs[j] <== W_t1[j] - W_t[j];
        if (j == 0) {
            partial_sum[j] <== diffs[j] * diffs[j];
        } else {
            partial_sum[j] <== partial_sum[j-1] + diffs[j] * diffs[j];
        }
    }

    distance_squared <== partial_sum[param_size - 1];

    // Ensure distance_squared <= max_distance
    component less_than = LessThan(252);
    less_than.in[0] <== distance_squared;
    less_than.in[1] <== max_distance;
    less_than.out === 1;
}

