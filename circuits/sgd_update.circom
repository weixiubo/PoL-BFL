pragma circom 2.0.0;

// Real SGD update proof wrapper using the shared ParameterUpdateProof.
// This replaces the previous placeholder circuit and provides production constraints.

include "parameter_update_template.circom";

// Public inputs: W_t_hash, W_t1_hash, data_hash, max_distance
// Private inputs and constraints are defined in the template.
component main {public [W_t_hash, W_t1_hash, data_hash, max_distance]} = ParameterUpdateProof(100, 32);
