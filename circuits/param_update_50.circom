pragma circom 2.0.0;
include "./parameter_update_template.circom";
component main {public [W_t_hash, W_t1_hash, data_hash, max_distance]} = ParameterUpdateProof(50, 32);

