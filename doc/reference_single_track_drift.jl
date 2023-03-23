#=
    This script is loaded into the main scripts to prepare the data
    for training and validation.
=#

rng = Random.default_rng();
Random.seed!(rng, 1);

data = readmat("reference_single_track_drift.mat");
#plot_inputs(data)

scaler_states, scaler_input = scalers(data, (0.1, 69.9));
σ_noise = 0.025;

vali =  sim_dat(data, (70.0, 99.9), scaler_states, σ_noise);
re_scale(mat) = StatsBase.reconstruct(scaler_states, mat);

# more data for training with various initial state and steering input
data2 = readmat("reference_single_track_drift2.mat");
data3 = readmat("reference_single_track_drift3.mat");

train_arr = [sim_dat(data2, (0.1, 69.9), scaler_states, σ_noise), 
             sim_dat(data3, (0.1, 69.9), scaler_states, σ_noise),
             sim_dat(data, (0.1, 69.9), scaler_states, σ_noise)
             ]