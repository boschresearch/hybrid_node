using DifferentialEquations, StatsBase, Random, Noise
using HybridNode

include("reference_single_track_drift.jl");
path_to_results = "results/reference_single_track_drift/ode";
train = train_arr[end];
u0 = train.states[:, 1];

mc = parameters_vehicle1();

prob = ODEProblem(st_model!, train.states[:, 1], train.tspan, [train.input_interpolant, mc]);
sol = Array(solve(prob, saveat=0.1));

#=
    final simulation error training set
=#
sim_err = l2loss(train.scaled_states, StatsBase.transform(scaler_states, sol));
println("Simulation error training set $sim_err");

plot_fv(train.tsteps, train.states, sol; safe_file="$path_to_results/ode.html");

#=
    final simulation error validation set
=#
prob = remake(prob, u0=vali.states[:, 1], tspan=vali.tspan);
sol = Array(solve(prob, saveat=0.1));

sim_err = l2loss(vali.scaled_states, StatsBase.transform(scaler_states, sol));
println("Simulation error validation set $sim_err");

plot_fv(vali.tsteps, vali.states, sol; safe_file="$path_to_results/ode_vali-set.html");
