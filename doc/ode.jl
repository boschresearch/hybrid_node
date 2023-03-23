using DifferentialEquations, StatsBase, Random, Noise
using HybridNode

include("reference_single_track_drift.jl");
path_to_results = "results/reference_single_track_drift/ode";
train = train_arr[end];
u0 = train.states[:, 1];

mc = parameters_vehicle1();

prob = ODEProblem(st_model!, train.states[:, 1], train.tspan, [train.input_interpolant, mc]);
sol = Array(solve(prob, saveat=0.1));

plot_fv(train.tsteps, train.states, sol; safe_file="$path_to_results/ode.html");