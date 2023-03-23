#= This script fits a hybrid neural ode (also known as universal differential equation
    UDE) to reference data from a vehicle single track drift model.
    
    The size of the network can be adjusted with `nn_size` and results for 
    training and prediction error as well as figures are saved. The hybrid neural ode runs 
    with exogenous inputs. 
    
    The first four states are modeled with equations from first principles, 
    whereas the last three states are modeled by a neural ode.

    A template for hybrid neural odes can also be found here:
    https://github.com/ChrisRackauckas/universal_differential_equations/blob/master/LotkaVolterra/scenario_1.jl
=#

using Lux, DiffEqFlux, DifferentialEquations, Optimization,
    StatsBase, Random, CSV
using HybridNode

#=
    manipulate the size of the hidden neural network layer here, e.g 
    nn_size=10
    nn_size=15
=#
nn_size = 5;
learning_rate = 0.025;

include("reference_single_track_drift.jl");

results = Dict("train_err"=>[], "pred_err"=>[]);
println("\n Experiment with layer_size $nn_size \n");
path_to_results = "results/reference_single_track_drift/hynode";

#=
    define hybrid neural ode (universal differential equation)
=#
Random.seed!(rng, 5); # optionally, adjust random nn weight initialization
nn = Lux.Chain(Lux.Dense(6, nn_size, tanh), Lux.Dense(nn_size, 3));
p0, st = Lux.setup(rng, nn);
p0 = Lux.ComponentArray(p0);

function ude_st_model!(du, u, p, t)
    ax = train.input_interpolant[1](t)
    v_delta = train.input_interpolant[2](t)

    Δx, Δy, ψ, δ, v, β, ψ̇ = u
    #= The first four states come from 
       single track state space model. 
       All four equations are kinematic raltions and do 
       not need specific vehicle parameters.
    =#
    du[1] = v * cos(ψ + β)
    du[2] = v * sin(ψ + β)
    du[3] = ψ̇ 
    du[4] = v_delta

    u_scaled = StatsBase.transform(scaler_states, u)
    input_scaled = StatsBase.transform(scaler_input, [ax, v_delta])

    du[5:7] .= nn(vcat(u_scaled[4:7], input_scaled), p, st)[1]

    return nothing 
end;

function predict(θ, tsteps)
    _prob = remake(prob_nn, p=θ)
    Array(solve(_prob, Tsit5(), saveat=tsteps, abstol=1e-8, reltol=1e-6))
end;

function callback(θ, loss, preds)
    display(loss)
    return false
end;

#= 
    multiple shooting training approach
=#
maxit = 2000;
group_size = 80;
continuity_term = 1;

loss_function(data, pred) = l2loss(data, pred, scaler_states);
adtype = Optimization.AutoZygote();

function train_model(res)
    isnothing(res) ? p = p0 : p = res.u

    # update model with new training data
    prob_nn = ODEProblem(ude_st_model!, train.states[:, 1], train.tspan, p)

    loss_multiple_shooting(p) = multiple_shoot(p, train.states, train.tsteps,
        prob_nn, loss_function, Tsit5(), group_size; continuity_term)
    optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, p)
        
    res = Optimization.solve(optprob, ADAM(learning_rate), callback=callback, maxiters=maxit)
    return res, prob_nn
end

function training_loop()
    res = nothing
    prob_nn = nothing
    for sample in train_arr
        global train = sample
        println("\n New training sample \n")
        res, prob_nn = train_model(res)
    end
    return res, prob_nn
end

res, prob_nn = training_loop()

# hybrid node fit for the training data
X̂_train = predict(res.u, train.tsteps);

#=
    final training error
=#
train_err = l2loss(train.scaled_states, StatsBase.transform(scaler_states, X̂_train));
println("Training error $train_err");
push!(results["train_err"], train_err);

# hybrid node prediction for the validation data
prob_nn = remake(prob_nn, u0=vali.states[:, 1], tspan=vali.tspan);
X̂_val = predict(res.u, vali.tsteps);
#=
    prediction error
=#
pred_err = l2loss(vali.scaled_states, StatsBase.transform(scaler_states, X̂_val));
println("Prediction error $pred_err");
push!(results["pred_err"], pred_err);

plot_fv(vcat(train.tsteps, vali.tsteps), hcat(train.states, vali.states), 
        hcat(X̂_train, X̂_val), 70; 
        safe_file="$path_to_results/hynode_val_$nn_size.html");

CSV.write("$path_to_results/hynode_$nn_size.csv", results);