#=  
    This script fits a neural ode to reference data from a vehicle single track drift 
    model. The size of the network can be adjusted with `nn_size` and results for 
    training and prediction error as well as figures are saved. The neural ode runs 
    with exogenous inputs. 
    
    A template code from DiffEqFlux docu is here:
    https://sensitivity.sciml.ai/dev/ode_fitting/exogenous_input/#Example-of-a-Neural-ODE-with-Exogenous-Input
=#

using Lux, DiffEqFlux, DifferentialEquations, Optimization,
    StatsBase, Random, CSV
using HybridNode

#=
    manipulate the size of the hidden neural network layer here, e.g 
    nn_size=10
    nn_size=15
=#
nn_size = 12;
learning_rate = 0.05;

include("reference_single_track_drift.jl");

results = Dict("train_err"=>[], "pred_err"=>[]);
println("\n Experiment with layer_size $nn_size \n");
path_to_results = "results/reference_single_track_drift/node";

#=
    define neural ode
=#
Random.seed!(rng, 3); # optionally, adjust random nn weight initialization
nn = Lux.Chain(Lux.Dense(9, nn_size, tanh), Lux.Dense(nn_size, 7));
p0, st = Lux.setup(rng, nn);
p0 = Lux.ComponentArray(p0);

function dudt!(du, u, p, t)
    ax = train.input_interpolant[1](t)
    v_delta = train.input_interpolant[2](t)

    input_scaled = StatsBase.transform(scaler_input, [ax, v_delta])

    du .= nn(vcat(u, input_scaled),  p, st)[1]
    return nothing
end;

function predict(θ, tsteps)
    _prob = remake(prob_nn, p=θ)
    Array(solve(_prob, Tsit5(), saveat=tsteps, abstol=1e-8, reltol=1e-6))
end;

function callback(Θ, loss, preds)
    display(loss)
    return false
end;

#=
    multiple shooting training approach
=#
maxit = 2000;
group_size = 80;
continuity_term = 1;

loss_function(data, pred) = l2loss(data, pred)
adtype = Optimization.AutoZygote()

function train_model(res)
    isnothing(res) ? p = p0 : p = res.u
    
    # update model with new training data
    prob_nn = ODEProblem(dudt!, train.scaled_states[:, 1], train.tspan, p)
    
    loss_multiple_shooting(p) = multiple_shoot(p, train.scaled_states, train.tsteps, 
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

# node prediction for the training data
X̂_train = predict(res.u, train.tsteps);
#=
    final training error
=#
train_err = l2loss(train.scaled_states, X̂_train);
println("Training error $train_err");
push!(results["train_err"], train_err);

# node prediction for the validation data
prob_nn = remake(prob_nn, u0=vali.scaled_states[:, 1], tspan=vali.tspan);
X̂_val = predict(res.u, vali.tsteps);

#=
    prediction error
=#
pred_err = l2loss(vali.scaled_states, X̂_val);
println("Prediction error $pred_err");
push!(results["pred_err"], pred_err);

plot_fv(vcat(train.tsteps, vali.tsteps), hcat(train.states, vali.states), 
        re_scale(hcat(X̂_train, X̂_val)), 70; 
        safe_file="$path_to_results/node_val_$nn_size.html");

CSV.write("$path_to_results/node_$nn_size.csv", results);