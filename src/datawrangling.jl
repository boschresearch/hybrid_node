using MAT, StatsBase, Noise

struct Sim
    tspan::Tuple{Float64, Float64}
    tsteps::StepRangeLen
    states_interpolant::Vector{Function}
    input_interpolant::Vector{Function}
    states::Matrix{Float64}
    scaled_states::Matrix{Float64}
end

function readmat(file)
    return matread(file)
end

function data_interpolants(data)
    return (states_interpolant(data), input_interpolant(data))
end

function scalers(data, tspan)
    tsteps = range(tspan[1], tspan[2], step=0.1)
    states_interpolant, input_interpolant = data_interpolants(data)
    states = interp_matrix(states_interpolant, tsteps)
    inputs = interp_matrix(input_interpolant, tsteps)
    scaler_states = StatsBase.fit(ZScoreTransform, states, dims=2)
    scaler_inputs = StatsBase.fit(ZScoreTransform, inputs, dims=2)

    return (scaler_states, scaler_inputs)
end

function sim_dat(data, tspan, scaler_states, σ_noise)
    states_interpolant, input_interpolant = data_interpolants(data)
    tsteps = range(tspan[1], tspan[2], step=0.1)

    states = interp_matrix(states_interpolant, tsteps)
    scaled_states = StatsBase.transform(scaler_states, states)
    scaled_states = add_gauss(scaled_states, σ_noise)
    states = StatsBase.reconstruct(scaler_states, scaled_states)

    return Sim(tspan, tsteps, states_interpolant, input_interpolant, 
                states, scaled_states)
end

function states_interpolant(data)
    xPos_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["x_State"]["xPos"]))
    yPos_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["x_State"]["yPos"]))
    psi_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["x_State"]["psi"]))
    delta_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["x_State"]["delta"]))
    v_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["x_State"]["v"]))
    β_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["x_State"]["beta"]))
    ω_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["x_State"]["dPsi"]))
    
    [t-> xPos_interpolant(t),
    t-> yPos_interpolant(t),
    t-> psi_interpolant(t),
    t-> delta_interpolant(t),
    t-> v_interpolant(t),
    t-> β_interpolant(t),
    t-> ω_interpolant(t)]
end

function input_interpolant(data)
    U_ax_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["u_Input"]["U_ax"]))
    U_vdelta_interpolant = LinearInterpolation(vec(data["t_Ref"]), vec(data["u_Input"]["U_vdelta"]))

    [t-> U_ax_interpolant(t),
    t-> U_vdelta_interpolant(t)]
end

function interp_vector(interpolant, t)
    u = Array{Float64}(undef, length(interpolant))

    for (idx, func) in enumerate(interpolant)
        u[idx] = func(t)
    end
    return u
end

function interp_matrix(interpolant, t)
   state_mat = Array{Float64}(undef, (length(interpolant), length(t)))

    for (idx, val) in enumerate(t)
        state_mat[:, idx] = interp_vector(interpolant, val)
    end
    return state_mat
end

function l2loss(x, y)
    return sum(abs2, x .- y)
end

function l2loss(x, y, scaler::ZScoreTransform)
    return l2loss(z_score(x, scaler), z_score(y, scaler))
end

function z_score(x, scaler::ZScoreTransform)
    return (x .- scaler.mean) ./ scaler.scale
end