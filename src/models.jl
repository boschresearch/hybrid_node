using Interpolations

struct ModelConsts
    m:: Float64
    Θ:: Float64
    lf:: Float64
    lr:: Float64
    hcg:: Float64
    c1:: Float64
    c2:: Float64
    μ:: Float64
    g:: Float64
end

function parameters_vehicle1()
    mc = ModelConsts(1225, 1538, 0.883, 1.508, 0.59436,
        20.89, 20.89, 1.048, 9.81)

    return mc
end

function st_model!(du, u, p, t)
    Δx, Δy, ψ, δ, v, β, ψ̇ = u

    input = p[1]

    ax = input[1](t)
    v_delta = input[2](t)
    mc = p[2]

    Θ = mc.Θ; m = mc.m; μ = mc.μ; C_Sf = mc.c1; 
    C_Sr = mc.c2; lr = mc.lr; lf = mc.lf; g = mc.g
    h = mc.hcg

    # single track state space model
    du[1] = v * cos(ψ + β)
    du[2] = v * sin(ψ + β)
    du[3] = ψ̇ 
    du[4] = v_delta
    du[5] = ax
    du[6] = (
        (μ / (v^2 * (lr+lf)) * (C_Sr * (g*lf + ax*h) * lr 
        - C_Sf * (g*lr - ax*h) * lf) - 1) * ψ̇  
        - μ / (v * (lr+lf)) * (C_Sr * (g*lf + ax*h) + C_Sf * (g*lr - ax*h)) * β 
        + μ / (v * (lr+lf)) * (C_Sf * (g*lr - ax*h)) * δ
    )
    du[7] = (
        - μ*m / (v*Θ * (lr+lf)) * (lf^2 * C_Sf * (g*lr - ax*h) 
        + lr^2 * C_Sr * (g*lf + ax*h)) * ψ̇ 
        + μ*m / (Θ * (lr+lf)) * (lr*C_Sr * (g*lf + ax*h) - lf*C_Sf * (g*lr - ax*h)) * β 
        + μ*m / (Θ * (lr+lf)) * lf * C_Sf * (g*lr - ax*h) * δ 
    )
end