module HybridNode
    using Revise

    include("datawrangling.jl")
    include("figures.jl")
    include("models.jl")

    export Sim, readmat, data_interpolants, interp_vector, 
        scalers, sim_dat, l2loss,

        plot_inputs, plot_fv,

        parameters_vehicle1, physical_model!, st_model!
end