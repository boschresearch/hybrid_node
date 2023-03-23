using Plots

function plot_inputs(reference_data)
    plotly()

    p1 = plot(reference_data["t_Ref"], reference_data["u_Input"]["U_ax"], color=:black)
    ylabel!("ax")

    p2 = plot(reference_data["t_Ref"], reference_data["u_Input"]["U_vdelta"], color=:black)
    ylabel!("vdelta")
    xlabel!("time")
    gui(plot(p1, p2, layout=(2,1), legend=false))

    return nothing
end

function plot_fv(tsteps, data, sim, vline_pos=nothing; safe_file="test.html")
    plotly()

    function line_plots(tsteps, data, idx, yaxis, label)
        plt = plot(tsteps, data[idx,:], yaxis=yaxis, label=label[1], lw=1.5, color=:black)
        plt = plot!(tsteps, sim[idx,:], color=:red, label=label[2], lw=1.5)
    end

    function panel(tsteps, data, idx, yaxis; label=["", ""])
        line_plots(tsteps, data, idx, yaxis, label)
    end

    function panel(tsteps, data, idx, yaxis, vline_pos; label=["", ""])
        line_plots(tsteps, data, idx, yaxis, label)
        vline!([vline_pos], line=:black, label="")
    end

    if vline_pos === nothing
        plt1 = panel(tsteps, data, 1, "x"; label=["data", "model"])
        plt2 = panel(tsteps, data, 2, "y")
        plt3 = panel(tsteps, data, 3, "ψ")
        plt4 = panel(tsteps, data, 4, "δ")
        plt5 = panel(tsteps, data, 5, "v")
        plt6 = panel(tsteps, data, 6, "β")
        plt7 = panel(tsteps, data, 7, "ω")
    else
        plt1 = panel(tsteps, data, 1, "x", vline_pos; label=["data", "model"])
        plt2 = panel(tsteps, data, 2, "y", vline_pos)
        plt3 = panel(tsteps, data, 3, "ψ", vline_pos)
        plt4 = panel(tsteps, data, 4, "δ", vline_pos)
        plt5 = panel(tsteps, data, 5, "v", vline_pos)
        plt6 = panel(tsteps, data, 6, "β", vline_pos)
        plt7 = panel(tsteps, data, 7, "ω", vline_pos)
    end

    l = @layout [a b; c d; e; f; g]

    gui(plot(plt1, plt2, plt3, plt4, plt5, plt6, plt7, layout=l)) 

    savefig(safe_file)

    return nothing

end