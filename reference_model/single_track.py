from scipy.integrate import odeint
from vehiclemodels.init_st import init_st
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
import src

initialState = src.init_state()
p = parameters_vehicle1()
tspan, sampling = src.t_steps()

src.plot_inputs(tspan)

# simulate the singe-track model
x0 = init_st(initialState)
sol = odeint(src.single_track_model, x0, tspan, args=(p,))

src.export_2_mat(tspan, sampling, sol, "reference_single_track.mat")
src.plot_states(sol, tspan)
