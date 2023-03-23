from scipy.integrate import odeint
from vehiclemodels.init_std import init_std
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1

import src
import importlib
importlib.reload(src)

initialState = src.init_state()
p = parameters_vehicle1()
tspan, sampling = src.t_steps()

src.plot_inputs(tspan)

# simulate single-track drift model
x0 = init_std(initialState, p)
sol = odeint(src.single_track_drift_model, x0, tspan, args=(p,))

src.export_2_mat(tspan, sampling, sol, "reference_single_track_drift.mat")
src.plot_states(sol, tspan)
