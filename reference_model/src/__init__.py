import math

from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
import numpy
from numpy import cos, sin
from scipy.io import savemat
import matplotlib.pyplot as plt
from matplotlib.pyplot import title


def steer_angle_velocity(t):
    return 0.02 * sin(t)


def acceleration_long(t):
    return 0.01 + 0.05 * cos(t) + 0.1 * sin(0.1*t)


def single_track_drift_model(x, t, params):
    u_t = [steer_angle_velocity(t), acceleration_long(t)]
    f = vehicle_dynamics_std(x, u_t, params)
    return f


def single_track_model(x, t, params):
    u_t = [steer_angle_velocity(t), acceleration_long(t)]
    f = vehicle_dynamics_st(x, u_t, params)
    return f


def init_state():
    sx0 = 0
    sy0 = 0
    delta0 = 0
    vel0 = 25
    psi0 = 0
    dot_psi0 = 0
    beta0 = 0

    return [sx0, sy0, delta0, vel0, psi0, dot_psi0, beta0]


def t_steps():
    start = 0
    final = 100
    sampling = 0.02
    return numpy.arange(start, final, sampling), sampling


def export_2_mat(tspan, sampling, sol, name):
    sol_export = {
        "t_Ref": tspan.T,
        "Ts": sampling,
        "x_State": {
            "xPos": sol[:, 0],
            "yPos": sol[:, 1],
            "delta": sol[:, 2],
            "v": sol[:, 3],
            "psi": sol[:, 4],
            "dPsi": sol[:, 5],
            "beta": sol[:, 6]
        },
        "u_Input": {
            "U_ax": acceleration_long(tspan),
            "U_vdelta": steer_angle_velocity(tspan),
            "U_delta": sol[:, 2]
        }
    }
    savemat("../" + name, sol_export, oned_as='column')


def plot_inputs(tspan):
    title('steer velocity')
    plt.plot(tspan, steer_angle_velocity(tspan))
    plt.xlabel("t")
    plt.show()

    title('acceleration')
    plt.plot(tspan, acceleration_long(tspan))
    plt.xlabel("t")
    plt.show()


def plot_states(sol, tspan):
    title('positions')
    plt.scatter([tmp[0] for tmp in sol],
                [tmp[1] for tmp in sol],
                c=[tmp[3] for tmp in sol])
    cbar = plt.colorbar()
    cbar.set_label('velocity')
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    title('steer angle')
    plt.plot(tspan, [tmp[2] for tmp in sol])
    plt.xlabel("t")
    plt.show()

    title('velocity')
    plt.plot(tspan, [tmp[3] for tmp in sol])
    plt.xlabel("t")
    plt.show()

    title('yaw angle')
    plt.plot(tspan, [tmp[4] for tmp in sol])
    plt.xlabel("t")
    plt.show()

    title('yaw rate')
    plt.plot(tspan, [tmp[5] for tmp in sol])
    plt.xlabel("t")
    plt.show()

    title('slip angle')
    plt.plot(tspan, [tmp[6] for tmp in sol])
    plt.xlabel("t")
    plt.show()
