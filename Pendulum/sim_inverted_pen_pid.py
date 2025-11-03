# %% Inverted Pendulum

import numpy as np
import matplotlib.pyplot as plt
from time import time
import math

M = 0.5  # Cart mass [kg]
k = 0.1  # Cart friction [N.s/m]

m = 0.2  # Pendulum mass [kg]
c = 0.01  # Pendulum friction [N.s/rad]
l = 0.3  # Length [m]

g = 9.81  # Gravity [m/s^2]

Lt = 60  # Simulation time [s]
dt = 0.01  # Discretization [s]
Nt = round(Lt / dt)
t = np.arange(Nt) * dt

tht0 = 0
x0 = -0.5

tht = tht0 * np.ones(Nt)
x = x0 * np.ones(Nt)
e = np.zeros(Nt)

spx = np.zeros(Nt)
spx[round(1 * Nt / 5):] = .5
spx[round(2 * Nt / 5):] = -.5
spx[round(3 * Nt / 5):] = 0

a0 = (M + m) / dt**2 + k / dt
a1 = (2 * (M + m) / dt**2 + k / dt) / a0
a2 = - ((M + m) / dt**2) / a0
a3 = m * l / a0
a4 = -m * l / a0

b0 = m * l / dt**2 + c / dt
b1 = (2 * m * l / dt**2 + c / dt) / b0
b2 = -(m * l / dt**2) / b0
b3 = m * g / b0
b4 = m / b0


def coords(x, tht):  # To draw pendulum and cart
    return ((x - l * np.sin(tht), x, x - l / 2, x - l / 2, x + l / 2, x + l / 2, x),
            (l * np.cos(tht), 0, 0, -l / 2, -l / 2, 0, 0))


fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
xlim = 5 * l
ylim = 1.1 * l
ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim))
ax.grid(True)
xx, yy = coords(x0, tht0)
line = ax.plot(xx[0], yy[0], "bo", xx, yy, "b")

# PID controller
P = 15
I = 25
D = 1
E = 0  # Cumulative error
F = 0  # Force

# %%

view = 1
view_max = 50
ctrl = True

t0 = time()
for nt in range(3, Nt):
    thd = (tht[nt - 1] - tht[nt - 3]) / (2 * dt)
    thdd = (tht[nt - 1] - 2 * tht[nt - 2] + tht[nt - 3]) / (dt**2)

    x[nt] = a1 * x[nt - 1] + \
            a2 * x[nt - 2] + \
            a3 * thdd * np.cos(tht[nt - 1]) + \
            a4 * thd**2 * np.sin(tht[nt - 1]) + \
            F / a0

    if x[nt] > xlim:
        x -= xlim
    elif x[nt] < -xlim:
        x += xlim

    xdd = (x[nt] - 2 * x[nt - 1] + x[nt - 2]) / (dt**2)

    tht[nt] = b1 * tht[nt - 1] + \
              b2 * tht[nt - 2] + \
              b3 * np.sin(tht[nt - 1]) + \
              b4 * xdd * np.cos(tht[nt - 1])

    if ctrl:
        e[nt] = 1.0 * (0. - tht[nt]) - 0.2 * (spx[nt] - x[nt])
        E += e[nt] * dt
        de = (e[nt] - e[nt - 1]) / dt
        F = P * e[nt] + I * E + D * de
        F += 1 * np.random.randn()  # Disturbances
    else:
        F = 0

    if nt > round(4 * Nt / 5):
        ctrl = False

    if not nt % math.ceil(view):
        xx, yy = coords(x[nt], tht[nt])
        line[0].set_data([xx[0]], [yy[0]])
        line[1].set_data(xx, yy)
        ax.set_title(f"Ctrler = {ctrl}, spx = {spx[nt]:.2f}, t = {nt * dt:.2f}")

    delay = nt * dt - time() + t0
    if delay <= 0:
        view = view + 1 if view < view_max else view_max
    else:
        view = view - 1 if view > 1 else 1
        plt.pause(delay)

# %%
fig, ax1 = plt.subplots()
ax1.plot(t, spx, color="cornflowerblue", label="Set point of cart position")
ax1.plot(t, x, color="blue", linewidth=1, label="Cart position")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("$x$ [m]", color="blue")
ax1.tick_params(axis="y", colors="blue")
ax2 = ax1.twinx()
ax2.plot(t, tht, color="orange", linewidth=1, label="Pendulum angle")
ax2.set_ylabel("$\\theta$ [rad]", color="orange")
ax2.tick_params(axis="y", colors="orange")
ax1.grid(True)
ax1.set_title("Pendulum")
fig.legend()