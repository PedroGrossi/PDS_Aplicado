import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles
from time import time
import math

print("pole placement")

M = 0.5  # Cart mass [kg]
k = 0.1  # Cart friction [N.s/m]

m = 0.2  # Pendulum mass [kg]
c = 0.01  # Pendulum friction [N.s/rad]
l = 0.3  # Length [m]

g = 9.81  # Gravity [m/s^2]

Lt = 50  # Simulation time [s]
dt = 0.01  # Discretization [s]

Nt = int(round(Lt / dt))
t = np.arange(Nt) * dt

spx = np.zeros(Nt)
spx[int(1 * Nt / 5):] = 0.5
spx[int(2 * Nt / 5):] = -0.5
spx[int(3 * Nt / 5):] = 0.0

x0 = -0.5
xdot0 = 0
th0 = 0
thdot0 = 0

state = np.array([x0, xdot0, th0, thdot0], dtype=float)
hist = np.zeros((Nt, 4))
us = np.zeros(Nt)

A = np.array([
    [0, 1, 0, 0],
    [0, 0, -m * g / M, 0],
    [0, 0, 0, 1],
    [0, 0, g * (M + m) / (M * l), 0]
], dtype=float)

B = np.array([0, 1.0 / M, 0, -1.0 / (M * l)], dtype=float).reshape((4, 1))

p_des = np.array([-2.5 + 2.5j, -2.5 - 2.5j, -3, -3.5])

K = place_poles(A, B, p_des).gain_matrix  # 1x4
K = np.atleast_2d(K)
K = K.reshape((1, 4))
# print("K = ", K)

# (M + m) * xdd + m*l*cos(theta)*thetadd - m*l*sin(theta)*thetadot^2 = u - k*xdot
# m*l*cos(theta)*xdd + m*l^2*thetadd - m*g*l*sin(theta) = - c*thetadot
def dynamics(state, u):
    x, xdot, th, thdot = state
    s, cth = np.sin(th), np.cos(th)
    D = np.array([[M + m,        m * l * cth],
                  [m * l * cth,  m * l**2  ]], dtype=float)
    rhs1 = u - k * xdot + m * l * s * thdot**2
    rhs2 = m * g * l * s - c * thdot
    rhs = np.array([rhs1, rhs2], dtype=float)
    xdd, thdd = np.linalg.solve(D, rhs)
    return np.array([xdot, xdd, thdot, thdd], dtype=float)

def rk4_step(state, u, dt):
    k1 = dynamics(state, u)
    k2 = dynamics(state + 0.5 * dt * k1, u)
    k3 = dynamics(state + 0.5 * dt * k2, u)
    k4 = dynamics(state + dt * k3, u)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

u_max = 50.0

# u = -K (x - x_ref)
# ref = [spx, 0, 0, 0]
for i in range(Nt):
    ref = np.array([spx[i], 0.0, 0.0, 0.0])
    err = state - ref
    u = float(-K @ err)
    u_total = u + np.random.randn()
    u_total = max(min(u_total, u_max), -u_max)
    us[i] = u_total
    hist[i, :] = state
    state = rk4_step(state, u_total, dt)

x = hist[:, 0]
theta = hist[:, 2]

def coords(x, th):
    return ((x - l * np.sin(th), x, x - l / 2, x - l / 2, x + l / 2, x + l / 2, x),
            (l * np.cos(th), 0, 0, -l / 2, -l / 2, 0, 0))

fig, ax = plt.subplots()
ax.set_aspect('equal', 'box')
xlim = 5 * l
ylim = 1.1 * l
ax.set(xlim=(-xlim, xlim), ylim=(-ylim, ylim))
ax.grid(True)
xx, yy = coords(x0, th0)
line = ax.plot(xx[0], yy[0], "bo", xx, yy, "b")

view = 1
view_max = 50
t0 = 0
for nt in range(0, Nt, max(1, int(view))):
    xx, yy = coords(x[nt], theta[nt])
    line[0].set_data([xx[0]], [yy[0]])
    line[1].set_data(xx, yy)
    ax.set_title(f"Ctrler = PP, spx={spx[nt]:.2f}, t={nt*dt:.2f}")
    plt.pause(0.001)

fig2, ax1 = plt.subplots()
ax1.plot(t, spx, label="setpoint x")
ax1.plot(t, x, label="x")
ax1.set_xlabel("t [s]")
ax1.set_ylabel("x [m]")
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(t, theta, color="orange", label="theta")
ax2.set_ylabel("theta [rad]")
fig2.legend()
plt.show()

# armazenando variaveis
x_pp = x.copy()
theta_pp = theta.copy()
us_pp = us.copy()

print("pid")
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

    if not nt % math.ceil(view):
        xx, yy = coords(x[nt], tht[nt])
        line[0].set_data([xx[0]], [yy[0]])
        line[1].set_data(xx, yy)
        ax.set_title(f"Ctrler = PID, spx = {spx[nt]:.2f}, t = {nt * dt:.2f}")

    delay = nt * dt - time() + t0
    if delay <= 0:
        view = view + 1 if view < view_max else view_max
    else:
        view = view - 1 if view > 1 else 1
        plt.pause(delay)

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
plt.show()

x_pid = x.copy()
theta_pid = tht.copy()

print("Comparative")

fig, ax1 = plt.subplots()
ax1.plot(t, spx, 'k--', label='Setpoint')
ax1.plot(t, x_pp, 'b', label='Pole Placement')
ax1.plot(t, x_pid, 'r', label='PID')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Cart position [m]')
ax1.legend()
ax1.grid(True)
plt.show()

fig, ax1 = plt.subplots()
ax1.plot(t, theta_pp, 'b', label='Pole Placement')
ax1.plot(t, theta_pid, 'r', label='PID')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('$\\theta$ [rad]')
ax1.legend()
ax1.grid(True)
plt.show()
