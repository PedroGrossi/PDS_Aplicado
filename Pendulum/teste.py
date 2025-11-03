import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles
from time import time

print("pole placement")

M = 0.5  # Cart mass [kg]
k = 0.1  # Cart friction [N.s/m]

m = 0.2  # Pendulum mass [kg]
c = 0.01  # Pendulum friction [N.s/rad]
l = 0.3  # Length [m]

g = 9.81  # Gravity [m/s^2]

Lt = 30  # Simulation time [s]
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

# p_des = np.array([-4+4j, -4-4j, -6, -7])
p_des = np.array([-5+5j, -5-5j, -7, -9])

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