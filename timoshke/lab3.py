import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
from scipy.integrate import odeint


def odesys(y, t, R, g):
    # y = [phi, ksi, phi', ksi']

    phi = y[0]
    ksi = y[1]
    dphi = y[2]
    dksi = y[3]

    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = -R
    a12 = 1
    a21 = ksi
    a22 = 0

    b1 = (2 / 3) * (ksi * (dphi**2) + g*np.cos(phi))
    b2 = -g * np.sin(phi) - 2*dksi*dphi + R*dphi**2

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)
    print(dy)
    return dy

g = 9.81
R = 3

phi0 = np.pi / 3 
ksi0 = 0.1
dphi0 = 0
dksi0 = 0

y0 = [phi0, ksi0, dphi0, dksi0]

Steps = 1000
t = np.linspace(0, 10, Steps)

Y = odeint(odesys, y0, t, (R, g))

Phi = Y[:, 0]
Ksi = Y[:, 1]
dPhi = Y[:, 2]
dKsi = Y[:, 3]

# print(Phi[:100])
print(Ksi[:10])


fig = plt.figure(figsize=[7, 7])
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim=[-20, 20],
       ylim=[-35, 5])

# Steps = 1000
# t = np.linspace(0, 10, Steps)
# Phi = 0.7 * np.cos(t)
# Ksi = 2 * t

Ox = 0
Oy = 4
length = 2 + Ksi
Bx = length * np.sin(Phi) + Ox
By = - length * np.cos(Phi) + Oy

# R = 3
beta = np.linspace(0, 6.28, 1000)
X_circle = R * np.sin(beta) + R
Y_circle = R * np.cos(beta)

Line = ax.plot([-4, 4], [Oy, Oy], 'g', lw=4)
Thread = ax.plot([Bx[0], Ox], [By[0], Oy], 'b', lw=4)[0]
Circle = ax.plot(X_circle + Bx[0], Y_circle + By[0], 'k', lw=2)[0]
Centre = ax.plot(Bx[0] + R, By[0], 'k', marker='o', ms=5)[0]
Base = ax.plot(Ox, Oy, 'r', marker='o', ms=10)[0]


def Animation(i):
    Circle.set_data(X_circle + Bx[i], Y_circle + By[i])
    Centre.set_data(Bx[i] + R, By[i])
    Thread.set_data([Bx[i], Ox], [By[i], Oy])
    return [Centre, Thread, Circle]


show = FuncAnimation(fig, Animation, frames=Steps, interval=10)

plt.show()
