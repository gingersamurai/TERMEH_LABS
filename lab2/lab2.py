import math

import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.animation import FuncAnimation

Steps = 1000

t = sp.Symbol('t')

phi = 4 * sp.sin(t)
thetta = math.pi / 4 * 5 * t

omega_phi = sp.diff(phi, t)
omega_thetta = sp.diff(thetta, t)

l = OA = AB = 5
X_A = OA * sp.cos(phi)
Y_A = OA * sp.sin(phi)

X_B =  X_A + AB * sp.sin(thetta)
Y_B =  Y_A  - AB * sp.cos(thetta)


V_A = sp.diff(phi,t) * l
V_r = sp.diff(thetta,t) * l

V_B = sp.sqrt(((l * omega_phi) ** 2) + ((l * omega_thetta) ** 2) - 2 * l * l * omega_phi * omega_thetta * sp.cos(phi - thetta))

Nv= 3
R1 = 0.001
R2 = 0.4
Ksi = np.linspace(0, 0, )


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-10,10], ylim=[-10,10])


T            = np.linspace(0, 10, Steps)
Phi          = np.zeros_like(T)
Omega_phi    = np.zeros_like(T)
XB           = np.zeros_like(T)
YB           = np.zeros_like(T)
XA           = np.zeros_like(T)
YA           = np.zeros_like(T)
Alpha        = np.zeros_like(T)
Phi          = np.zeros_like(T)
VB           = np.zeros_like(T)
VA           = np.zeros_like(T)

for i in range(len(T)):
    Phi[i] = sp.Subs(phi, t, T[i])
    Omega_phi[i] = sp.Subs(omega_phi, t, T[i])
    XA[i] = sp.Subs(X_A, t, T[i])
    YA[i] = sp.Subs(Y_A, t, T[i])
    XB[i] = sp.Subs(X_B, t, T[i])
    YB[i] = sp.Subs(Y_B, t, T[i])
    Phi[i] = sp.Subs(phi, t, T[i])
    VB[i] = sp.Subs(V_B, t, T[i])
    VA[i] = sp.Subs(V_A, t, T[i])

alpha = np.linspace(0, Nv*6.283+Phi[0], Steps)
X_SpiralSpr = -(R1 + alpha * (R2 - R1) / alpha[-1]) * np.sin(alpha)
Y_SpiralSpr = (R1 + alpha * (R2 - R1) / alpha[-1]) * np.cos(alpha)

beta = np.linspace(0, 2*math.pi, Steps)
R_Circle = 0.5
X_Circle = R_Circle * np.cos(beta)
Y_Circle = R_Circle * np.sin(beta)

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(T, Phi, color='blue')
ax_for_graphs.set_title("Phi(t)")
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(T, VA, color='red')
ax_for_graphs.set_title('Va(t)')
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,3)
ax_for_graphs.plot(T, Omega_phi, color='green')
ax_for_graphs.set_title("phi'(t) = omega_phi(t)")
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(T, VB, color='black')
ax_for_graphs.set_title("Vb(t)")

ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

OX = ax.plot([-12, 12], [-0.5,-0.5], 'black', linestyle = '--')
Draw_Spring = ax.plot(X_SpiralSpr, Y_SpiralSpr, color='#666666')[0]
Draw_OA=ax.plot([0, XA[0]], [0, YA[0]], color='#808080')[0]
Draw_AB=ax.plot([XA[0], XB[0]], [YA[0], YB[0]], color='#808080' )[0]
PointB = ax.plot(XB[0], YB[0])[0]
PointA = ax.plot(XA[0], YA[0], color='#a0a0a0', marker='o')[0]
Draw_Circle = ax.plot(X_Circle + XB[0], Y_Circle + YB[0], color='black', linewidth=1)[0]
triangle = pat.Polygon([(0,0), (-0.5, -0.5), (0.5, -0.5)], color='#d3d3d3')
ax.add_patch(triangle)

def update(i):
    PointB.set_data(XB[i],YB[i])
    Draw_OA.set_data([0, XA[i]], [0, YA[i]])
    PointA.set_data(XA[i], YA[i])
    Draw_AB.set_data([XA[i], XB[i]], [YA[i], YB[i]])
    Draw_Circle.set_data(X_Circle + XB[i], Y_Circle + YB[i])
    alpha = np.linspace(0, Nv*6.28+Phi[i], 100)
    X_SpiralSpr = -(R1 + alpha * (R2 - R1) / alpha[-1]) * np.sin(alpha - 1.57)
    Y_SpiralSpr = (R1 + alpha * (R2 - R1) / alpha[-1]) * np.cos(alpha - 1.57)
    Draw_Spring.set_data(X_SpiralSpr, Y_SpiralSpr)
    return [PointB, Draw_OA, Draw_Spring, Draw_AB, PointA, Draw_Circle]

anima = FuncAnimation(fig, update, frames=Steps, interval=1)
plt.show()