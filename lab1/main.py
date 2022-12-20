import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp


def Rod2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY



t = sp.symbols('t')

r = 2 + sp.sin(12 * t)
phi = 1.8 * t + 0.2 * sp.cos(12 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)

Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)

V = sp.sqrt(Vx**2 + Vy**2)
A = sp.sqrt(Ax**2 + Ay**2)

Atan = sp.diff(V, t)
Anorm = sp.sqrt(A**2 - Atan**2)

Curve = V**2 / Anorm ** 2

T = np.linspace(0, 10, 100)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
A = np.zeros_like(T)
AT = np.zeros_like(T)
CurveV = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    CurveV[i] = sp.Subs(Curve, t, T[i])

fig, ax = plt.subplots()
ax.plot(X, Y)

Point, = ax.plot(X[0], Y[0], marker='o')
Vline, = ax.plot([X[0], X[0]+VX[0]],[Y[0], Y[0]+VY[0]], color='red')

ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])

VVecArrowX, VVecArrowY = Rod2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VVecArrow, = ax.plot(VVecArrowX+X[0]+VX[0], VVecArrowY+ Y[0]+VY[0])


plt.show()