import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.integrate
from scipy.integrate import odeint
import sympy as sp

def formY(y, t, fV, fOm):
    y1,y2,y3,y4 = y
    dydt = [y3,y4,fV(y1,y2,y3,y4),fOm(y1,y2,y3,y4)]
    return dydt

def formY2(y, t, fOm):
    y1,y2 = y
    dydt = [y2,fOm(y1,y2)]
    return dydt

#defining parameters
m = 0.1
R = 1.5
# phi0 = 3*np.pi/4
phi0=np.pi/4
ksi0 = 0.1
g = 9.81
l = 10

#defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

#defining ksi, phi, V=dksi/dt and om=dphi/dt as functions of 't'
# ksi = 3
ksi = 0
phi = sp.Function('phi')(t)
V = 0
om = sp.Function('om')(t)

#constructing the Lagrange equations
#1 defining the kinetic energy
v2 = om**2 * (R**2 + ksi**2) + V**2 - 2 * V * om * R
w = V / R - om
J = 1 / 2 * m * R**2
TT = (m*v2)/2 + (J*w**2)/2
# TT = 0.5*m*(om**2 * (R**2+ksi**2) + V**2 - (2 * om * V * R)) + 0.25 * m * R**2 * (V / R - om)**2
# TT = om / 2 * (m * (3/2*R**2 + l ** 2))
#2 defining potential energy
PP = -m*g*(ksi*sp.cos(phi)-R*sp.sin(phi))
#Lagrange function
L = TT-PP

#equations
# ur1 = sp.diff(sp.diff(L,V),t)-sp.diff(L,ksi)
ur2 = sp.diff(sp.diff(L,om),t)-sp.diff(L,phi)
print(ur2)
a22 = ur2.coeff(sp.diff(om,t),1)
b2 = -ur2.coeff(sp.diff(om,t),0).subs(sp.diff(phi,t), om)

domdt = b2/a22
print(domdt)

countOfFrames = 300

# Constructing the system of differential equations
T = np.linspace(0, 12, countOfFrames)
fOm = sp.lambdify([phi,om], domdt, "numpy")
y0 = [-2, -0.1]
sol = odeint(formY2, y0, T, args = (fOm, ))

#sol - our solution
#sol[:,0] - ksi
#sol[:,1] - phi
#sol[:,2] - dksi/dt
#sol[:,3] - dphi/dt

Ksi = l
# Phi = sol[:,0] + np.pi/8
# Phi = sol[:,0]/10+ np.pi/20
Phi = sol[:,0] + np.pi/2
print(sol[:,0])
Om  = sol[:,1]

Steps = 300
t = np.linspace(0, 10, Steps)

X_O = 3 # координаты точки О
Y_O = 10

X_A = Ksi * np.sin(Phi) + X_O
Y_A = - Ksi * np.cos(Phi) + Y_O

X_C = X_O + Ksi * np.sin(Phi) + R * np.cos(Phi)
Y_C = Y_O - Ksi * np.cos(Phi) + R * np.sin(Phi)

angle = np.linspace(0, np.pi*2, 150)
X_Circle = R*np.cos(angle)
Y_Circle = R*np.sin(angle)

X_Ground = [0, 6] # это подвес, на котором держится точка О
Y_Ground = [10, 10]
lSt= 10
fig = plt.figure(figsize=[lSt + 0.5, lSt + 0.5])
ax = fig.add_subplot(1, 2, 1)
ax.axis('equal')
ax.set(xlim=[X_O - (lSt + 0.5), X_O + (lSt + 0.5)],
       ylim=[Y_O/2 - (lSt + 0.5), Y_O/2 + (lSt + 0.5)])

ax.plot(X_Ground, Y_Ground, color='black', linewidth=3)
Drawed_Circle = ax.plot(X_C[0]+X_Circle, Y_C[0]+Y_Circle)[0]
Line_OA = ax.plot([X_O, X_A[0]],[Y_O, Y_A[0]])[0] # линия, соединяющая точки O и А

Point_O = ax.plot(X_O, Y_O, marker='o', markersize=10)[0]
Point_A = ax.plot(X_A[0], Y_A[0], marker='o')[0]
Point_C = ax.plot(X_C[0], Y_C[0], marker='o')[0]

ax1 = fig.add_subplot(4, 2, 2)
ax1.plot(T, sol[:,0])
plt.xlabel('T')
plt.ylabel('Phi')

ax1 = fig.add_subplot(4, 2, 6)
ax1.plot(T, sol[:,1])
plt.xlabel('T')
plt.ylabel("Phi'")


def Kino(i):
    Point_O.set_data(X_O, Y_O)
    Point_A.set_data(X_A[i], Y_A[i])
    Line_OA.set_data([X_O, X_A[i]], [Y_O, Y_A[i]])

    Point_C.set_data(X_C[i], Y_C[i])
    Drawed_Circle.set_data(X_C[i]+X_Circle, Y_C[i]+Y_Circle)
    return [Point_O, Point_A, Line_OA, Point_C, Drawed_Circle]

anima = FuncAnimation(fig, Kino, frames=Steps, interval=10)

plt.show()