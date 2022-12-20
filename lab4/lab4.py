import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as pat
import sympy as sp
from scipy.integrate import odeint
import math


def form2(y, t, fomega_phi):
    y1,y2 = y
    dydt = [y2, fomega_phi(y1,y2)]
    return dydt


Steps = 960
l = OA = AB = 1
R_Circle = l / 10
k1 = 1
k2 = 0
m = 0
m_oa = 5
g = 9.81
c = m_oa * g * l * 2
t_fin = 10



phi0 = 0.2
thetta0 = -3.1428/3
dphi0 = 0
dthetta0 = 0

t = sp.Symbol('t')

phi =  sp.Function('phi')(t) 
thetta = 0;
omega_phi = sp.Function('omega_phi')(t) 
omega_thetta = 0;

#1 defining the kinetic energy
K = m_oa * (omega_phi * l/2) * (omega_phi * l/2) / 2 + m_oa * 1/3 * l/2 * l/2 * omega_phi * omega_phi / 2
#2 defining the potential energy
Pmg = m_oa * g * l * (sp.cos(phi)) / 2
P2 = c * phi * phi / 2
P = Pmg + P2

Q1 = - k1 * omega_phi 
#Lagrange function
L = K - P

#equations
ur1 = sp.diff(sp.diff(L,omega_phi),t)-sp.diff(L,phi) - Q1

a11 = ur1.coeff(sp.diff(omega_phi, t), 1)
b1 = -ur1.coeff(sp.diff(omega_phi, t), 0).subs(sp.diff(phi,t), omega_phi)

print(b1)

domega_phidt = b1/a11

Time = np.linspace(0, t_fin, Steps)
fomega_phi = sp.lambdify([phi, omega_phi], domega_phidt, "numpy")

y0 = [phi0, dphi0]

sol = odeint(form2, y0, Time, args =(fomega_phi,))

Phi = sol[:, 0]
Omega_phi = sol[:, 1]


T = 2 * 3.14 * sp.sqrt(m*l*l/(3*(c-m*g*l)))

X_A = sp.lambdify(phi, l * sp.cos(phi + 3.14/2))
Y_A = sp.lambdify(phi, l * sp.sin(phi + 3.14/2))

X_B = sp.lambdify(phi, l * sp.cos(phi + 3.14/2))
Y_B = sp.lambdify(phi, l * sp.sin(phi + 3.14/2) - l )

XA = X_A(sol[:, 0])
YA = Y_A(sol[:, 0])

XB = X_B(sol[:, 0])
YB = Y_B(sol[:, 0])


Nv= 3
R1 = 0.001
R2 = l / 12


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-l * 2 - R_Circle - l/10, l *2 + R_Circle + l/10], ylim=[-l*2 - R_Circle - l/10,l*2 + R_Circle + l/10])


alpha = np.linspace(0, Nv*6.283+Phi[0]+ 3.14/2, 100)
X_SpiralSpr = -(R1 + alpha * (R2 - R1) / alpha[-1]) * np.sin(alpha)
Y_SpiralSpr = (R1 + alpha * (R2 - R1) / alpha[-1]) * np.cos(alpha)

beta = np.linspace(0, 2*math.pi, 100)

X_Circle = R_Circle * np.cos(beta)
Y_Circle = R_Circle * np.sin(beta)

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(1, 2, 1)
ax_for_graphs.plot(Time, Phi, color='blue')
ax_for_graphs.set_title("Phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(1, 2, 2)
ax_for_graphs.plot(Time, Omega_phi, color='red')
ax_for_graphs.set_title('Phi\'(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)


ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

OX = ax.plot([-l * 2, l*2], [-l/10,-l/10], 'black', linestyle = '--')
Draw_Spring = ax.plot(X_SpiralSpr, Y_SpiralSpr, color='#666666')[0]
Draw_OA=ax.plot([0, XA[0]], [0, YA[0]], color='#808080')[0]
Draw_AB=ax.plot([XA[0], XB[0]], [YA[0], YB[0]], color='#808080' )[0]
PointB = ax.plot(XB[0], YB[0])[0]
PointA = ax.plot(XA[0], YA[0], color='#a0a0a0', marker='o')[0]
Draw_Circle = ax.plot(X_Circle + XB[0], Y_Circle + YB[0], color='black', linewidth=1)[0]
triangle = pat.Polygon([(0,0), (-l/10, -l/10), (l/10, -l/10)], color='#d3d3d3')
ax.add_patch(triangle)

def update(i):
    PointB.set_data(XB[i],YB[i])
    Draw_OA.set_data([0, XA[i]], [0, YA[i]])
    PointA.set_data(XA[i], YA[i])
    Draw_AB.set_data([XA[i], XB[i]], [YA[i], YB[i]])
    Draw_Circle.set_data(X_Circle + XB[i], Y_Circle + YB[i])
    alpha = np.linspace(0, Nv*6.28+Phi[i] + 3.14/2, 100)
    X_SpiralSpr = -(R1 + alpha * (R2 - R1) / alpha[-1]) * np.sin(alpha - 1.57)
    Y_SpiralSpr = (R1 + alpha * (R2 - R1) / alpha[-1]) * np.cos(alpha - 1.57)
    Draw_Spring.set_data(X_SpiralSpr, Y_SpiralSpr)
    return [PointB, Draw_OA, Draw_Spring, Draw_AB, PointA, Draw_Circle]

anima = FuncAnimation(fig, update, frames=Steps, interval=1)
PERIOD = 2 * 3.14 * sp.sqrt(m_oa*l*l/(3*(c-(m_oa*g*l/2))))
print("PERIOD = ", PERIOD)
plt.show()