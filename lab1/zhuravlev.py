# импорт необходимых библиотек
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

# матрица поворота
def Rod2D(X, Y, Alpha):
    RX = X * np.cos(Alpha) - Y * np.sin(Alpha)
    RY = X * np.sin(Alpha) + Y * np.cos(Alpha)
    return RX, RY

#рисование векторов в момент времени i
def Animation(i): 
    Point.set_data(X[i], Y[i])
    Vline.set_data([X[i], X[i] + VX[i]/4], [Y[i], Y[i] + VY[i]/4])
    VVecArrowX, VVecArrowY = Rod2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VVecArrow.set_data(VVecArrowX + X[i] + VX[i]/4, VVecArrowY + Y[i] + VY[i]/4)
    Rline.set_data([0, X[i]], [0, Y[i]])
    RVecArrowX, RVecArrowY = Rod2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RVecArrow.set_data(RVecArrowX+X[i], RVecArrowY+Y[i])
    Aline.set_data([X[i], X[i] + AX[i]/50], [Y[i], Y[i] + AY[i]/50])
    AVecArrowX, AVecArrowY = Rod2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    AVecArrow.set_data(AVecArrowX+X[i]+AX[i]/50, AVecArrowY+Y[i]+AY[i]/50)
    CurveVec.set_data([X[i], X[i] + (Y[i] + VY[i]) * CurveV[i] / sp.sqrt((Y[i] + VY[i]) ** 2 +
    (X[i] + VX[i]) ** 2)], [Y[i], Y[i] - (X[i] + VX[i]) * CurveV[i] /
    sp.sqrt((Y[i] + VY[i]) ** 2+(X[i] + VX[i]) ** 2)])
    return Point, Vline, VVecArrow, Rline, RVecArrow, Aline, AVecArrow, CurveVec

# Задача начальных параметров , вычисление необходимых характеристик точки
t = sp.Symbol('t')
r = 2+sp.cos(6*t)
phi = t+1.2*sp.cos(6*t)
x = r*sp.cos(phi)
y = r*sp.sin(phi)
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
Ax = sp.diff(Vx, t)
Ay = sp.diff(Vy, t)
V = sp.sqrt(Vx**2+Vy**2)
A = sp.sqrt(Ax**2+Ay**2)
Atan = sp.diff(V, t)
Curve = V**2/sp.sqrt(A**2-Atan**2)

# Создание массивов времен
T = np.linspace(0, 10, 1000)
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)
A = np.zeros_like(T)
AT = np.zeros_like(T)
CurveV = np.zeros_like(T)

#заполнение массивов значениями из интервала T
for i in np.arange(len(T)): 
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])
    CurveV[i] = sp.Subs(Curve, t, T[i])

# Отрисовка поля, начальных положений векторов
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-10, 10], ylim=[-10, 10])
ax1.plot(X, Y)
Point, = ax1.plot(X[0], Y[0], marker='o', color='black')
Vline, = ax1.plot([X[0], X[0]+VX[0]],[Y[0], Y[0]+VY[0]], color='red')
ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])
VVecArrowX, VVecArrowY = Rod2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VVecArrow, = ax1.plot(VVecArrowX+X[0]+VX[0], VVecArrowY+Y[0]+VY[0], 'red')
Rline, = ax1.plot([0, X[0]], [0, Y[0]], 'green')
RVecArrowX, RVecArrowY = Rod2D(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
RVecArrow, = ax1.plot(RVecArrowX+VX[0]+X[0], RVecArrowY+VY[0]+Y[0], 'green')
Aline, = ax1.plot([X[0],X[0]+AX[0]],[Y[0], Y[0]+AY[0]], 'blue')
AVecArrowX, AVecArrowY = Rod2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))
AVecArrow, = ax1.plot(AVecArrowX+X[0], AVecArrowY+Y[0], 'blue')
CurveVec, = ax1.plot([X[0], X[0]+(Y[0]+VY[0])*CurveV[0]/sp.sqrt((Y[0]+VY[0])**2+
    (X[0]+VX[0])**2)], [Y[0], Y[0]-(X[0]+VX[0])*CurveV[0]/sp.sqrt((Y[0]+VY[0])**2+
    (X[0]+VX[0])**2)], 'orange')

# Вывод на экран
anim = FuncAnimation(fig, Animation, frames=1000, interval=20, blit=True, repeat=False)
plt.show()