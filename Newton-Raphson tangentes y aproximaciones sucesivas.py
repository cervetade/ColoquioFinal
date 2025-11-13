import numpy as np
import matplotlib.pyplot as plt

def f(T):
    return np.exp(-T/10.0) - 0.1*T

def fp(T):
    return -0.1*(np.exp(-T/10.0) + 1.0)

def newton_points(T0, n_steps=4):
    xs = [T0]
    ys = [f(T0)]
    T = T0
    for _ in range(n_steps):
        T_next = T - f(T)/fp(T)
        xs.append(T_next)
        ys.append(f(T_next))
        T = T_next
    return np.array(xs), np.array(ys)

T0 = 5.0
steps = 3
xs, ys = newton_points(T0, steps)

xmin = min(min(xs) - 1.0, 0.0)
xmax = max(max(xs) + 1.0, 10.0)
T_grid = np.linspace(xmin, xmax, 600)

plt.figure(figsize=(9,6))
plt.plot(T_grid, f(T_grid), linewidth=2, label="f(T)")
plt.axhline(0, linestyle='--', linewidth=1)

for k in range(len(xs)-1):
    xk = xs[k]
    yk = f(xk)
    slope = fp(xk)
    x_tan = np.linspace(xmin, xmax, 200)
    y_tan = slope*(x_tan - xk) + yk
    plt.plot(x_tan, y_tan, linewidth=1.5, label=f"Tangente en T{k}")
    plt.scatter([xk], [yk], zorder=5)
    plt.scatter([xs[k+1]], [0], zorder=5)
    plt.annotate(f"T{k}", (xk, yk), xytext=(5,5), textcoords="offset points")
    plt.annotate(f"T{k+1}", (xs[k+1], 0), xytext=(5,5), textcoords="offset points")

plt.title("Newton-Raphson: tangentes y aproximaciones sucesivas (T0=5)")
plt.xlabel("T")
plt.ylabel("f(T)")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("tangentes_newton_equilibrio.png", dpi=200)
plt.show()
