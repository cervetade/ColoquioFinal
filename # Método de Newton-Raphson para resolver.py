# Método de Newton-Raphson para resolver f(T) = e^(-T/10) - 0.1*T = 0
# Ejemplo: cálculo del equilibrio térmico con T0 = 5
import numpy as np
# Definición de la función y su derivada
def f(T):
    return np.exp(-T/10.0) - 0.1*T
def f_prime(T):
    return -0.1 * (np.exp(-T/10.0) + 1.0)
# Implementación del método de Newton-Raphson
def newton_raphson(T0, tol=1e-6, max_iter=20):
    T = T0
    print(f"{'Iteración':>10} | {'T':>12} | {'f(T)':>12}")
    print("-" * 38)
    for i in range(max_iter):
        f_val = f(T)
        f_der = f_prime(T)
        if abs(f_der) < 1e-10:
            print("Derivada cercana a cero. El método puede fallar.")
            break

        T_next = T - f_val / f_der
        print(f"{i:10d} | {T:12.6f} | {f_val:12.6e}")

        # Criterio de parada
        if abs(T_next - T) < tol:
            print(f"\nConvergencia alcanzada en {i+1} iteraciones.")
            print(f"Raíz aproximada: T* = {T_next:.6f}")
            return T_next
        T = T_next
    print("\nNo se alcanzó la convergencia en el número máximo de iteraciones.")
    return T
# Ejecución del método con T0 = 5
T0 = 5.0
T_star = newton_raphson(T0)


import numpy as np
import matplotlib.pyplot as plt

# Definición de la función
def f(T):
    return np.exp(-T/10.0) - 0.1*T

# Valores de T y f(T)
T_vals = np.linspace(0, 10, 200)
f_vals = f(T_vals)

# Raíz aproximada
T_star = 5.671433

# Crear el gráfico
plt.figure(figsize=(8,5))
plt.plot(T_vals, f_vals, label=r"$f(T)=e^{-T/10}-0.1T$", linewidth=2)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.scatter(T_star, 0, color='red', zorder=5, label=fr"Raíz: T* ≈ {T_star:.3f}")
plt.title("Método de Newton-Raphson – Equilibrio térmico", fontsize=14)
plt.xlabel("Temperatura (T)")
plt.ylabel("f(T)")
plt.legend()
plt.grid(True)
plt.show()