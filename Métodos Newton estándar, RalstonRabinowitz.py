# -*- coding: utf-8 -*-
# Raíces múltiples en f(x) = (x+1)(x-1)^2
# Métodos: Newton estándar, Ralston–Rabinowitz con m, y con u(x)=f/f'

import numpy as np

# --------- Definición del problema ---------
def f(x):
    # f(x) = (x+1)(x-1)^2 = x^3 - x^2 - x + 1
    return (x + 1.0) * (x - 1.0)**2

def fp(x):
    # f'(x) = 3x^2 - 2x - 1
    return 3.0*x**2 - 2.0*x - 1.0

def fpp(x):
    # f''(x) = 6x - 2
    return 6.0*x - 2.0

# --------- Utilitarios ---------
def print_header(title):
    print("\n" + title)
    print("{:>3s} | {:>20s} | {:>20s} | {:>20s}".format("k","x_k","f(x_k)","|Δx|"))
    print("-"*72)

def print_row(k, xk, fxk, dx):
    print("{:3d} | {:20.12f} | {:20.12e} | {:20.12e}".format(k, xk, fxk, dx))

# --------- 1) Newton-Raphson estándar ---------
def newton_estandar(x0, tol=1e-12, max_iter=50):
    title = "Newton-Raphson estándar"
    x = float(x0)
    print_header(title)
    print_row(0, x, f(x), np.nan)
    for k in range(1, max_iter+1):
        fx, fpx = f(x), fp(x)
        if abs(fpx) < 1e-16:
            print("Derivada casi nula: posible falla numérica.")
            break
        x_new = x - fx/fpx
        dx = abs(x_new - x)
        x = x_new
        print_row(k, x, f(x), dx)
        if dx < tol or abs(f(x)) < tol:
            print("Convergencia alcanzada en {} iteraciones.".format(k))
            break
    return x

# --------- 2) Ralston–Rabinowitz con multiplicidad m ---------
def newton_multiplicidad(x0, m, tol=1e-12, max_iter=50):
    # x_{k+1} = x_k - m * f(x_k)/f'(x_k)
    title = "Ralston–Rabinowitz (multiplicidad m={})".format(m)
    x = float(x0)
    print_header(title)
    print_row(0, x, f(x), np.nan)
    for k in range(1, max_iter+1):
        fx, fpx = f(x), fp(x)
        if abs(fpx) < 1e-16:
            print("Derivada casi nula: posible falla numérica.")
            break
        x_new = x - m * fx / fpx
        dx = abs(x_new - x)
        x = x_new
        print_row(k, x, f(x), dx)
        if dx < tol or abs(f(x)) < tol:
            print("Convergencia alcanzada en {} iteraciones.".format(k))
            break
    return x

# --------- 3) Ralston–Rabinowitz con u(x)=f/f' (no necesita m) ---------
def newton_u(x0, tol=1e-12, max_iter=50):
    # x_{k+1} = x_k - f f' / ( (f')^2 - f f'' )
    title = "Ralston–Rabinowitz con u(x)=f/f' (sin m)"
    x = float(x0)
    print_header(title)
    print_row(0, x, f(x), np.nan)
    for k in range(1, max_iter+1):
        fx, fpx, fppx = f(x), fp(x), fpp(x)
        denom = (fpx*fpx - fx*fppx)
        if abs(denom) < 1e-16:
            print("Denominador casi nulo: posible falla numérica.")
            break
        x_new = x - (fx * fpx) / denom
        dx = abs(x_new - x)
        x = x_new
        print_row(k, x, f(x), dx)
        if dx < tol or abs(f(x)) < tol:
            print("Convergencia alcanzada en {} iteraciones.".format(k))
            break
    return x

# --------- Ejecuciones de ejemplo ---------
if __name__ == "__main__":
    # Raíz múltiple (doble) en x=1. Probá cerca de 1 para ver la diferencia de convergencia.
    x0_cerca_1 = 1.2
    x0_otro_lado = 0.8

    # Newton estándar (con raíz múltiple converge linealmente y más lento)
    rN1 = newton_estandar(x0_cerca_1)
    rN2 = newton_estandar(x0_otro_lado)

    # Ralston–Rabinowitz con m (acá m=2 para la raíz en x=1)
    rM1 = newton_multiplicidad(x0_cerca_1, m=2)
    rM2 = newton_multiplicidad(x0_otro_lado, m=2)

    # Ralston–Rabinowitz con u(x)=f/f' (no requiere conocer m)
    rU1 = newton_u(x0_cerca_1)
    rU2 = newton_u(x0_otro_lado)

    print("\nResumen:")
    print("  Newton estándar desde 1.2  -> x* ≈ {:.12f}".format(rN1))
    print("  Newton estándar desde 0.8  -> x* ≈ {:.12f}".format(rN2))
    print("  RR con m=2 desde 1.2       -> x* ≈ {:.12f}".format(rM1))
    print("  RR con m=2 desde 0.8       -> x* ≈ {:.12f}".format(rM2))
    print("  RR con u(x) desde 1.2      -> x* ≈ {:.12f}".format(rU1))
    print("  RR con u(x) desde 0.8      -> x* ≈ {:.12f}".format(rU2))

    # También podés probar la raíz simple en x=-1 (semilla por ejemplo -0.5 o -2):
    print("\nPrueba sobre raíz simple x=-1 (Newton estándar desde x0=-2):")
    r_simple = newton_estandar(-2.0)
