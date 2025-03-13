import numpy as np

# Coeficientes del sistema de ecuaciones (matriz A)
A = np.array([
    [15, -4, -1, -2, 0, 0, 0, 0, 0, 0],
    [-3, 18, -2, 0, -1, 0, 0, 0, 0, 0],
    [-1, -2, 20, 0, 0, -5, 0, 0, 0, 0],
    [-2, -1, -4, 22, 0, 0, -1, 0, 0, 0],
    [0, -1, -3, -1, 25, 0, 0, -2, 0, 0],
    [0, 0, -2, 0, -1, 28, 0, 0, -1, 0],
    [0, 0, 0, -4, 0, -2, 30, 0, 0, -3],
    [0, 0, 0, 0, -1, 0, -1, 35, -2, 0],
    [0, 0, 0, 0, 0, -2, 0, -3, 40, -1],
    [0, 0, 0, 0, 0, 0, -3, 0, -1, 45]
])

# Términos independientes (vector b)
b = np.array([200, 250, 180, 300, 270, 310, 320, 400, 450, 500])

# Inicialización de las soluciones iniciales (por ejemplo, cero)
x = np.zeros_like(b)

# Parámetros de iteración
iterations = 25
tolerance = 1e-6

# Método de Jacobi
for k in range(iterations):
    x_new = np.zeros_like(x)
    
    for i in range(len(b)):
        # La fórmula de Jacobi
        sum_ax = np.dot(A[i, :], x) - A[i, i] * x[i]
        x_new[i] = (b[i] - sum_ax) / A[i, i]

    # Verificar si la solución ha convergido
    if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
        print(f"Solución convergente después de {k+1} iteraciones")
        break

    x = x_new

# Resultado final
print("Solución del sistema:")
print(x)
