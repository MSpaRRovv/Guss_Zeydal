import numpy as np

from input_matrix import input_matrix
from gauss_method import gauss_method
from zeidal_method import zeydal_method

# Ввод матрицы и вектора
A, b = input_matrix()

# Метод Гаусса
print("\nМетод Гаусса:")
gauss_result, gauss_residual, gauss_residual_norm = gauss_method(A.copy(), b.copy())


# Метод Зейделя
print("\nМетод Зейделя:")
x0 = np.zeros(len(b))
zeidel_result, zeidel_residual, zeidel_residual_norm = zeydal_method(A.copy(), b.copy(), x0)

# Выводим решение и невязку для метода Зейделя
print("Решение методом Зейделя:", zeidel_result)
print("Вектор невязки для метода Зейделя:", zeidel_residual)
print(f"Норма невязки для метода Зейделя: {zeidel_residual_norm}")
print("Решение методом Гаусса:", gauss_result)
#print("Вектор невязки для метода Гаусса:", gauss_residual)
#print(f"Норма невязки для метода Гаусса: {gauss_residual_norm}")
