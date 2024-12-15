import numpy as np

from input_matrix import input_matrix
from gauss_method import gauss_method
from zeidal_method import zeydal_method

A, b  = input_matrix()

# Метод Гаусса
print("\nМетод Гаусса:")
gauss_result = gauss_method(A.copy(), b.copy())
print("Решение:", gauss_result)

# Невязка для метода Гаусса
#residual_gauss = np.dot(A, gauss_result) - b
#print(f"Норма невязки для метода Гаусса: {np.linalg.norm(residual_gauss)}")

# Метод Зейделя
print("\nМетод Зейделя:")
x0 = np.zeros(len(A))
zeidel_result = zeydal_method(A.copy(), b.copy(), x0)

# Невязка для метода Зейделя
#residual_zeidel = np.dot(A, zeidel_result) - b
#print(f"Норма невязки для метода Зейделя: {np.linalg.norm(residual_zeidel)}")

# Выводим решение
print("Решение методом Зейделя:", zeidel_result)
