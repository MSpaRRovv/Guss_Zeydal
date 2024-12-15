import numpy as np

from input_matrix import input_matrix
from gauss_method import gauss_method
from zeidal_method import zeydal_method

# Ввод матрицы и вектора
A, b = input_matrix()

# Метод Гаусса
print("\nМетод Гаусса:")
gauss_result = gauss_method(A.copy(), b.copy())


# Метод Зейделя
print("\nМетод Зейделя:")
x0 = np.zeros(len(b))
zeidel_result = zeydal_method(A.copy(), b.copy(), x0)

print("Решение методом Гаусса:", gauss_result)
print("Решение методом Зейделя:", zeidel_result)

