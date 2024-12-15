from input_matrix import input_matrix
from gauss_method import gauss_method

A, b  = input_matrix()
print("Решение системы: ", gauss_method(A, b))