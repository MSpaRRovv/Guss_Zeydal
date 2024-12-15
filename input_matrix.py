import numpy as np

def input_matrix():
    n = int(input("Введите размер матрицы (n): "))
    print("Введите матрицу A построчно, разделяя элементы пробелами:")
    A = np.array([list(map(float, input().split())) for _ in range(n)])
    print("Введите вектор b (элементы через пробел):")
    b = np.array(list(map(float, input().split())))
    return A, b
