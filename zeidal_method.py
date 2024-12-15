import numpy as np

def has_diagonal_dominance(matrix_a: np.ndarray) -> bool:
    """
    Проверяет, обладает ли матрица строгим диагональным преобладанием.
    """
    n = matrix_a.shape[0]
    for i in range(n):
        # Сумма абсолютных значений элементов строки, кроме диагонального
        row_sum = sum(abs(matrix_a[i][j]) for j in range(n) if j != i)

        # Проверяем, что диагональный элемент строго больше суммы остальных
        if abs(matrix_a[i][i]) <= row_sum:
            return False
    return True

def reorder_matrix_for_diagonal_dominance(matrix_a: np.ndarray, vector_b: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Пытается переставить строки матрицы и вектора, чтобы достичь диагонального преобладания.
    """
    matrix_a = matrix_a.copy()  # создаем копию
    vector_b = vector_b.copy()  # создаем копию
    n = len(matrix_a)
    for i in range(n):
        # Находим строку с максимальным диагональным элементом
        max_row = max(range(i, n), key=lambda r: abs(matrix_a[r, i]))
        if i != max_row:
            # Меняем строки местами
            matrix_a[[i, max_row]] = matrix_a[[max_row, i]]
            vector_b[[i, max_row]] = vector_b[[max_row, i]]
    return matrix_a, vector_b

def zeydal_method(
    matrix_a: np.ndarray,
    vector_b: np.ndarray,
    x0: np.ndarray,
    tol: float = 0.001,
    max_iterations: int = 1000
) -> np.ndarray:
    """
    Решение СЛАУ методом Зейделя с проверкой диагонального преобладания.
    """
    matrix_a, vector_b = reorder_matrix_for_diagonal_dominance(matrix_a, vector_b)

    # Проверяем диагональное преобладание после перестановки строк
    if not has_diagonal_dominance(matrix_a):
        print("Предупреждение: матрица не обладает диагональным преобладанием.")

    n = len(matrix_a)
    x = np.copy(x0)

    for iteration in range(max_iterations):
        x_new = np.copy(x)

        for i in range(n):
            sum1 = sum(matrix_a[i][j] * x_new[j] for j in range(i))
            sum2 = sum(matrix_a[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (vector_b[i] - sum1 - sum2) / matrix_a[i][i]

        # Невязка: r = A * x_new - b
        residual = np.dot(matrix_a, x_new) - vector_b

        # Норма невязки (норма 2)
        residual_norm = np.linalg.norm(residual)

        # Выводим информацию об итерации
        print(f"Итерация {iteration + 1}:")
        print(f"   Вектор невязки: {residual}")
        print(f"   Норма невязки: {residual_norm}")

        # Проверка на сходимость
        if residual_norm < tol:
            print(f"Метод Зейделя завершился на шаге {iteration + 1} с невязкой: {residual_norm}")
            return x_new, residual, residual_norm

        x = x_new

    print("Метод Зейделя не сходится за заданное количество итераций.")
    return x, residual, residual_norm  # Возвращаем последнее приближение

