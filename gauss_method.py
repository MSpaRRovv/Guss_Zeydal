import numpy as np

def gauss_method(matrix_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
    """
    Решение СЛАУ методом Гаусса с проверкой на совместность системы и на бесконечное количество решений.

    Аргументы:
        matrix_a (np.ndarray):  Матрица коэффициентов
        vector_b (np.ndarray): Вектор правых частей

    Возвращает:
        np.ndarray или str: Вектор решений или текст с сообщением о проблемах с системой
    """
    n = len(vector_b)

    # Расширенная матрица [A|b]
    augmented_matrix = np.hstack((matrix_a, vector_b.reshape(-1, 1)))

    # Прямой ход (приведение к верхнетреугольному виду)
    for i in range(n):
        # Поиск максимального элемента в столбце для устойчивости
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        if augmented_matrix[max_row, i] == 0:
    # Проверка на вырождение и совместность
            if np.all(np.isclose(augmented_matrix[max_row, :-1], 0)) and not np.isclose(augmented_matrix[max_row, -1], 0):
                return "Система не имеет решений"
            else:
                return "Система имеет бесконечное множество решений"

        
        # Перестановка строк для улучшения устойчивости
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Приведение матрицы к верхнетреугольному виду
        for j in range(i + 1, n):
            ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] = augmented_matrix[j, i:] - ratio * augmented_matrix[i, i:]

    # Проверка на совместность системы
    for i in range(n):
        # Если строка выглядит как [0, 0, ..., 0 | c != 0], то система несовместна
        if np.all(np.isclose(augmented_matrix[i, :-1], 0)) and not np.isclose(augmented_matrix[i, -1], 0):
            return "Система не имеет решений"

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = 0
        # Ручной расчет суммы произведений для обратного хода
        for j in range(i + 1, n):
            sum_ax += augmented_matrix[i, j] * x[j]
        
        # Проверка на размерности срезов перед расчетом
        if augmented_matrix[i, i] == 0:
            return "Система не имеет решений"
        x[i] = (augmented_matrix[i, -1] - sum_ax) / augmented_matrix[i, i]

    return x

