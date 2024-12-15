import numpy as np
import json

def gauss_method(matrix_a: np.ndarray, vector_b: np.ndarray, output_file: str = "gauss_steps.txt"):
    """
    Решение СЛАУ методом Гаусса с записью всех шагов в текстовый файл.
    """
    n = len(vector_b)
    augmented_matrix = np.hstack((matrix_a, vector_b.reshape(-1, 1)))
    
    # Прямой ход
    with open(output_file, "w") as f:  # Открываем файл для записи
        for i in range(n):
            max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
            if np.isclose(augmented_matrix[max_row, i], 0):
                raise ValueError("Система не имеет единственного решения")

            # Записываем текущую матрицу до перестановки
            f.write(f"Перед перестановкой строк {i+1}:\n")
            f.write(format_matrix(augmented_matrix))
            f.write("\n")

            # Перестановка строк
            augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

            # Записываем текущую матрицу после перестановки
            f.write(f"После перестановки строк {i+1}:\n")
            f.write(format_matrix(augmented_matrix))
            f.write("\n")

            for j in range(i + 1, n):
                ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j, i:] -= ratio * augmented_matrix[i, i:]

            # Записываем матрицу после приведения к верхнетреугольному виду
            f.write(f"После приведения к верхнетреугольному виду {i+1}:\n")
            f.write(format_matrix(augmented_matrix))
            f.write("\n")

        # Обратный ход
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])) / augmented_matrix[i, i]

        # Добавляем итоговые результаты
        f.write("Итоговые результаты:\n")
        f.write(f"Решение: {x}\n")

    return x

def format_matrix(matrix: np.ndarray) -> str:
    """
    Форматирует матрицу в строку с выравниванием столбцов и разделением строк.
    """
    formatted_matrix = []
    for row in matrix:
        # Преобразуем числа в строку с нужной точностью
        formatted_matrix.append("  ".join(f"{item:8.3f}" for item in row))  # Форматируем числа
    return "\n".join(formatted_matrix)