# Курс: AI+Python
# Модуль 2. Комп’ютерний зір
# Тема: масиви. Частина 1
import numpy as np

# --- ЗАВДАННЯ 1 ---
print("\n--- ЗАВДАННЯ 1 ---")
# Створіть масив:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 13 14 15 16
array_num = np.array(
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
    ]
)
print("Створюємо масив даних (4 на 4):\n", array_num)
print("МД (4 на 4) має розмір: ", array_num.shape)
print("Тип МД: ", array_num.dtype)

# Використовуючи індекси виведіть:
# ● число 14
print("\nВиводимо число: ", array_num[3,1])

# ● третій рядок
print("Виводимо 3-й рядок: ", array_num[2])

# ● перший стовпчик
print("Виводимо 1-й стовпчик: ", array_num[:,0])

# ● верхню половину
print("Виводимо верхню половину:\n", array_num[0:2])

# ● замініть числа в рядках 2-3 на 100
array_num[:2] = 100
print("Заміна чисел 2-3 рядків на 100:\n", array_num)

# ● зробіть другий рядок таким як останній рядок
print("Отримуємо останній рядок: ",array_num[3])
array_num[1] = array_num[3]
print("Перетворюємо 2-й рядок на останній:\n", array_num)


# --- ЗАВДАННЯ 2 ---
print("\n--- ЗАВДАННЯ 2 ---")
# У масиві з попереднього завдання створіть маску для парних чисел.
array_num2 = np.array(
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
    ]
)

# З її допомогою
# ● виведіть самі числа
mask = array_num2 % 2 == 0
array_data = array_num2[mask].reshape(2,4)
print("Парні числа із МД (4 на 4):\n", array_data)

# ● замініть їх на 100
array_num2[mask] = 100
print("Замінюємо парні числа на 100:\n", array_num2)

# --- ЗАВДАННЯ 3 ---
print("\n--- ЗАВДАННЯ 3 ---")
# Створіть 2 масиви типу uint8:
# Масив 1: 128 200 10
# Масив 2: 250 10 34
array_1 = np.array ([128, 200, 10])
array_2 = np.array ([250, 10, 34])
print("Створюємо два масиви даних:\n", array_1 ,"\nта\n", array_2)

array_1 = array_1.astype(np.uint8)
array_2 = array_2.astype(np.uint8)
print("Кількість даних у кожному масиві: ", array_1.shape, array_2.shape)
print("Тип даних: ", array_1.dtype, array_2.dtype)

# Об’єднайте їх у пропорції 20% першого масив + 80% другого масиву.
# В результаті має бути тип даних uint8 та числа в діапазоні 0-255

# переводимо у 'float16' для обчислень без переповнення

array_1 = array_1.astype(np.float16)
array_2 = array_2.astype(np.float16)

array_20 = array_1 * 0.2
array_80 = array_2 * 0.8

result = array_20 + array_80

mask = result > 255
result[mask] = 255

result = result.astype(np.uint8)

print("\nРезультат виконання: ")
print(result)


