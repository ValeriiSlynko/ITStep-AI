# Курс: AI+Python
# Модуль 2. Комп’ютерний зір
# Тема: Масиви. Частина 1

# --- ЗАВДАННЯ 1 ---
# Створіть масив з числами від 1 до 10. Виведіть його, його
# розмір, тип даних.
import numpy as np

nums = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(nums)
print(nums.shape)
print(nums.dtype)

# Змініть розмір масиву на (5, 2). Знову виведіть масив,
# розмір та тип даних

new_nums = nums.reshape(5, 2)
print(new_nums)
print(new_nums.shape)
print(new_nums.dtype)

# --- ЗАВДАННЯ 2 ---
# Створіть масив:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12

nums_1 = np.arange(1, 13)
new_nums_1 = nums_1.reshape(3, 4)
print(new_nums_1)
print(new_nums_1.shape)
print(new_nums_1.dtype)

print(new_nums_1[1, 2])
print(new_nums_1[1, :])

# Використовуючи індекси виведіть:
nums = np.array(
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
    ]
)

print(nums)
# ● число 7
print(nums[1, 2])

# ● другий рядок
print(nums[1])

# ● останній стовпчик
print(nums[:, -1])

# ● праву половину
print(nums[:, 2:4])

# ● жовту область
print(nums[1:3, 1:3])

# ● замініть жовту область на -1
nums[1:3, 1:3] = -1

# ● зробіть перший стовпчик таким самим як і другий
# print(nums[:, 0])
# print(nums[:, 1:3])
# print(nums)


# --- ЗАВДАННЯ 3 ---
print(nums)
# У масиві з попереднього завдання створіть маску для чисел які більші за 6.
# З її допомогою
# ● виведіть кількість чисел більших за 6
mask = nums > 6
print(mask.sum())

# ● виведіть самі числа
# nums[mask]
print(nums[mask])

# ● до кожного числа яке відповідає масці додайте 10
nums[mask] += 10
print(nums)

# ● кожне число, що не відповідає масці помножте на -1
new_mask = ~mask
print(new_mask)
nums[new_mask] *= -1
print(nums)

# ● замініть ці числа які відповідають масці на відповідні їм з масиву
new_nums2 = np.array(
    [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0]
    ]
)
nums[new_mask] = new_nums2[new_mask]
print(nums)


# --- ЗАВДАННЯ 4 ---
# Створіть масив
# -10 24 35
# 250 -6 7
# 12 180 11
# -2 -45 -26
array_num = np.array(
    [
    [-10,24, 35],
    [250, -6, 7],
    [12, 180, 11],
    [-2, -45, -26]
    ]
)
array_num[array_num < 0] = 0
print("МД (3 на 4) в якому числа < '0' замінюємо на '0':\n", array_num)
print("Масив даних (3ст. на 4ряд.):\n", array_num)

# Усі числа менші за 0 замініть на 0.
array_num[array_num < 0] = 0
print("МД (3ст. на 4ряд.) в якому числа < '0' замінюємо на '0':\n", array_num)

# Усі числа більші за 100 замініть на 100
array_num[array_num > 100] = 100
print("МД (3ст. на 4ряд.) в якому числа > '100' замінюємо на '100':\n", array_num)

# --- ЗАВДАННЯ 5 ---
# Створіть масив та виведіть його тип даних
# 100 120 200 250 10
numbers = np.array([100, 120, 200, 250, 10])
print("Створено масив даних: ", numbers)

# Додайте до кожного числа 50 та виведіть результат.
numbers += 50
print("До масиву даних додано 50: ", numbers)

# Створіть такий самий масив але з типом uint8
numbers = numbers.astype(np.uint8)

# Знову додайте 50 та виведіть результат
numbers += 50
print("До масиву даних з типом 'uint8' додаємо 50: ", numbers)

# Зробіть так щоб обчислення працювали правильно, якщо число виходить більшим за 255, то зробіть його 255
numbers = numbers.astype(int) + 100
numbers[numbers > 255] = 255
numbers = numbers.astype(np.uint8)
print("Залишаємо числа 255, якщо воно більше за 255: ", numbers)

# ЗАВДАННЯ 6
# Створіть масив типу uint8
# 10 4 25 40 200
integer = np.array([10, 4, 23, 40, 200])

integer = integer.astype(np.uint8)
integer = integer.astype(np.int64)

# Помножте всі значення на 2.
integer *= 2

# Результат має бути типу uint8, а всі значення в діапазоні 0-255
mask = integer > 255
integer[mask] = 255
print(integer)

print(integer)
print(integer.shape)
print(integer.dtype)

integer = integer.astype(np.uint8)
print(integer)
print(integer.shape)
print(integer.dtype)



# Помножте всі значення на 1.5.
integer = integer * 1.5
print(integer)
print(integer.shape)
print(integer.dtype)

# Результат має бути типу uint8 а всі значення в діапазоні 0-255

mask = integer > 255
integer[mask] = 255

integer = integer