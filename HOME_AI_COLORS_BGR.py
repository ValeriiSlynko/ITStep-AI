# Курс: AI+Python
# Модуль 12. Структури даних
# Тема: Стеки. Частина 2

# Завдання 1
# Відкрийте зображення data\lesson2\darken.png.
# Проведіть з ним наступні операції, переведіть його в HSV формат та
# обробіть канал Value наступними способами:

#  застосуйте вирівнювання гістограм
#  збільшіть значення десь на 20-50%,
# оскільки тут результат буде типу float32 та явно вийде за межі [0-255]
# застосуйте np.clip(value, 0, 255) та value.astype(np.uint8)

# Виведіть результати обох обробок на екран

import cv2
import numpy as np
from numpy.ma.core import shape

darken_image = cv2.imread('data/lesson2/darken.png')
cv2.imshow('Darken', darken_image)

# Змінюємо розмір зображення (для кращої візуалізації всіх подальших картинок)
darken_image = cv2.resize(darken_image, (500, 500))
print(darken_image.shape)
print(darken_image.dtype)

# Вивід зображення, його тип даних та розміру
cv2.imshow('Darken', darken_image)
print("Check to resize the image: ", darken_image.shape)
print("Check the data type:", darken_image.dtype)


# Переводимо зображення в HSV рядок
hsv = cv2.cvtColor(darken_image, cv2.COLOR_BGR2HSV)

# Проводимо роботу з каналом "Value"
h, s, value = cv2.split(hsv)

# ОБРОБКА 1. ЗАСТОСОВУЄМО ВИРІВНЮВАННЯ ГІСТОГРАМИ
# Використовуємо функцію 'equalizeHist' яка автоматично вирівнює яскравість
darken_image = cv2.resize(darken_image, (500, 500))
value_method_1 = cv2.equalizeHist(value)

# збираємо канали для обробки ГІСТОГРАМИ
new_hsv_method_1 = cv2.merge((h, s, value_method_1))

# Переводимо назад у кольоровий формат 'BGR' для показу
result_method_1 = cv2.cvtColor(new_hsv_method_1, cv2.COLOR_HSV2BGR)


# ОБРОБКА 2. ЗБІЛЬШЕННЯ ЗНАЧЕННЯ на 20-50% із застосуванням np.clip(value, 0, 255) та value.astype(np.uint8)
# Переводимо у float32, щоб при множенні значення не зламались на позначці 255
value_float = value.astype(np.float32)

# Збільшуємо значення яскравості на 50% (тобто кеф 1.5)
value_float = value_float * 1.5

# Застосовуємо 'np.clip', щоб повернути значення, які вийшли за межі 0:255
value_clip = np.clip(value_float, 0, 255)

# Повертаємо початковий тип даних застосовуючи 'value.astype(np.uint8)'
value_method_2 = value_clip.astype(np.uint8)

# Збираємо канали для ОБРОБКИ 2 збільшеного значення в межах 20-50% (обрано 50% = кеф. 1.5)
new_hsv_method_2 = cv2.merge((h, s, value_method_2))

# Переводимо назад у кольоровий формат 'BGR' для показу
result_method_2 = cv2.cvtColor(new_hsv_method_2, cv2.COLOR_HSV2BGR)


# ВИВОДИМО РЕЗУЛЬТАТ ОБРОБКИ 2-х МЕТОДІВ НА ЕКРАН
# Результат обробки картинки методом equalizeHist
cv2.imshow("Method 1 'equalizeHist: ", result_method_1)

# Результат обробки картинки методом equalizeHist
cv2.imshow("Method 2 'np.clip' method: ", result_method_2)


cv2.waitKey(0)