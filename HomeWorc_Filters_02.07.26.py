# Курс: AI+Python
# Модуль 12. Структури даних
# Тема: Стеки. Частина 2
import cv2
import numpy as np

import utils

# ЗАВДАННЯ 1
# Відкрийте зображення data/lesson3/sonet.png.
# Проведіть бінарізацію.
# Обов’язково використайте:
#  розмиття або наведення різкості
#  адаптивну бінарізацію
#  очищеня шумів

# 1. Відкриваємо зображення в сірому форматі:
image_sonnet = cv2.imread("data/lesson3/sonet.png", cv2.IMREAD_GRAYSCALE)

# Перевіряємо чи завантажився файл (захист від помилок)
if image_sonnet is None:
    print("Помилка: Не вдалося завантажити зображення! \nПеревірте шлях до файлу.")
    exit()

cv2.imshow("sonet", image_sonnet)
print("Розмір зображення 'sonet':", image_sonnet.shape)
print("Тип зображення 'sonet': ", image_sonnet.dtype)

# 2. Корегуємо розмиття /ДВОСТОРОННІЙ ФІЛЬТР/
bilat = cv2.bilateralFilter(
    image_sonnet,       # передаємо зображення з шумом
    d=5,                # розмір фільтру
    sigmaColor=15,      # наскільки важливі пікселі іншого кольору
    sigmaSpace=8,      # наскільки важливими є далекі пікселі
)
cv2.imshow("Bilateral Filter", bilat)

# 3. Корегуємо розмиття /GAUSE/
gauss = cv2.GaussianBlur(
    image_sonnet,       # зображення з шумом
    (1, 1),     # розмір фільтру(ядра)
    sigmaX=0,      # наскільки важливими є далекі пікселі, adaptive -> value
)
cv2.imshow('gauss', gauss)

# Адаптивна бінарізація /ADAPTIVE/ робить текст чітким
# Використовуємо метод Гауса
res = cv2.adaptiveThreshold(
    gauss,              # зображення з текстом (чорно-біле)
        255,    # білий колір
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     # фільтр для обрахунку порогу (Гаус)
    cv2.THRESH_BINARY,      # цей рядок просто треба вказаати
    21,     # розмір фільтру
    4,           # наскільки ПІКСЕЛЬ має відрізнятися від порогу
)
cv2.imshow("Adaptive Threshold", res)

# 4. ОЧИЩЕННЯ ШУМІВ (Морфологія) - прибираємо дрібні "крихти" та цятки
kernel = np.ones((1, 1), np.uint8)  # Створюємо маленький квадратик-пензлик
cleaned = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
cv2.imshow("4. Cleaned Noises", cleaned)


cv2.waitKey(0)

cv2.destroyAllWindows()

# ЗАВДАННЯ 2
# Відкрийте зображення data/lesson3/sonnet_noised.png.
# Проведіть бінарізацію.
# Застосуйте код з завдання 1 та  спробуйте покращити результат

# 1. Відкриваємо зображення в сірому форматі:
image_sonnet_noised = cv2.imread("data/lesson3/sonet_noised.png", cv2.IMREAD_GRAYSCALE)

cv2.imshow("Original Sonnet", image_sonnet_noised)
# Перевіряємо чи завантажився файл (захист від помилок)
if image_sonnet_noised is None:
    print("Помилка: Не вдалося завантажити зображення! \nПеревірте шлях до файлу.")
    exit()

# 2. Корегуємо розмиття /ДВОСТОРОННІЙ ФІЛЬТР/
bilat = cv2.bilateralFilter(
    image_sonnet_noised,       # передаємо зображення з шумом
    d=1,                # розмір фільтру
    sigmaColor=7,      # наскільки важливі пікселі іншого кольору
    sigmaSpace=7,      # наскільки важливими є далекі пікселі
)
cv2.imshow("Bilateral Filter", bilat)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(25,9))
result = clahe.apply(image_sonnet_noised)
cv2.imshow('clahe', result)

# 3. Корегуємо розмиття /GAUSE/
gauss = cv2.GaussianBlur(
    image_sonnet_noised,       # зображення з шумом
    (1, 1),     # розмір фільтру(ядра)
    sigmaX=0,      # наскільки важливими є далекі пікселі, adaptive -> value
)
cv2.imshow('gauss', gauss)

# Адаптивна бінарізація /ADAPTIVE/ робить текст чітким
# Використовуємо метод Гауса
res = cv2.adaptiveThreshold(
    gauss,              # зображення з текстом (чорно-біле)
        255,    # білий колір
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,     # фільтр для обрахунку порогу (Гаус)
    cv2.THRESH_BINARY,      # цей рядок просто треба вказаати
    35,     # розмір фільтру
    10,           # наскільки ПІКСЕЛЬ має відрізнятися від порогу
)
cv2.imshow("Adaptive Threshold", res)



cv2.waitKey(0)

cv2.destroyAllWindows()