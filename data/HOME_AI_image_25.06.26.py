# Курс: AI+Python
# Модуль 12. Структури даних
# Тема: Стеки. Частина 2
import cv2
import numpy as np

# Завдання 1
print("---З А В Д А Н Н Я  1---")

# Відкрийте зображення data/Lenna.png.
image_lenna = cv2.imread(
                "../data/lesson1/Lenna.png",    # вказуємо шлях до файлу
                cv2.IMREAD_GRAYSCALE    # прапорець, як читати зображення
                )
cv2.imshow("Виводимо зображення: 'Lenna'", image_lenna)
print("Тип даних зображення 'Lenna':", image_lenna.dtype)
print("Розмір зображення 'Lenna':", image_lenna.shape)

# Прочитайте маски data/mask1.png та data/mask2.png.
image_mask_1 = cv2.imread(
                "../data/lesson1/mask1.png",    # вказуємо шлях до файлу
                cv2.IMREAD_GRAYSCALE    # прапорець, як читати зображення
                )
image_mask_2 = cv2.imread(
                "../data/lesson1/mask2.png",    # вказуємо шлях до файлу
                cv2.IMREAD_GRAYSCALE    # прапорець, як читати зображення
                )

cv2.imshow("Виводимо зображення 'Mask1':", image_mask_1)
print("Тип даних зображення 'Mask1':", image_mask_1.dtype)
print("Розмір зображення 'Mask1':", image_mask_1.shape)

cv2.imshow("Виводимо зображення 'Mask2'\n", image_mask_2)
print("Тип даних зображення 'Mask2':", image_mask_2.dtype)
print("Розмір зображення 'Mask2':", image_mask_2.shape)

# Об’єднайте дві маски в одну, скористайтесь cv2.bitwise_or() та виведіть результат
mask_combination = cv2.bitwise_or(image_mask_1, image_mask_2)
cv2.imshow("\nОб'єднання масок: ", mask_combination)

# Виведіть ту частину зображення, яка відповідає:
#  mask1
#  mask2
#  mask1 і mask2

# Усі пікселі які не відповідають маскам замінити на 0, перед застосуванням змініть тип даних у масці на bool
mask_bool = mask_combination.astype(bool)

result_image_lenna = image_lenna.copy()

result_image_lenna[~mask_bool] = 30

cv2.imshow("Виводимо результат об'єднання та картинки в ньому", result_image_lenna)

# Завдання 2
print("--- З А В Д А Н Н Я   2 ---")
# Домашнє завдання
# Виведіть зображення. Підберіть самостійно межі
image_baboon = cv2.imread(
                "../data/lesson1/baboo.jpg",    # вказуємо шлях до файлу
                cv2.IMREAD_GRAYSCALE    # прапорець, як читати зображення
                )

copy_image_baboon = image_baboon.copy()

cv2.imshow("Виводимо зображення: 'Baboon'", image_baboon)
print("Тип даних зображення 'Baboon':", image_baboon.dtype)
print("Розмір зображення 'Baboon':", image_baboon.shape)

image_baboon[:, :60] = 255
image_baboon[:, 192:] = 255
image_baboon[:10, : ] = 255
image_baboon[50:, :] = 255
cv2.imshow("Виводимо очі: 'Baboon'", image_baboon)

print("ДОДАТКОВО, за власною ініціативою використовуючи копію картинки 'copy_image_baboon' малюємо окуляри")

copy_image_baboon[5:55, 55:58,] = 0     # ЛІВА лінія лівої сторони
copy_image_baboon[5:55, 115:118] = 0    # ПРАВА лінія лівої сторони
copy_image_baboon[5:8, 55:115] = 0      # ВЕРХНЯ лінія лівої сторони
copy_image_baboon[52:55, 55:115] = 0    # НИЖНЯ лінія лівої сторони

copy_image_baboon[5:55, 135:138] = 0    # ЛІВА лінія ПРАВОЇ сторони
copy_image_baboon[5:55, 198:201] = 0    # ПРАВА лінія ПРАВОЇ сторони
copy_image_baboon[5:8, 138:198] = 0     # ВЕРХНЯ лінія ПРАВОЇ сторони
copy_image_baboon[52:55, 138:198] = 0   # НИЖНЯ лінія ПРАВОЇ сторони

copy_image_baboon[27:30, 115:135] = 0   # перегородка-з'єднання лівої та правої сторони


cv2.imshow("'Baboon glasses'", copy_image_baboon)


cv2.waitKey(0)



