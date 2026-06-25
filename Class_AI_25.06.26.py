# Курс: AI+Python
# Модуль 12. Аналіз даних
# Тема: Стеки. Частина 2
# Завдання 1
# Відкрийте зображення data/Lenna.png.
# Виведіть на екран розмір зображення, тип даних, максимальну та мінімальну
# інтенсивність пікселів, саме зображення з підписом.
import cv2
import numpy as np
from matplotlib.pyplot import imshow

image = cv2.imread(
    "data/lesson1/Lenna.png",   # шлях до файлу
    cv2.IMREAD_GRAYSCALE,       # прапорець, як читати зображення
)

cv2.imshow("Lenna", image)
# cv2.waitKey(0)
print(image.dtype)
print(image.shape)

print(image.max())
print(image.min())


# # Завдання 2
# # Відкрийте зображення data/Lenna.png. Виведіть на екран такі зображень:
# #  Верхній лівий кут розміром 100х50
# segment_1 = image[0:100, 0:50]
# cv2.imshow("segment_1", segment_1)
# cv2.waitKey(0)
#
# #  Центральний квадрат розміром 100х100
# segment_2 = image[78:156, 78:156]
# cv2.imshow("segment_2", segment_2)
# print(segment_2.shape)
# cv2.waitKey(0)
#
# #  Верхню половину
segment_3 = image[:128, :1]
cv2.imshow("segment_2", segment_3)
print(segment_3.shape)
#
# #  Нижню половину
segment_4 = image[129:255, :]
cv2.imshow("segment_4", segment_4)
print(segment_4.shape)
#
# #  Ліву половину
segment_5 = image[:, :128]
cv2.imshow("segment_5", segment_5)
print(segment_3.shape)
#
# #  Праву половину
segment_6 = image[:, 129:255]
cv2.imshow("segment_6", segment_6)
print(segment_6.shape)
#
# # Завдання 3
# # Відкрийте зображення data/Lenna.png.
# # Створіть наступні зображення
image[:20, :] = 0
image[235:255, :] = 255

cv2.imshow("Lenna", image)

image[:, 0:20] = 0
image[:, 240:257] = 0
image[:40, : ] = 0
image[230:255, :] = 0
cv2.imshow("Lenna", image)

image[:50, :] = 0
image[240:257, :] = 0
cv2.imshow("Lenna", image)
cv2.waitKey(0)

# Завдання 4
# Відкрийте зображення data/Lenna.png.
# Створіть маску для пікселів з інтенсивністю більше 128 та виведіть її.
# Також виведіть заперечення цієї маски.
# На оригінальному зображенні, усі пікселі які не відповідають масці замініть на 0 та виведіть результат

mask = image > 128
print(mask)


new_mask = mask.astype(np.uint8)
print(new_mask)

cv2.imshow("mask", new_mask * 255)

image[~mask] = 0

cv2.imshow("new image", image)

cv2.imwrite("new_image.png", image)

result = (image / 255) ** 0.5 * 255 # gamma параметри
result = result.astype(np.uint8)

cv2.imshow("image", result)


cv2.waitKey(0)