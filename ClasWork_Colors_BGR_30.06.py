# Курс: AI+Python
# Модуль 12. Аналіз даних
# Тема: Стеки. Частина 2
# Завдання 1
# Відкрийте зображення data/lesson2/marbles.png.
# Використайте кольорову сегментацію для отримання масок до
# кульок:
#  синього кольору
#  зеленого і червоного
#  чорного
#  білого
#  усіх кульок
import cv2
from matplotlib.pyplot import hsv

image = cv2.imread('data/lesson2/marbles.png')

cv2.imshow('image', image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.waitKey(0)

lower = (100, 100, 120)
upper = (130, 255, 255)

mask = cv2.inRange(hsv, lower, upper)

cv2.imshow('mask', mask)

cv2.waitKey(0)

image = cv2.imread('data/lesson2/marbles.png')

cv2.imshow('image', image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Червоний ті зелений кольори
lower = (0, 120, 150)
upper = (15, 255, 255) # отримання червоного

mask_red = cv2.inRange(hsv, lower, upper)
cv2.imshow('mask red', mask_red)

lower = (35, 90, 80)
upper = (90, 255, 255) # отримання зеленого

mask_green = cv2.inRange(hsv, lower, upper)
cv2.imshow('mask green', mask_green)

mask_red_green = cv2.bitwise_and(mask_red, mask_green)

cv2.waitKey(0)

# Чорний колір
lower = (0, 0, 0)
upper = (140, 100, 50) # отримання чорного

mask_black = cv2.inRange(hsv, lower, upper)
cv2.imshow('mask red', mask_black)
cv2.waitKey(0)

# Білий колір
lower = (0, 0, 200)
upper = (180, 30, 255) # отримання білого

mask_wight = cv2.inRange(hsv, lower, upper)
cv2.imshow('mask wight', mask_wight)
cv2.waitKey(0)

# Завдання 2
# Відкрийте зображення data/lesson2/cell.png. Покращте
# зображення за допомогою вирівнювання гістограми. Оскільки
# зображення кольорове, вам доведеться зробити наступні
# кроки:
#  перевести зображення в LAB
#  розбити зображення на канали l, a та b
#  вирівняти гістограму для l
#  зібрати канали назад в зображення
#  перевести результат назад в BGR
# Порівняйте результати для 2 алгоритмів.

image_1 = cv2.imread('data/lesson2/cell.png')
image_1 = cv2.resize(image_1, (500, 500))

cv2.imshow('image', image_1)
lab = cv2.cvtColor(image_1, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

new_1 = cv2.equalizeHist(l)

new_lab = cv2.merge((new_1, a, b))


new_image_1 = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

cv2.imshow('new_image', new_image_1)

cv2.waitKey(0)