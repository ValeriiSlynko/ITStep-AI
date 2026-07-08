# --- ЛЕКЦІЯ ---
# РОБОТА з ВІДЕО (07.07.2026р.)

import cv2

# відкрити відео
cap = cv2.VideoCapture(
    0,   # шлях до файлу з відео або 0 для відеокамери комп'ютери
)


# інформація про відео
# розмір кадрів
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width)
print(height)

# FPS -- кількість кадрів у секунду
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)


# отримати перший кадр
# success, frame = cap.read()
# success -- True  якщо вдалось отримати кадр інакше False
# frame -- саме зображення кадру
#
# cv2.imshow("camera frame 1", frame)
# # cv2.waitKey(0)
#
# # оримати наступний кадр
# success, frame = cap.read()
# cv2.imshow("camera frame 2", frame)
# cv2.waitKey(0)

# - ЗБЕРЕЖЕННЯ ВІДЕО -
# кодек(розширення файлу(mp4, avi, xvd))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(
    "result_book.mp4",   # файл куди зберігати відео
    fourcc,      # кодек
    fps,         # частота кадрів в секунду
    (width, height),   # розмір (ширина, висота)
    isColor=False,   # чи є зображення кадрів кольоровими
)


# показ відео
while True:
    # отримати наступний кадр
    success, frame = cap.read()

    # перевірка чи вдалось отримати кадр
    if not success:
        break

    # обробка одного кадру
    cv2.imshow("camera", frame)

    # перевести в чорно-біле
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    # розмиття гауса
    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        sigmaX=1
    )
    cv2.imshow("blur", blur)

    # бінарізація
    adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    cv2.imshow("adapt", adapt)


    # очистка від шуму морфологічними операторами
    res = cv2.dilate(adapt, (3, 3), iterations=4)
    res = cv2.erode(res, (3, 3), iterations=4)

    cv2.imshow("morph", res)

    # запис відео у файл
    out_writer.write(adapt)

    # показувати кадри із затримкою 1 мс
    # якщо натиснута кнопка 'q', то зупинити відео
    # для Esc зробити  == 27
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# в кінці все закрити
out_writer.release()
cap.release()