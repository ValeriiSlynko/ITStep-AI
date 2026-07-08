# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 3

# ЗАВДАННЯ 1
# Відкрийте відео з файлу data\lesson7\meter.mp4.
# Проведіть бінаризацію кадрів та збережіть в новий файл.
# Можливо очистіть від шуму або наведіть різкість через bilateralFilter

import cv2
import numpy as np

# відкриваємо відео 'meter.mp4' з файлу
cap = cv2.VideoCapture(
    'data/lesson7/meter.mp4',   # шлях до файлу з відео
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

#  ЗБЕРЕЖЕННЯ ВІДЕО у новий файл
# кодек(розширення файлу(mp4, avi, xvd))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # кодек зберігаємо у 'mp4v' форматі
out_writer = cv2.VideoWriter(
    "New_meter_video.mp4",   # файл куди зберігати відео
    fourcc,      # кодек
    fps,         # частота кадрів в секунду
    (400, 300),   # розмір (ширина, висота)
    isColor=False,   # чи є зображення кадрів кольоровими
)

# запуск нескінченного циклу для запуску відео
while True:
    success, frame = cap.read()

    if not success:
        break

    # cv2.imshow("Meter", frame) # для попереднього перегляду відеофайлу
    # print("Video 'Meter' має тип:", frame.dtype)
    # print("Video 'Meter' має розміри", frame.shape)

    # --- зміна розміру кадрів(відео) ---
    new_frame = cv2.resize(
        frame,
        (400, 300)
    )
    #cv2.imshow("Orig_new_frame", new_frame)

    # перевід кольорового зображення на чорно-біле
    gray_image_meter = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # GAUSS (Гаусове розмиття)
    gauss_frame = cv2.GaussianBlur(
        gray_image_meter,  # зображення з шумом
        (3, 3),  # розмір фільтру(ядра)
        sigmaX=5,  # наскільки важливими є далекі пікселі, adaptive -> value
    )
    #cv2.imshow('gauss', gauss_frame)

    # ДВОСТОРОННІЙ ФІЛЬТР
    # bilat = cv2.bilateralFilter(
    #     gray_image_meter,  # зображення з шумом
    #     d=11,  # розмір фільтра
    #     sigmaColor=25,  # наскільки важливі пікселі іншого кольору
    #     sigmaSpace=25,  # наскільки важливими є далекі пікселі
    # )
    # cv2.imshow("Bilateral Filter", bilat)

    result = cv2.adaptiveThreshold(
        gauss_frame,  # зображення з текстом (чорно-біле)
        255,  # білий колір
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # фільтр для обрахунку порогу (Гаус)
        cv2.THRESH_BINARY,  # цей рядок просто треба вказати
        31,  # розмір фільтра
        4,  # наскільки ПІКСЕЛЬ має відрізнятися від порогу
    )
    cv2.imshow('Adaptive', result)

    # очищення від шуму морфологічними операторами
    # робимо білі пікселі білими
    # dilat = cv2.dilate(
    #     bilat,
    #     (3, 3),
    #     iterations=3  # скільки разів застосувати
    # )
    # cv2.imshow("morph-dilat", dilat)

    # робимо чорні пікселі чорними
    # erode = cv2.erode(
    #     bilat, (3, 3), iterations=3)
    # cv2.imshow("morph-erode", erode)

    out_writer.write(result)  # запис відео у файл
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# в кінці все закриваємо
cap.release()
out_writer.release()  # ОЦЕЙ РЯДОК ПЕРЕСТАНЕ ОБРІЗАТИ ВІДЕО!
cv2.destroyAllWindows()

print("Відео 'Meter' успішно оброблено та збережено як 'New_meter_video.mp4'!\n Кращим методом обрано 'Гаусове розмиття'")