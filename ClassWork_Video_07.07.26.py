# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 3
import cv2

# -- ЗАВДАННЯ 1 --
# Виведіть відео з файлу data\lesson7\text.mp4 на екран та збережіть в новий файл.
# Змініть розмір зображення.
cap = cv2.VideoCapture(
    "data/lesson7/text.mp4", # шлях до файлу з відео або 0 для відеокамери комп'ютери
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

# - ЗБЕРЕЖЕННЯ ВІДЕО -
# кодек(розширення файлу(mp4, avi, xvd))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(
    "result_book.mp4",   # файл куди зберігати відео
    fourcc,      # кодек
    fps,         # частота кадрів в секунду
    (500, 500),   # розмір (ширина, висота)
    isColor=True,   # чи є зображення кадрів кольоровими
)

while True:
    success, frame = cap.read()
    if not success:
        break

    #cv2.imshow("origin", frame)
    print(frame.dtype)
    print(frame.shape)

    # --- зміна розміру кадрів(відео) ---
    new_frame = cv2.resize(
        frame,
        (500, 500)
    )
    cv2.imshow("new_frame", new_frame)

    out_writer.write(new_frame)     # запис відео у файл
    cv2.waitKey(50)

# в кінці все закрити
out_writer.release()
cap.release()

# -- ЗАВДАННЯ 2 --
# Відкрийте відео з файлу data\lesson7\text.mp4.
# Проведіть бінаризацію кадрів та збережіть в новий файл.

    # Відкриваємо відео text.mp4 з файлу
cap = cv2.VideoCapture(
    "data/lesson7/text.mp4", # шлях до файлу з відео або 0 для відеокамери комп'ютери
)
    # запуск нескінченного циклу для запуску відео
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # cv2.imshow("Frame", frame)  # показуємо відео
    print(frame.dtype)            # виводимо тип кадрів(відео)
    print(frame.shape)            # виводимо розмір кадрів(відео)

    # --- зміна розміру кадрів(відео) ---
    new_frame =  cv2.resize(
        frame,
        (500, 500)
        )

    cv2.imshow("new_frame", new_frame)

    # перевід кольорового зображення на чорно-біле
    gray_image = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # експериментальні дії
    # result_gray = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # cv2.imshow('NLMean', result_gray)

    # GAUSS (метод - Гаусове розмиття)
    gauss = cv2.GaussianBlur(
        gray_image,       # зображення з шумом
        (3, 3),     # розмір фільтру(ядра)
        sigmaX=0,      # наскільки важливими є далекі пікселі, adaptive -> value
    )
    cv2.imshow('gauss', gauss)

    # ДВОСТОРОННІЙ ФІЛЬТР
    bilat = cv2.bilateralFilter(
        gray_image,  # зображення з шумом
        d=5,  # розмір фільтра
        sigmaColor=75,  # наскільки важливі пікселі іншого кольору
        sigmaSpace=50,  # наскільки важливими є далекі пікселі
    )
    cv2.imshow("Bilateral Filter", bilat)

    result = cv2.adaptiveThreshold(
        bilat,  # зображення з текстом (чорно-біле)
        255,  # білий колір
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # фільтр для обрахунку порогу (Гаус)
        cv2.THRESH_BINARY,  # цей рядок просто треба вказаати
        7,  # розмір фільтра
        3,  # наскільки ПІКСЕЛЬ має відрізнятися від порогу
    )
    cv2.imshow('Adaptive', result)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


# -- ЗАВДАННЯ 3 --
# Відкрийте відео з файлу data\lesson7shapes.mp4.
# Проведіть виділення країв на кадрах та збережіть в новий файл

cap = cv2.VideoCapture(
    "data/lesson7/shapes.mp4" # шлях до файлу з відео або 0 для відеокамери комп'ютери
)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- зміна розміру кадрів(відео) ---
    new_frame = cv2.resize(
        frame,
        (500, 500)
    )

    cv2.imshow("new_frame", new_frame)
    cv2.imshow("frame", frame)

    hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
    lower = (40, 80, 50)
    upper = (65, 255, 255)

    mask_green = cv2.inRange(hsv, lower, upper)
    cv2.imshow("mask", mask_green)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break