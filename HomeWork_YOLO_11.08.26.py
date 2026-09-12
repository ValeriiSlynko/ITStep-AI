# Курс: AI+Python
# Модуль 2. Комп’ютерний зір
# Тема: opencv. Частина 2
# Завдання 1
# Відкрийте відео data/lesson_pose/squat.mp4
# Ваша задача рахувати кількість присідань.
# Отримайте перший кадр та виділіть основні точки.
# Отримайте координати 3-ох точок ноги
# Визначте кут між цими трьома точками.
# Скористайтесь функцією utils.get_angle(x1, y1, x2, y2, x3, y3) де x2, y2 – координати коліна(центральна точка)
# Запустіть відео та добавте на сам кадр кут згинання ніг.
# Визначіть нижню межу кута(якщо людина опустилась нижче вважаємо що вона достатньо опустилась)
# та верхню межу кута(якщо людина піднялась вище вважаємо, що вона достатньо піднялась)
# Добавте кількість присідань та кут на кожен кадр

import cv2
import ultralytics
from sympy import deg

import utils

cap = cv2.VideoCapture("data/lesson_pose/squat.mp4")
model = ultralytics.YOLO("yolo11s-pose.pt")

MIN_ANGLE = 55
MAX_ANGLE = 170

count_sitting = 0
is_sitting = True

while True:
    success, frame = cap.read()

    if not success:
        break

    # змінюємо(зменшуємо) розмір кадру
    frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)

    # cv2.imshow("frame",frame)

    results = model.predict(
        frame,
        device="cuda"
    )
    result = results[0]  #  дістаємо один результат зі списку
    # print(result)

    result_img = result.plot()
    cv2.imshow("result", result_img)

    keypoints = result.keypoints
    xy = keypoints.xy.cpu().numpy()

    xy = keypoints.xy
    xy = xy.cpu().numpy()

    if len(xy) > 0 and len(xy[0]) > 0:
        xy = xy[0]
        xy = xy.astype(int)
        # отримуємо координати 3-х точок ПРАВОЇ ноги: Таз/Pelvis(12), Коліно/Knee(14), Стопа/Foot(16)
        x_right_pelvis, y_right_pelvis = xy[12]
        x_right_knee, y_right_knee = xy[14]
        x_right_foot, y_right_foot = xy[16]

        # -I- МАЛЮЄМО ЛІНІЇ МІЖ ТОЧКАМИ (немов скелет ноги)
        # 1. Лінія від Таза до Коліна
        cv2.line(
            frame,
            pt1=(x_right_pelvis, y_right_pelvis),
            pt2=(x_right_knee, y_right_knee),
            color=(255, 255, 255),  # буде білий колір
            thickness=3,    # Обираємо товщину лінії
        )
        # 2. Лінія від Коліна до Стопи
        cv2.line(
            frame,
            pt1=(x_right_knee, y_right_knee),
            pt2=(x_right_foot, y_right_foot),
            color=(255, 255, 255),  # буде білий колір
            thickness=3,  # Обираємо товщину лінії
        )

        # -II- МАЛЮЄМО КРУГИ НА ТОЧКАХ
        # Таз (Зелений)
        cv2.circle(
            frame,  # зображення малювати коло на правій стороні таза
            center=(x_right_pelvis, y_right_pelvis),  # координати центру
            radius=12,  # радіус в пікселях
            color=(0, 255, 0),  # колір в BGR (зелений)
            thickness=-1,  # товщина ліній, -1 означає повністю заповнити кольором
        )
        # Коліно (Синій)
        cv2.circle(
            frame,  # зображення малювати коло на коліні правої ноги
            center=(x_right_knee, y_right_knee),  # координати центру
            radius=12,  # радіус в пікселях
            color=(255, 0, 0),  # колір в BGR (синій)
            thickness=-1,  # товщина ліній, -1 означає повністю заповнити кольором
        )
        # Стопа (Червоний)
        cv2.circle(
            frame,  # зображення малювати коло на лівому коліні
            center=(x_right_foot, y_right_foot),  # координати центру
            radius=12,  # радіус в пікселях
            color=(0, 0, 255),  # колір в BGR (червоний)
            thickness=-1,  # товщина ліній, -1 означає повністю заповнити кольором
        )

        # cv2.imshow("man", frame)
        # -III- ОБРАХУНОК КУТА ТА ЛІЧИЛЬНИК
        angle = utils.get_angle(x_right_pelvis, y_right_pelvis, x_right_knee, y_right_knee, x_right_foot, y_right_foot)

        if angle < MIN_ANGLE and is_sitting:
            count_sitting += 1
            is_sitting = False

        if angle > MAX_ANGLE:
            is_sitting = True

        # Виводимо кут безпосередньо біля коліна
        cv2.putText(
            frame,
            f"{int(angle)} degrees",
            (x_right_knee + 15, y_right_knee),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),    # колір в BGR (жовтий)
            2   # товщина ліній
        )

        # Виводимо загальну СТАТИСТИКУ у кутах кадру
        cv2.putText(
            frame,  # зображення де пишемо текст
            f"Angle: {int(angle)}, Squats: {count_sitting}",  # текст
            (30, 40),  # позиція, лівий нижній кут
            cv2.FONT_HERSHEY_SIMPLEX,  # шрифт
            1,  # розмір шрифту
            (255, 255, 255),  # колір в BGR
            2  # товщина ліній

        )
    cv2.imshow("Squat Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
