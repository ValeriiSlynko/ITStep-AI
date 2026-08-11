# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 3
# Завдання 1
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та виведіть результат, підберіть параметри
# Можете змінити розмір кадру для кращої візуалізації
# cv2.resize()
import cv2
import ultralytics
from pandas.conftest import names
from torch.cuda import device

# model = ultralytics.YOLO("yolo11x.pt")
#
# # отримуємо зображення з відео
# meet = cv2.VideoCapture(r'data\lesson8\meetings.mp4')
# success, img = meet.read()
#
# # зміна розміру кадру
# img = cv2.resize(img, None, fx=0.25, fy=0.25)
#
# cv2.imshow("Meeting", img)
#
# # застосування моделі
# result = model.predict(
#     img,
#     device="cuda",  # графічний процесор
#     conf=0.25,    # мінімальна ймовірність для об'єктів
#     iou=0.9,      # наскільки сильно перетинаються рамки (при більшому перетині залишаємо знач. з більш. ймовірністю)
#     #classes=[0, 60],  # класи, які враховувати
# )
# print(type(result[0]))
# print(result)
#
# # results - список з одним елементом
# # отримуємо результат
# result = result[0]
#
# # отримати назви класів
# names = result.names
# print(type(result))
# print(names)
#
# # самі об'єкти
# boxes = result.boxes
# print(type(boxes))
# print(boxes)
#
# # візуалізація результатів
# result_meet = result.plot()
# cv2.imshow("Result", result_meet)
#
# cv2.waitKey(0)
#
# # Запускаємо цикл, який працює, поки відео відкрито
# while meet.isOpened():
#     # Отримуємо черговий кадр з відео
#     success, img = meet.read()
#
#     # Якщо кадрів більше немає (відео закінчилося) — виходимо з циклу
#     if not success:
#         break
#
#     # Зміна розміру поточного кадру
#     img = cv2.resize(img, None, fx=0.25, fy=0.25)
#
#     # Застосування моделі на графічному процесорі
#     # verbose=False вимикає текстовий спам YOLO в консоль для кожного кадру
#     results = model.predict(img, device="cuda", verbose=False)
#
#     # Отримуємо перший результат зі списку (для поточного кадру)
#     result = results[0]
#
#     # Візуалізація результатів детекції на кадрі
#     result_meet = result.plot()
#
#     # Показуємо оброблений кадр у динаміці
#     cv2.imshow("Result Meeting", result_meet)
#
#     # Чекаємо 1 мілісекунду між кадрами.
#     # Якщо ви захочете зупинити відео достроково, натисніть англійську клавішу 'q'
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()
#
# # Після завершення відео звільняємо пам'ять та закриваємо вікна
# meet.release()
# cv2.destroyAllWindows()

# Завдання 2
# Відкрийте відео з файлу data\lesson8\meetings.mp4
# Застосуйте детекцію та почніть показувати відео з
# моменту, коли людей стало 5

# Завантажуємо модель
model = ultralytics.YOLO("yolo11s.pt")

# Відкриваємо відеофайл
meet = cv2.VideoCapture(r'data\lesson8\meetings.mp4')

# Прапорець, який дозволить нам почати показ відео
start_showing = False

while meet.isOpened():
    success, img = meet.read()

    if not success:
        break

    # Зміна розміру
    img = cv2.resize(img, None, fx=0.25, fy=0.25)

    # Детекція на CUDA
    results = model.predict(img, device="cuda", verbose=False)
    result = results[0]  # беремо перший результат

    # Рахуємо кількість людей на поточному кадрі (клас людини в YOLO - це 0)
    # Перетворюємо тензор з класами в список методм .tolist() для зручності
    detected_classes = result.boxes.cls.tolist()
    people_count = detected_classes.count(0)

    # Якщо ми ще не показуємо відео, але нарахували 5 або більше людей
    if not start_showing and people_count >= 5:
        start_showing = True
        print(f"Знайдено {people_count} людей! Починаємо показ відео.")

    # Візуалізуємо результати
    result_meet = result.plot()

    # УМОВА ПОКАЗУ: imshow викликається тільки тоді, коли start_showing став True
    if start_showing:
        # Додамо лічильник людей прямо на екран для наочності
        cv2.putText(result_meet, f"People: {people_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Result Meeting - Task 2", result_meet)

    # Робимо маленьку затримку (навіть якщо відео не показується, щоб програма не зависала)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

meet.release()
cv2.destroyAllWindows()
