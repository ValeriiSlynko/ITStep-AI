# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 3
import cv2
# -- Завдання 1 --
# Отримайте перший кадр з файлу data\lesson8\animals.mp4
# та виведіть його на екран.
# Проведіть детекцію об’єктів зо допомогою YOLO та
# виведіть результати.
# Змініть параметри моделі conf та iou і подивіться як це
# впливає на результат.
# Отримайте рамки для кожного об’єкта, виріжіть їх та
# виведіть як окремі зображення
import ultralytics
from torch.cuda import device

import cv2
import ultralytics

# 1. Ініціалізація моделі
model = ultralytics.YOLO("yolo11s.pt")

# 2. Відкриваємо відео
cap = cv2.VideoCapture("data/lesson8/animals.mp4")

# --- Частина 1: Робота з першим кадром ---
success, img = cap.read()
if not success:
    print("Не вдалося завантажити відео або файл порожній!")
    exit()

# Зменшуємо розмір для зручності відображення
img_resized = cv2.resize(img, (600, 400))
cv2.imshow("First Frame - Original", img_resized)

# Детекція на першому кадрі
result_first = model.predict(img_resized, device="cuda:0", conf=0.5, iou=0.7)[0]

# Відображення кадру з усіма рамками YOLO
res_plot = result_first.plot()
cv2.imshow("First Frame - YOLO Detections", res_plot)
cv2.waitKey(0)  # Чекаємо натискання клавіші

# Вирізаємо та показуємо кожен об'єкт окремо
names = result_first.names
for idx, box in enumerate(result_first.boxes):
    # Отримуємо координати та конвертуємо в int на CPU
    xyxy = box.xyxy.cpu().numpy().astype(int)[0]
    conf = box.conf.cpu().numpy()[0]
    cls = int(box.cls.cpu().numpy()[0])

    x1, y1, x2, y2 = xyxy

    # Вирізаємо об'єкт з кадру
    cropped_obj = img_resized[y1:y2, x1:x2]

    # Назва об'єкта
    obj_name = names[cls]

    # Виводимо у вікно (назва вікна має бути унікальною для кожного об'єкта)
    cv2.imshow(f"Obj_{idx}: {obj_name} ({conf * 100:.1f}%)", cropped_obj)

print("Натисніть будь-яку клавішу в вікні OpenCV, щоб закрити вікна першого кадру та запустити відео...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Частина 2: Цикл обробки всього відео ---
# Скидаємо вказівник відео на початок, щоб обробити його повністю
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Робимо ресайз кадру відео, щоб координати рамок збігалися з картинкою
    frame_resized = cv2.resize(frame, (600, 400))

    # Детекція поточного кадру
    result = model.predict(frame_resized, device="cuda:0", conf=0.5, iou=0.7, verbose=False)[0]

    # Показуємо загальний кадр з рамками
    cv2.imshow("Video Stream", result.plot())

    # Якщо потрібно вирізати об'єкти на льоту (у відео):
    for idx, box in enumerate(result.boxes):
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]
        cls = int(box.cls.cpu().numpy()[0])

        x1, y1, x2, y2 = xyxy
        cropped_obj = frame_resized[y1:y2, x1:x2]

        # Перевірка, щоб кроп не був порожнім (іноді рамка виходить за межі екрана)
        if cropped_obj.size > 0:
            cv2.imshow(f"Live_Crop_{idx}", cropped_obj)

    # Зупинка на клавішу 'q' (затримка 1 мс для плавного відео)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очищення пам'яті
cap.release()
cv2.destroyAllWindows()


# -- Завдання 2 --
# Напишіть програму по відстеженню об’єкта на відео.
# Відкрийте відео з файлу data\lesson8\animals.mp4 та виведіть на екран результат детекції.
# Попросіть користувача ввести ID об’єкта, який потрібно відстежувати
# Для віх наступних кадрів проведіть детекцію, отримайте
# рамку для об’єкта з потрібним ID та виведіть її на екран.
# Додатково показуйте оригінальне відео.
# Скористайтесь для роботи model.track()
