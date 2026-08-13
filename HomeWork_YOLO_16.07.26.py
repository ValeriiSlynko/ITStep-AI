# Курс: AI+Python
# Модуль 2. Комп’ютерний зір
# Тема: opencv. Частина 2

# Завдання 1
# Відкрийте зображення data/lesson_seg/tumor1.jpg
# Проведіть сегментацію зображення використовуючи
# модель data/lesson_seg/brain-tumor-seg.jpg
# Визначте площу пухлини в пікселях.
# Визначте площу в см2 (1 піксель – 0,0025 см2).
# Залежно від площі присвойте пухлині певний тип
#  <10см2 – small
#  10-25см2 – middle
#  >25см2 – large
# Покажіть пухлину – за допомогою маски усі лишні
# пікселі зробіть 0, а як назву зображення використайте її тип

# Курс: AI+Python
# Модуль 2. Комп’ютерний зір
# Тема: opencv. Частина 2
# Завдання 1

import cv2
import numpy as np
import ultralytics

# 1. Завантажуємо оригінальне зображення
img = cv2.imread('data/lesson_seg/tumor1.jpg')
# показуємо
cv2.imshow("tumor1", img)

# Завантажуємо модель сегментації YOLO
model = ultralytics.YOLO('data/lesson_seg/brain-tumor-seg.pt')

if img is None or model is None:
    print("Помилка: Перевірте шляхи до файлів зображення та моделі!")
    exit()

# 2. Проводимо сегментацію зображення за допомогою моделі
results = model.predict(img, device="cuda", verbose=False)
result = results[0]  # Беремо перший результат

# 3. Витягуємо бінарну маску пухлини з результатів YOLO
# Перевіряємо, чи модель знайшла хоч якийсь об'єкт для сегментації
if result.masks is not None:
    # Отримуємо маску у вигляді масиву numpy (розміром як оригінальне зображення)
    # .data[0] бере маску першого знайденого об'єкта
    binary_mask = result.masks.data[0].cpu().numpy()

    # YOLO повертає маску з плаваючою крапкою (0.0 або 1.0),
    # приведемо її до стандартного типу OpenCV (uint8) та значень 0 і 255
    binary_mask = (binary_mask * 255).astype(np.uint8)

    # Змінюємо розмір маски, щоб вона точно збігалася з розмірами оригінального фото
    binary_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]))
else:
    print("Модель не виявила пухлини на зображенні!")
    exit()

# 4. Визначаємо площу пухлини в пікселях (рахуємо білі пікселі)
tumor_pixels = np.sum(binary_mask == 255)

# 5. Визначаємо площу в см2 (1 піксель = 0.0025 см2)
area_cm2 = tumor_pixels * 0.0025

# 6. Присвоюємо тип залежно від площі
if area_cm2 < 10:
    tumor_type = "small"
elif 10 <= area_cm2 <= 25:
    tumor_type = "middle"
else:
    tumor_type = "large"

# Вивід інформації в консоль
print(f"--- Результати аналізу ---")
print(f"Площа: {tumor_pixels} пікселів")
print(f"Площа: {area_cm2:.2f} см²")
print(f"Тип пухлини: {tumor_type}")

# 7. Накладаємо бінарну маску (робимо всі зайві пікселі 0)
segmented_img = cv2.bitwise_and(img, img, mask=binary_mask)

# 8. Показуємо результат. Назва вікна — це її тип (small, middle або large)
cv2.imshow(tumor_type, segmented_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

