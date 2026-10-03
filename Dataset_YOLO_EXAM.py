import os
import cv2

from collections import Counter
from ultralytics import YOLO
from sympy import deg

import utils

# ==================================================
# ЕКЗАМЕНАЦІЙНИЙ ПРОЄКТ: ДЕТЕКЦІЯ ОБ'ЄКТІВ (YOLO)
# Етап 1: Аналіз та підрахунок класів у датасеті
# ==================================================

# 1. ВКАЗУЄМО абсолютний шлях до папки з розміткою train/labels
# Перевірте, щоб шлях відповідав вашому розпакованому архіву
LABELS_PATH = r"D:\IT_STEP_Academy\EXAM\train\labels"

print("=" * 60)
print("       ЗЧИТУВАННЯ ТА АНАЛІЗ ДАТАСЕТУ АЕРОЗЙОМКИ")
print("=" * 60)

# створюємо екземпляр лічильника
class_counter = Counter()
total_files = 0
total_objects = 0

# 2. ПЕРЕВІРЯЄМО існування і шлях до папки
if not os.path.exists(LABELS_PATH):
    print(f"ПОМИЛКА! Папку не знайдено за шляхом {LABELS_PATH} або не вірно вказаний шлях")
    print("Будь-ласка, перевірте назву папки та правильність шляху!")
else:
    # 3. СКАНУЄМО всі файли у папці LABELS_PATH
    for file_name in os.listdir(LABELS_PATH):
        if file_name.endswith(".txt"):
            total_files += 1
            file_full_path = os.path.join(LABELS_PATH, file_name)

            # відкриваємо кожен файл для зчитування рядків
            with open(file_full_path, "r", encoding="utf-8") as file:
                for line in file:
                    line_data = line.strip().split()
                    if line_data:
                        # перший елемент рядка - це ID класу(число)
                        class_id = int(line_data[0])
                        class_counter[class_id] += 1
                        total_objects += 1

# ==================================================
# ВИВІД РЕЗУЛЬТАТІВ
# ==================================================
print(f"Опрацьовано текстових файлів (labels): {total_files}")
print(f"Загальна кількість виявлених об'єктів: {total_objects}")
print("-" * 60)
print("РОЗПОДІЛ ОБ'ЄКТІВ ЗА КЛАСАМИ:")

for class_id, count in sorted(class_counter.items()):
    percentage = (count / total_objects) * 100 if total_objects > 0 else 0
    print(f"Клас{class_id}: {count}шт. ({percentage:.2f}%)")

print("=" * 60)