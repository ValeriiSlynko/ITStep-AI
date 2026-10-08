import os
import cv2
# from ultralytics import YOLO

from collections import Counter

import utils

# ==================================================
# ЕКЗАМЕНАЦІЙНИЙ ПРОЄКТ: ДЕТЕКЦІЯ ОБ'ЄКТІВ (YOLO)
# Етап 1: Аналіз та підрахунок класів у датасеті
# ==================================================

# 1. ВКАЗУЄМО абсолютний шлях до папки з розміткою train/labels
# Перевірте, щоб шлях відповідав вашому розпакованому

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

# --------------------
# ВИВІД РЕЗУЛЬТАТІВ
# --------------------
print(f"Опрацьовано текстових файлів (labels): {total_files}")
print(f"Загальна кількість виявлених об'єктів: {total_objects}")
print("-" * 60)
print("РОЗПОДІЛ ОБ'ЄКТІВ ЗА КЛАСАМИ:")

for class_id, count in sorted(class_counter.items()):
    percentage = (count / total_objects) * 100 if total_objects > 0 else 0
    print(f"Клас{class_id}: {count} шт. ({percentage:.2f} %)")

print("=" * 60)

# -------------------------------------------------------------
#  РЕЗУЛЬТАТ ЕТАПУ 1: ЗЧИТУВАННЯ ТА АНАЛІЗ ДАТАСЕТУ АЕРОЗЙОМКИ
# -------------------------------------------------------------
# Опрацьовано текстових файлів (labels): 2094
# Загальна кількість виявлених об'єктів: 43575
# -------------------------------------------------------------
# РОЗПОДІЛ ОБ'ЄКТІВ ЗА КЛАСАМИ:
# Клас 0:   8696 шт. (19.96 %)
# Клас 1:   1483 шт. (3.40 %)
# Клас 2:   33396 шт. (76.64 %)
# -------------------------------------------------------------

# Аналіз показує виражений дисбаланс!
# Якщо залишити все як є, модель YOLO ідеально навчиться знаходити Клас 2, але практично ігноруватиме Клас 1.

#   ДАНЕ ПИТАННЯ має два шляхи рішення:
# 1. Видалити частину зображень Класу 2.
# 2. Продублювати (розмножити) зображення з Класом 1 у 3 рази.

# Другий варіант (дублювання) — вважаю набагато кращим для точності моделей на БПЛА!
# бо ми не втрачаємо цінні фотографії, а навпаки підтягуємо рідкісний 'Клас1' до інших!

# Як працює дублювання (оверсемплінг)?
# Ми знайдемо всі  .txt  файли розмітки, де є 'Клас 1', і створимо їхні копії з новими іменами
# (наприклад, image_001_copy1.png та image_001_copy1.txt).


# ==================================================
# ЕКЗАМЕНАЦІЙНИЙ ПРОЄКТ: ДЕТЕКЦІЯ ОБ'ЄКТІВ (YOLO)
# Етап 2: Балансування класів у датасеті
# ==================================================

# Скрипт Етапу 2 (Оверсемплінг / Дублювання Класу 1)

# Додатково підвантажуємо бібліотеку для високорівневих операцій з файлами (копіювання, переміщення, створення архівів)
import shutil
#
# Змінні шляхів до розпакованих папок (великі літери за правилами Python, бо це константи).
TRAIN_LABELS_PATH = r"D:\IT_STEP_Academy\EXAM\train\labels"
TRAIN_IMAGES_PATH = r"D:\IT_STEP_Academy\EXAM\train\images"

# Змінна: скільки СТВОРЮВАТИ ДОДАТКОВИХ копій файлів з Класом 1
NUM_COPIES = 3

print("=" * 60)
print("       ЕТАП 2: ДУБЛЮВАННЯ ДЕФІЦИТНОГО КЛАСУ 1")
print("=" * 60)

# Змінна-лічильник успішно створених дублікатів
copied_files_count = 0

# Отримуємо список усіх файлів розмітки перед початком дублювання
# Зберігаємо у змінну список тільки початкових файлів .txt
original_label_files = [f for f in os.listdir(TRAIN_LABELS_PATH) if f.endswith(".txt")]
#   де:
# original_label_files - змінна, куди ми зберігаємо список тільки початкових файлів .txt.
# f — наша тимчасова змінна для кожного файла у циклі.
# # os.listdir() — вбудована функція (повертає список файлів у папці).
# # f.endswith(".txt") — вбудований метод (перевіряє розширення файла).

for filename in original_label_files:
    label_path = os.path.join(TRAIN_LABELS_PATH, filename)

    # Змінна-прапорець: чи є в цьому файлі хоча б один об'єкт 'Класу 1'
    has_class_1 = False

    # Відкриваємо файл і шукаємо 'Клас 1'
    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            # Перевіряємо перше число в рядку (це ID класу)
            if parts and int(parts[0]) == 1:
                has_class_1 = True
                break   # break — спрацьовує команда зупинки циклу коли 'Клас 1' вже знайдено

    # Якщо файл містить 'Клас 1' — робимо дублікати файлу розмітки та відповідного фото
    if has_class_1:

        base_name = os.path.splitext(filename)[0]
                    # os.path.splitext() — вбудована функція, відрізає розширення .txt, залишаючи ім'я

    # Визначаємо розширення фотографії/зображення (.png або .jpg)
    img_ext = ".png"
    image_path = os.path.join(TRAIN_IMAGES_PATH, f"{base_name}.png")

    if not os.path.exists(image_path):
            # os.path.exists() — вбудована функція перевірки наявності файла на диску
        img_ext = ".jpg"
        image_path = os.path.join(TRAIN_IMAGES_PATH, f"{base_name}.jpg")

    # Якщо зображення існує — створюємо NUM_COPIES копій(= 3)
    if os.path.exists(image_path):
        for i in range(1, NUM_COPIES + 1):
            new_base_name = f"{base_name}_dup{i}"
            new_label_path = os.path.join(TRAIN_LABELS_PATH, f"{new_base_name}.txt")
            new_image_path = os.path.join(TRAIN_IMAGES_PATH, f"{new_base_name}.{img_ext}")

            # НАРЕШТІ ВИКОРИСТОВУЄМО shutil.copyfile() — вбудовану функцію для фізичного дублювання файлів на диску
            shutil.copyfile(label_path, new_label_path)
            shutil.copyfile(image_path, new_image_path)
            copied_files_count += 1

print(f"[ІНФО] Успішно створено {copied_files_count} копій файлу для 'Класу 1'")
print("=" * 60)

# -------------------------------------------------------
#   ПІДРАХУНОК НОВОЇ СТАТИСТИКИ КЛАСІВ ПІСЛЯ ДУБЛЮВАННЯ
# -------------------------------------------------------
final_counter = Counter()  # Наша змінна: підсумковий лічильник
final_total_files = 0      # Наша змінна: загальна кількість файлів
final_total_objects = 0    # Наша змінна: загальна кількість об'єктів

for filename in os.listdir(TRAIN_LABELS_PATH):
    if filename.endswith(".txt"):
        final_total_files += 1
        with open(os.path.join(TRAIN_LABELS_PATH, filename), "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    final_counter[class_id] += 1
                    final_total_objects += 1

print("ОНОВЛЕНА СТАТИСТИКА ДАТАСЕТУ ПІСЛЯ БАЛАНСУВАННЯ:")
print(f"Загальна кількість текстових файлів (labels): {final_total_files}")
print(f"Нова загальна кількість об'єктів: {final_total_objects}")
print("-" * 60)

for class_id, count in sorted(final_counter.items()):
    percentage = (count / final_total_objects) * 100 if final_total_objects > 0 else 0
    print(f"  • Клас {class_id}: {count} шт. ({percentage:.2f}%)")

print("=" * 60)

# -------------------------------------------------------------
#  РЕЗУЛЬТАТ ЕТАПУ 2: ОНОВЛЕНА СТАТИСТИКА ДАТАСЕТУ ПІСЛЯ БАЛАНСУВАННЯ:
# -------------------------------------------------------------
# Загальна кількість текстових файлів (labels): 4182
# Нова загальна кількість об'єктів: 48777
# -------------------------------------------------------------
# РОЗПОДІЛ ОБ'ЄКТІВ ЗА КЛАСАМИ:
#   • Клас 0: 9539 шт. (19.56%)
#   • Клас 1: 5797 шт. (11.88%)
#   • Клас 2: 33441 шт. (68.56%)
# -------------------------------------------------------------

# =========================================================
#   ЕТАП 3: ПАКУВАННЯ ЗБАЛАНСОВАНОГО ДАТАСЕТУ В ZIP-АРХІВ
# =========================================================
# [ОГОЛОШУЄМО НАШІ ЗМІННІ ПРЯМО ТУТ, ЩОБ PYTHON ЇХ БАЧИВ]:
# 1. Наша змінна: шлях до папки EXAM, яку пакуємо
PROJECT_ROOT = r"D:\IT_STEP_Academy\EXAM"  # Корінь проєкту

# 2. Наша змінна: шлях та ім'я підсумкового архіву (без розширення .zip)
OUTPUT_ZIP_PATH = r"D:\IT_STEP_Academy\EXAM_balanced"  # Шлях збереження ZIP

print("=" * 60)
print("       ЕТАП 3: СТВОРЕННЯ ZIP-АРХІВУ ДЛЯ KAGGLE")
print("=" * 60)

print(f"[ІНФО] Початок архівації папки: {PROJECT_ROOT}")
print("[ІНФО] Зачекайте деякий час поки створиться папка з zip архівом")

archive_path = shutil.make_archive(
    base_name=OUTPUT_ZIP_PATH,
    format="zip",
    root_dir=PROJECT_ROOT
)
# - Де:
# shutil.make_archive() — вбудована функція з бібліотеки shutil для пакування:
# 1. base_name = OUTPUT_ZIP_PATH (куди і під якою назвою зберегти)
# 2. format = "zip" (формат стиснення)
# 3. root_dir = PROJECT_ROOT (яку саме папку пакувати)

print(f"[УСПІХ] Архів успішно створено за шляхом:")
print(f"  -> {archive_path}")
print("=" * 60)