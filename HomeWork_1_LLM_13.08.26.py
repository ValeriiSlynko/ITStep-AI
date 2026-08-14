# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 1
# Завдання 1
# Прочитайте файл data\lesson9\return_policy.txt Та напишіть простий чат бот для відповідей на питання
# користувачів стосовно повернення товару.
# Діалог завершується коли користувач вводить порожній рядок.
# Передавайте усю історію спілкування у форматі:
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI:

# завантеження api key як змінну середовища
import os
import dotenv


# завантаження даних з файлу .env
dotenv.load_dotenv()

# сам api key
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("Помилка: Перевірте наявність GEMINI_API_KEY у файлі .env!")
    exit()

# сама модель LMM
import langchain
from langchain_google_genai import GoogleGenerativeAI
import langchain_google_genai

print(langchain.__version__)
# root/
#   - langchain.py
#   - langchain_google_genai.py

# створення моделі
# параметри креативності
llm = GoogleGenerativeAI(
    model='gemini-3.5-flash-lite',   # назва моделі
    api_key=api_key,
    temperature=0.5
)

with open("data/lesson9/return_policy.txt", "r", encoding="utf-8") as text_file_policy:
    product_policy = text_file_policy.read()

# print(product_policy)

instructions_answer = f"""
        1. Ти менеджер і консультант магазину 'ROZETKA'!
        2. Привітайся один раз: Вітаю! \nМене звати Кутаб! \nЯ фахівець по роботі з клієнтами. \nЧим можу допомогти? 
        2. В тебе є чітка інструкція {product_policy} з умовами повернення товару.  
        3. Перш за все надавай відповіді відвідувачам на основі умов, що викладені в {product_policy}.
        4. Запамятовуй всі питання та відповіді
        5. Якщо в інструкції не має чіткої відповіді, то скажи 'Пропоную Вам такі варіанти: ' (пропонуєш)
        6. Спробуй сам запропонувати альтернативу (вигадай), щоб зберегти клієнта та його настрій."""

questions = []
responses = []

while True:
    question = input("\n Human: ")

    # Перевірка на порожній рядок для завершення діалогу
    if question.strip() == "":
        print("Діалог завершено. Гарного дня!")
        break

    # Формуємо історію спілкування відповідно до шаблону ТЗ

    history = ""
    for old_question, old_response in zip(questions, responses):
        history += f"\nHuman: {old_question}"
        history += f"\nAI: {old_response}"

    # Збираємо повний текст промпту (суворо за форматом завдання)
    whole_text = f"Instruction: {instructions_answer}\n\n{history}Human: {question}\nAI:"

    # Отримуємо відповідь від LLM
    response = llm.invoke(whole_text)

    # Виводимо відповідь користувачу
    print(f"AI: {response}")

    # Зберігаємо поточне питання та відповідь в історію ДЛЯ НАСТУПНИХ ітерацій
    questions.append(question)
    responses.append(response)

