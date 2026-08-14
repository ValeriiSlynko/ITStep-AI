# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 1
# Завдання 1
# Підключіть модель LLM за допомогою свого API key.
# Попросіть модель згенерувати:
# ● відповідь на питання у вигляді одного слова(наприклад яка столиця Франції?)
# ● код python
# ● коротку історію
# Підберіть параметри креативності та довжини завантеження api key, як змінну середовища

# завантеження api key як змінну середовища
import os
import dotenv

# завантаження даних з файлу .env
dotenv.load_dotenv()

# сам api key
api_key = os.getenv('GEMINI_API_KEY')

# сама модель LMM
import langchain
from langchain_google_genai import GoogleGenerativeAI

print(langchain.__version__)
# root/
#   - langchain.py
#   - langchain_google_genai.py

# створення моделі
# параметри креативності
llm = GoogleGenerativeAI(
    model='gemini-3.6-flash',   # назва моделі
    api_key=api_key,
    temperature=0.2

)

# запуск моделі
response = llm.invoke('Привіт! Серпень - яка пора року? Дай відповідь одним словом')
print(response)
response_2 = llm.invoke('Напиши коротенький код інтерактивки Питання-Відповідь')

print(response_2)

# Завдання 2
# Прочитайте файл data\lesson9\rules.txt з правилами користування атракціону.
# Напишіть програму яка отримує від користувачі питання та дає відповідь на нього виходячи з текстового файлу.
# Для цього об’єднайте правила користування з питанням користувача.
# Користувач задає питання поки не введе порожній рядок.
# Змініть файл rules.txt, щоб переконатись, що модель дійсно його читає.

with open("data/lesson9/rules.txt","r", encoding="utf-8") as text_file:
    rules = text_file.read()

# print(rules)

# while True:
#     question = input("My question: ")
#     response = llm.invoke(f"""
#         Ти адміністратор та консультант парку-атракціону.
#         Надавай відповіді відвідувачам на основі правил {rules}.
#         Опирайся лише на ті правила, що прописані в правилах.
#         Якщо не маєш відповіді, то відповідай: 'Ого у вас питаннячко! Давайте наступне!'
#         Питання від користувача {question}""")
#
#     print(f"Відповідь адмінчика: {response}")

# Завдання 3
# Створіть найпростіший чат бот. Напишіть моделі якого
# персонажа вона повинна вдавати(відомий актор, персонаж кіно\книги, тощо).
# Реалізуйте двома способами:
# 1. Модель отримує інструкцію в якому стилі відповідати
# та нове повідомлення.
# 2. Модель отримує інструкцію та історію попередніх
# повідомлень як від користувача, так і її власні відповіді у форматі
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI:

with open("data/lesson9/rules.txt", "r", encoding="utf-8") as text_file:
    rules = text_file.read()
instructions = f"""
        Ти адміністратор та консультант парку-атракціону.
        Надавай відповіді відвідувачам на основі правил {rules}.
        Опирайся лише на ті правила, що прописані в правилах."""

questions = []
responses = []

while True:
    question = input("My question: ")

    questions.append(question)

    # формуємо історію спілкування

    history = ""

    for old_question, old_response in zip(questions, responses):
        history += f"\nUser: {old_question}"
        history += f"\nModel: {old_response}"

    whole_text = f"""
    {instructions}

    Історія спілкування
    {history}
    User: {question}
    Model:
    """

    response = llm.invoke(whole_text)

    print(f"Відповідь адмінчика: {response}")

