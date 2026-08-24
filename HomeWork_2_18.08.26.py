# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 2
# Завдання 1
# Напишіть промпт для створення плану навчального курсу з певної теми
# для цільової аудиторії (початківці, професіонали, діти тощо).
# Вхідні параметри: тема, опис цільової аудиторії
# Реалізуйте двома способами:
#  Zero-shot
#  Few-shot

import os
import dotenv
import langchain
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# завантаження даних з файлу .env
dotenv.load_dotenv()

# сам API KEY
api_key = os.getenv("GEMINI_API_KEY")

# створення моделі
llm = GoogleGenerativeAI(
    model='gemini-3.5-flash-lite',  # назва моделі
    api_key=api_key,
    temperature=0.7
)

print("\nРеалізація способом 'Zero-shot'")

prompt_zero_shot = PromptTemplate.from_template("""
    Ти - керівник ІТ академії
    ЗАДАЧА: написати план навчального курсу з певної теми для цільової аудиторії {target_audience}.
    
    ###ВИМОГИ###
    1. Вступне привітальне речення для цільової аудиторії {target_audience}
    2. Навчання повинно враховувати рівень обізнаності цільової аудиторію
    3. Завдання повинні мати 12 бальну систему оцінювання
    
    4. Складність курсу ПОВИННО поступово збільшуватись
    5. Система заохочення
    
    
    ### ПРИКЛАД ###
    Тема курсу: {topic}
    Цільова аудиторія: {target_audience}
    
    ### ПЛАН КУРСУ ###
    
""")

topic = "Python"
target_audience = "Діти"

# Передаємо дані у промпт
prompt_zero_shot_result = prompt_zero_shot.invoke({
    "topic": topic,
    "target_audience": target_audience
})
response_zero_shot = llm.invoke(prompt_zero_shot_result)

print(response_zero_shot)

print("\nРЕАЛІЗАЦІЯ СПОСОБОМ 'Few-shot'")

prompt_few_shot = PromptTemplate.from_template("""
    Ти - керівник ІТ академії
    ЗАДАЧА: написати план навчального курсу з певної теми для цільової аудиторії {target_audience}.

    ###ВИМОГИ###
    1. Вступне привітальне речення для цільової аудиторії {target_audience}.
    2. Навчання повинно враховувати рівень обізнаності цільової аудиторію.
    3. Завдання повинні мати 12 бальну систему оцінювання.
    4. Складність курсу ПОВИННО поступово збільшуватись.
    5. Наприкінці кожного кроку має бути система заохочення.

    ### ПРИКЛАД ###
    Тема курсу: Робототехніка
    Цільова аудиторія: Діти (початківці)
    План курсу:
    Привіт, юні інженери! Ласкаво просимо у світ роботів!
    Модуль 1: Знайомство з Lego-роботами. Збираємо першу машинку. 
            Практика: запуск двигуна (Оцінка: 12 балів). 
            Заохочення: стікер "Перший старт".
    Модуль 2: Програмування рухів. Вчимо робота об'їжджати перешкоди. 
            Практика: проходження лабіринту (Оцінка: 12 балів). 
            Заохочення: звання "Майстер логіки".

    ### НОВЕ ЗАВДАННЯ ###
    Тема курсу: {topic}
    Цільова аудиторія: {target_audience}
    План курсу:
    """)

topic = "Python"
target_audience = "Початківці без досвіду"

# Передаємо дані у промпт
prompt_few_shot_result = prompt_few_shot.invoke({
    "topic": topic,
    "target_audience": target_audience
})

# Передаємо створений промпт моделі
response_zero_shot = llm.invoke(prompt_zero_shot_result)

# отримуємо результат
print("--- Результат Zero-shot ---")
print(response_zero_shot)