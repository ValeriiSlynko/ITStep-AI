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
import langchain_google_genai
from langchain_core.prompts import PromptTemplate

from lesson2 import response

# завантаження даних з файлу .env
dotenv.load_dotenv()

# сам api key
api_key = os.getenv('GEMINI_API_KEY')

# створення моделі
llm = GoogleGenerativeAI(
    model='gemini-3.5-flash-lite',   # назва моделі
    api_key=api_key,
    temperature=0.7
)

print("\nРеалізація способом 'Zero-shot'")
# --- Реалізація Zero-shot ---
prompt_zero_shot = PromptTemplate.from_template("""
    ТИ - досвічений викладач.
    ЗАДАЧА: потрібно написати план навчального курсу для вказаної теми та цільової аудиторії.
    
    ### ВИМОГИ ###
    1. Курс повинен містити 7-10 занять.
    2. Складність курсу повинен починатись з легший тем і до складніших.
    3. Кожне заняття повинно мати: порядковий номер, назву, короткий зміст
    4. Матеріал курсу повинен відповідати рівню цільової аудиторії
    
    ### ПРИКЛАД ###
    Тема курсу: {topic}
    Цільова аудиторія: {target_audience}
    
    ### ПЛАН КУРСУ ###
    """)

topic = "Python"
target_audience = "Початківці"

# Передаємо дані у промпт
prompt_zero_shot_result = prompt_zero_shot.invoke({
    "topic": topic,
    "target_audience": target_audience
})

# Передаємо створений промпт моделі
response_zero_shot = llm.invoke(prompt_zero_shot_result)

print("--- Результат Zero-shot ---")
print(response_zero_shot)


print("\nРЕАЛІЗАЦІЯ СПОСОБОМ 'Few-shot'")

prompt_few_shot = PromptTemplate.from_template("""
    ТИ - досвічений викладач.
    ЗАДАЧА: потрібно написати план навчального курсу для вказаної теми та цільової аудиторії.
    
    ### ВИМОГИ ###
    1. Курс повинен містити 10-15 занять.
    2. Складність курсу повинен починатись з легших тем і до складніших.
    3. Кожне заняття повинно мати: порядковий номер, назву, короткий зміст
    4. Матеріал курсу повинен відповідати рівню цільової аудиторії
    5. Курс не може мати заплутаної та зайвої інформації
    
    ### ПРИКЛАД ###

    Тема курсу: Excel
    
    Цільова аудиторія: початківці
    
    План курсу:
    1. Числа та математичні дії
        Повторення чисел, додавання, віднімання, множення та ділення.
    
    2. Дроби
        Знайомство з простими дробами та їх використанням у повсякденному житті.
    
    3. Геометричні фігури
        Вивчення основних фігур, їхніх властивостей та вимірювання.
    
    4. Величини та одиниці вимірювання
        Довжина, маса, час, площа та робота з одиницями вимірювання.
    
    5. Математичні задачі
        Розв'язування практичних задач та розвиток логічного мислення.
    
    ### НОВЕ ЗАВДАННЯ ###
    
    Тема курсу: {topic}
    
    Цільова аудиторія: {target_audience}
    
    Створи план курсу за аналогією з прикладом.
    """)

topic = "Python"
target_audience = "Діти 8-10 років"

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