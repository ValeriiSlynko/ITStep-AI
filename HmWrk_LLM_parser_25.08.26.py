# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 3

# Завдання 1
# Напишіть модель для генерації персонального плану тренувань з двох ланцюгів:
#  Перший ланцюг отримує мету тренування (схуднення, набір м’язів тощо) та повертає список вправ
#  Другий ланцюг отримує список вправ, рівень підготовки користувача (низький, середній, професіонал)
# та кількість часу на тиждень(в годинах) і повертає план тренувань

import dotenv
import os

import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# завантажити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# будуємо модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",       # назва моделі
    api_key=api_key       # ключ до сервера з моделлю
)

# ЛАНЦЮГ 1: ОТРИМУЄ МЕТУ -> ПОВЕРТАЄ СПИСОК ВПРАВ
# Будуємо модель для першого кроку. LLM повертає список рядків (list[str])
class Training(BaseModel):
    exercises: list[str] = Field(description="Список вправ для виконання")

parser1 = PydanticOutputParser(pydantic_object=Training)

prompt1 = PromptTemplate.from_template("""
    ТИ - професійний фітнес-тренер.
    Твоя задача: підібрати курс - список ефективних тренувальних вправ під конкретну мету користувача.
    
    ### ФОРМАТ ВІДПОВІДІ ###
    {format_instruction}
    
    ### ВХІДНІ ДАНІ ###
    Мета тренування: {goal}
""",
    partial_variables={"format_instruction":parser1.get_format_instructions()}
)

# Збираємо 1-й ланцюг
chain1 = prompt1 | llm | parser1

# ДАні для тестування 1-го ланцюга

user_goal = "Набір м'язів (акцент на руки, груди, спина)"

print("\n--- Запуск 1-го ланцюга ---")
response1 = chain1.invoke({"goal": user_goal})

# Отримуємо список вправ із моделі (Pydantic)
generated_exercises = response1.exercises
print(f"Підібрані вправи згідно запиту/бажань: {generated_exercises}")

# ЛАНЦЮГ 2: ВПРАВИ + РІВЕНЬ + ЧАС -> ПОВЕРТАЄ ГОТОВИЙ ПЛАН
# Модель для другого кроку. Повертає один готовий текст плану
class TrainingPlan(BaseModel):
    content: str = Field(description="ДЕтальний розклад вправ/тренувань по днях")

parser2 = PydanticOutputParser(pydantic_object=TrainingPlan)

prompt2 = PromptTemplate.from_template("""
    ТИ - головний тренер фітнес-клубу
    Твоя задача: взяти список вправ і розробити персональний тижневий план тренувань.
                Враховуй рівень фізичної підготовки та вільний час
    
    ### ІНСТРУКЦІЇ ###
    1. Рівень фізичної підготовки визначає складність виконання та кількість підходів/повторень.
    2. Загальний час усіх тренувань на тиждень = ліміту годин
    3. Рівномірно розподіли вправи по днях тижня.
    
    ### ФОРМАТ ВІДПОВІДІ ###
    {format_instruction}
    
    ### ВХІДНІ ДАНІ ###
    Список вправ: {exercises}
    Рівень підготовки користувача: {level}
    Кількість годин на тиждень: {hours} годин
""",
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

# Збираємо 2-й ланцюг
chain2 = prompt2 | llm | parser2

# Дані користувача
user_level = "Початковий"
user_hours = "5"

print("\n--- Запуск Ланцюга 2 ---")

# Передаємо список вправ з 1-го ланцюга та додаткові параметри
data_for_plan = {
    "exercises": generated_exercises,
    "level": user_level,
    "hours": user_hours,
}

response2 = chain2.invoke(data_for_plan)

print("\n===== ПЕРСОНАЛЬНИЙ ПЛАН ТРЕНУВАНЬ =====")
print(response2.content)





