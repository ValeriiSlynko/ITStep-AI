# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 3
# Завдання 1
# Напишіть модель для рекомендації книг з двох ланцюгів:
#  Перший ланцюг отримує назву книги та визначає її жанр
#  Другий отримує назву книги, жанр та повертає список схожих книг (того ж самого жанру та іншого)

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

# ПЕРШИЙ ланцюг отримує назву книги та визначає її жанр
class GenreBooks(BaseModel):
    genre: str = Field(description="Жанр книги")

# створюємо парсер
parser = PydanticOutputParser(pydantic_object=GenreBooks)

instructions = parser.get_format_instructions()

# print(instructions)

prompt_genre = PromptTemplate.from_template("""
    Ти - чатбот-бібліотекар.
    Твоя задача визначити жанр книги.

    ###ІНСТРУКЦІЇ###
    Відповідь має бути до 2-3 слів

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instruction}

    ### ВХІДНІ ДАНІ ###
    {book}

""",
    partial_variables={"format_instruction":instructions}
    )

# ланцюг першого кроку
chain1 = prompt_genre | llm | parser

book_name = "Володар перснів"

data_book = {
    "book": book_name,
}
response = chain1.invoke(data_book)

print(response.genre)

# ДРУГИЙ отримує назву книги, жанр та повертає список схожих книг (того ж самого жанру та іншого)
class Recommendations(BaseModel):
    recommend_same: list[str] = Field(description="Список цікавих книг за визначеним жанром ")
    recommend_other: list[str] = Field(description="Список схожих книг але іншого жанру")

# створюємо парсер
parser = PydanticOutputParser(pydantic_object=GenreBooks)

instructions = parser.get_format_instructions()

# print(instructions)

prompt_genre = PromptTemplate.from_template("""
    Ти - чатбот-бібліотекар.
    Твоя задача: підібрати список 3-4 схожих книг беручи за основу їх  назву та їх жанр.

    ###ІНСТРУКЦІЇ###
    Відповідь має бути до 2-3 слів

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instruction}

    ### ВХІДНІ ДАНІ ###
    Назва книги: {book_name} | Жанр: {genre}

""",
    partial_variables={"format_instruction": instructions}
)

# ланцюг першого кроку
chain2 = prompt_genre | llm | parser

user_book_question = "Кобзар"

data_book = {
    "book_name": user_book_question
    }
response1 = chain1.invoke(data_book)

print(f"Відповідь книги по жанруЖ: {response.genre}")

data_book = {
    "book_name": user_book_question,
    "genre": response1.genre
    }

response2 = chain2.invoke(data_book)

print(f"Рекомендації книг: {response2}")


# Завдання 2
# Напишіть модель для генерації листа:
#  Перший ланцюг отримує короткий опис листа та генерує основний зміст
#  Другий ланцюг отримує основний зміст та стиль листа (формальний, неформальний тощо) та генерує лист

# ПЕРШИЙ ЛАНЦЮГ отримує короткий опис листа та генерує основний зміст
class GetLetter (BaseModel):
    content: str = Field(description="Основний зміст листа без зайвих привітань")

# створюємо парсер
parser1 = PydanticOutputParser(pydantic_object=GetLetter)

prompt_letter = PromptTemplate.from_template("""
    Ти - чатбот автор написання листів.
    Твоя задача: генерувати та написати основний зміст листа на основі короткого опису

    ###ІНСТРУКЦІЇ###
    Лист має бути написано із зверненням до особи, посадовця, організації тощо

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instruction}

    ### ВХІДНІ ДАНІ ###
    {letter_description}

""",
    partial_variables={"format_instruction": parser1.get_format_instructions()}
    )

# ланцюг першого кроку
chain1 = prompt_letter | llm | parser1

text_letter = ("1.Потрібно вибити борги із податкової служби бо їх накопилось цілих 20грн."
            "\n2. Обов'язково пригрози облиттям холодною водою з цистерни"
            "\n3. Нагадай завдяки кому їхня організація ще працює"
            "\n4. Напиши про строки погашення 24 години")


response1 = chain1.invoke({"letter_description": text_letter})

generated_content = response1.content
print(f"Згенерований зміст листа: {generated_content}\n")


# ДРУГИЙ ЛАНЦЮГ отримує основний зміст та стиль листа (формальний, неформальний тощо) та генерує лист
class FinalLetter(BaseModel):
    styled_letter: str = Field(description="Готовий текст листа у заданому стилі")

# створюємо парсер
parser2 = PydanticOutputParser(pydantic_object=FinalLetter)

prompt_letters = PromptTemplate.from_template("""
    Ти - професійний автор-редактор листів
    Твоя задача: взяти основний зміст і перетворити його на повноцінний лист у заданому стилі ({style})

    ###ІНСТРУКЦІЇ###
    1. Лист має бути не менше 3 речень
    2. Обов'язкове звернення до адресата
    3. Якщо лист ФОРМАЛЬНИЙ: форма звернення поважлива, офіційний тон, юридично грамотні формулювання
    4. Якщо лист не НЕФОРМАЛЬНИЙ: більш розмовний, дружній та іронічний тон 

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instruction}

    ### ВХІДНІ ДАНІ ###
    Основний зміст: {main_content}
""",
    partial_variables={"format_instruction": parser2.get_format_instructions()}
    )

# ланцюг першого кроку
chain2 = prompt_letters | llm | parser2

print("Оберіть стиль листа:")
print("1 - ФОРМАЛЬНИЙ")
print("2 - НЕФОРМАЛЬНИЙ")

user_choice = input("Введіть номер стилю (1 або 2): ")

# Визначаємо, який стиль передати в LLM на основі вибору
if user_choice == "1":
    chosen_style: str = "ФОРМАЛЬНИЙ"
elif user_choice == "2":
    chosen_style: str = "НЕФОРМАЛЬНИЙ"

print(f"\n---Запуск 2-го ланцюга для стилю: {chosen_style} ---\n")

# Передаємо стиль обраний користувачем
data_letter = {
    "main_content": generated_content,
    "style": chosen_style,
}

response_final = chain2.invoke(data_letter)

print("\nРезультат генерації:")
print(response_final.styled_letter)

# Завдання 3
# Напишіть модель для генерації резюме:
#  Перший ланцюг отримує опис вакансії та повертає основні навички, які необхідні
#  Другий ланцюг отримує основні навички та опис кандидата і генерує резюме

class Skills(BaseModel):
    experience: float = Field(description="Досвід роботи(роки)")
    english_level: str = Field(description="Рівень англійської мови")
    frameworks: list[str] = Field(description="Список бібліотек")
    technologies: list[str] = Field(description="Список технологій")
    lang_programming: str = Field(description="Мова програмування")

parser = PydanticOutputParser(pydantic_object=Skills)

instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template("""
    ТИ - досвічений рекрутер
    Тобі надається опис вакансії з яких потрібно повернути основні навички

    ### ФОРМАТ ВІДПОВІДІ ###
    {instructions}

    ### ВХІДНІ ДАНІ ###
    Опис вакансії: {vacancy_description}

""", partial_variables={"instructions":instructions}
    )

chain3 = prompt_genre | llm | parser

vacancy = """
    Are you a Data Scientist with a love of LLMs, generative AI?

We are looking for a passionate Data Scientist to implement AI solutions aimed at achieving business goals.

This role offers the opportunity to work on cutting-edge AI adoption projects that helps to improve current business processes.

    You'll be a great fit if you have:
Strong Python Experience (2 year +);
Experience with LLM , Diffusion models;
Knowledge of Prompt engineering;
Experience with Gen AI-related technologies such as LangChain and RAG;
Experience with Neural Networks (Optional) ;
Experience with NLP , Predictive analytics and Machine learning;
Experience with Pandas;
Experience with SQL, including experience with large datasets;
Strong experience in statistics;
Bachelor's degree in Computer Science or a related field.
What you'll do:
Develop AI agents that utilize LLM, RAG and langchain approach;
Implement LLM and Diffusion models to boost business productivity;
Utilize LLM (LLM Vision) to improve object detection, text classification and extraction;
Create forecasting, recommendation, and classification models;
Transform business challenges to AI applications.

    We ensure your growth with:
Competitive salary fixed in USD;
Flexible working schedule and fully remote work format;
Paid vacation days and sick leave days ;
Personal and professional development opportunities;
Participation in building innovative projects from scratch using modern technologies;
Team-building activities and corporate events;
English classes and educational events.
"""

data ={
    "vacancy_description": vacancy,
}

response = chain3.invoke(data)
print(response)
