# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 4

# Завдання 1
# Напишіть чат модель яка підсумовує всю розмову в декілька речень.
# Вкажіть, щоб модель зберігала якомога більше деталей.
# Використайте цю модель для простого чат-бота, який замість trim_massages використовує модель з підсумуванням.
# Підсумовуйте повідомлення, коли їх більше 4-х.
# Старі повідомлення треба видалити
# НЕ ВИДАЛЯТИ SystemMessage та не використовувати його для підсумування

import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Завантажити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Основна модель для чату
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)

# Початкова історія з SystemMessage
messages: list[BaseMessage] = [
    SystemMessage("""
    ТИ - чатбот-співбесідник, який допомагає отримати цікаву інформацію 
    Твоя задача - надавати чіткі відповіді українською мовою користувачу у відповідному стилі в залежності

    ### ІНСТРУКЦІЯ ###
    1. Відповіді надавай чітко і коротко (2-3 речення)
    2. Стиль відповіді вибирай залежно від запиту користувача.
    3. Зберігай якомога більше деталей 
    4. Підсумовуй більше 4-х повідомлень
    """)
]

while True:
    user_text = input("User: ")

    if user_text.strip() == "":
        break

    # Додаємо повідомлення користувача в історію
    messages.append(HumanMessage(content=user_text))

    # ЛОГІКА ПІДСУМОВУВАННЯ (ЗБЕРЕЖЕННЯ ПАМ'ЯТІ)

    # Рахуємо повідомлення без урахування SystemMessage (індекс 0)
    chat_history_count = len(messages) -1

    # Якщо повідомлень більше 4-х -> підсумовуємо
    if chat_history_count > 4:
        print("[\nЧат-бот генератор: Повідомлень вже більше 4-х]")

        history_summary = messages[1:]

        summary_prompt = [
            SystemMessage(content="""
                ТИ - архіватор зі стажем.
                Твоя задача: зробити чіткий підсумок діалогу у декілька речень.
                Обов'язково збережи якомога більше важливих деталей, фактів, імен та цифр, які згадувалися.
            """),
            # Передаємо накопичену історію
            history_summary,
            HumanMessage(content="Зроби підсумок усієї історії діалогу вищ з максимальним збереженням деталей")
        ]

        # Викликаємо модель для генерації підсумку
        summary_response = llm.invoke(summary_prompt)
        summary_text = summary_response.content
        print(f"[Система: Новий підсумок історії: {summary_text}]\n")

        # Очищаємо старі повідомлення: залишаємо SystemMessage
        # і додаємо підсумок розмови як ОДНЕ ПОВІДОМЛЕННЯ від користувача
        messages = [messages[0], SystemMessage(f"Короткий зміст попередньої розмови: {summary_text}"),
                    HumanMessage(content=user_text)]

        # Після очищення додаємо поточне повідомлення користувача знову, щоб бот на нього відповів

    # 2. Отримуємо відповідь від бота на основі поточної (або вже стиснутої) історії
    response = llm.invoke(messages)

    # Виправляємо виведення тексту відповіді
    print(f"AI: {response.content}")

    # 3. Додаємо відповідь бота в історію
    messages.append(response)
    print(f"(Повідомлень у пам'яті: {len(messages) - 1})\n")