# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 5
import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)


# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")

# модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key    # ключ до сервера з моделлю
)

search = GoogleSerperAPIWrapper(serper_api_key=serper_key)


# Завдання 1
# Напишіть функцію яка перевіряє складність паролю:
#  кількість символів(>8)
#  наявність хоча б однієї літери\цифри\спеціального символу
#  наявність літер в різних регістрах
# Функція повертає тест з описом паролю (що добре, а що погано) 
# На основі цієї функції створіть агента.

def count_char(password: str) -> int:
    total_alpha = 0
    total_digit = 0
    total_special = 0
    total_upper = 0
    total_lower = 0
    
    for char in password:
        if char.isalpha():
            total_alpha += 1

        elif char.isdigit():
            total_digit += 1

        else:
            total_special += 1

        if char.isupper():
            total_upper += 1

        if char.lower():
            total_lower += 1

    return total_alpha, total_digit, total_special, total_upper, total_lower

@tool
def password_check(password: str):

    """
    Перевірка складності пароля
    :param password: str -- пароль
    :param total_special:
    :return:
    """

    if len(password) < 8:
        return"В паролі має бути більше 8 символів"

    total_alpha, total_digit, total_special, total_upper, total_lower = count_char(password)

    if total_alpha == 0:
        return "В паролі мають бути літери"

    if total_digit == 0:
        return "МАють бути цифри"

    if total_special == 0:
        return "МАють бути спеціальні символи"

    if total_upper == 0:
        return "Має бути ВЕЛИКА літера"

    if total_lower == 0:
        return "Має бути мала літера"

    else:
        "Пароль чудесний :)"


@tool
def search_person(name: str)-> str:
    """
    Отримує ім'я людини та шукає в інтернеті інформацію про неї
    :param name: str - ім'я та прізвище
    :return: інформація про людину
    """
    info = search.results(f"Новини про {name}")
    return info


agent = create_agent(
    model=llm,
    tools=[password_check, search_person],
    )

messages = [
    SystemMessage("""
    ТИ - крутий чат-бот
    """)
]

while True:
    query = input("YOU: ")

    if query == "":
        break

    user_messages = HumanMessage(query)

    messages.append(user_messages)

    data = {
        "messages": messages
    }

    data = agent.invoke(data)

    messages = data["messages"]

    response = messages[-1]

    print(response.text)


# Завдання 3
# Напишіть модель яка конвертує одну валюту в іншу за нинішнім курсом.
# Для цього напишіть функції, яка отримує номінал та курс і робить конвертацію.
# Реалізуйте 2 ланцюга:
#  перший отримує назви валют та шукає курс в інтернеті
#  другий отримує номінал та курс і застосовує функцію конвертації



# Завдання 4
# Напишіть модель яка рекомендує міста для проведення
# вихідних. Користувач вводить назву країни та стиль відпочинку.
# Перший агент шукає популярні міста для відпочинку в потрібному стилі.
# Другий агент перевіряє погоду в цих містах та відсіює невдалі варіанти
# Третій агент виводить кожне місто що залишилось, та причину чому його варто відвідати(коротко)