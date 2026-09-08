# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 5

# Завдання 1
# Напишіть чат-бот, з інструментом по рекомендації ресторанів.
# Для цього скористайтесь GoogleSerperAPIWrapper(type="places")
# Інструмент повинен отримувати запит для пошуку та повертати таку інформацію про ресторани:
#  назва
#  посилання на сайт(якщо є)
#  рейтинг
# Більш детально дивись документацію

import os
import dotenv

from langgraph.prebuilt import create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

# 1. Завантажуємо дані з файлу .env (тобто завантаження ключів)
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")

# 2. Ініціалізація моделі (налаштовуємо модель ШІ)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key,    # ключ до сервера з моделлю
    temperature=0       # регулятор креативу (від 0 до 9)
)


# 3. Створення інструменту (Tool) для пошуку ресторанів
@tool
def restaurant_search_tool(query: str) -> str:
    """
    Шукає ресторани за допомогою Google Serper API.
    Передається запит з назвою ресторану або кухнею країни.

    :param query:
    :return:
    """

    # використовуємо type="places" для пошуку на картах
    search = GoogleSerperAPIWrapper(serper_api_key=serper_key, type="places")
    result = search.results(query)
    places = result.get("places")

    if not places:
        return "Ресторанів за вашим запитом не знайдено."

    formatted_results = []

    for place in places:
        website = place.get("website")

        # якщо заклад немає веб-сайт - цей рядок ігнорує його і переходить до наступного
        if not website:
            continue

        name = place.get("title", "Немає назви")
        rating = place.get("rating", "Немає рейтингу")

        # форматуємо рядок на вимогу: назва, рейтинг, сайт
        formatted_results.append(f"Назва: {name} \nРейтинг: {rating} \nСайтЖ: {website}")

        # Обмежуємо найкращих ресторанів із сайтами до 5
        if len(formatted_results) > 5:
            break

    if not formatted_results:
        return "Знайдено заклади, але жоден з них не має офіційного вебсайту."

    return "\n\t".join(formatted_results)

# 4. Створюємо агента і даємо йому наш інструмент
tools = [restaurant_search_tool]
agent_executor = create_react_agent(llm, tools)

# 5. Запускаємо нескінченний цикл чату
print("БОТ рекомендацій ГОТОВИЙ! (Напишіть Вихід для завершення")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["Вихід", "exit", "quit"]:
        print("До побачення. \n\tХай щастить!")
        break

    # відпрацьовуємо запит користувача агенту
    response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})

    # виводимо відповідь БОТАна на екран
    print(f"\n Бот: {response['messages'][-1].content}")

















