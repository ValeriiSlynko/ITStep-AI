# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 4
# Завдання 1
# Напишіть чат-бот, який спілкується у стилі різних персонажів книг\фільмів або відомих людей.
# Ким саме бути чат-бот вирішує з повідомлення від користувача.
# Якщо персонаж або книга невідомі, то відповісти,
# що невідома інформація та запропонувати декілька відомих прикладів на вибір
import dotenv
import os

from Parser_ClsWrk_270826 import chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)
import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# завантажити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)
message: list[BaseMessage] = [
    SystemMessage("""
    ТИ - чатбот, який імітує спілкування різних персонажів 30 книг, фільмів, казок,
    Твоя задача - надавати відповіді користувачу у відповідному стилі в залежності від персонажів

    ### ІНСТРУКЦІЯ ###
    1. Відповіді надавай чітко і коротко (до 2 речень)
    2. Стиль відповіді вибирай залежно від запиту користувача.
    3. Якщо персонаж або книги невідомі, то відповісти що невідома інформація і запропонувати декілька відомих прикладів
    """)
]

while True:
    user_text = input("User: ")

    if user_text == "":
        break

    user_message = HumanMessage(content=user_text)

    message.append(user_message)

    response = llm.invoke(message)

    print(f"AI: {response.content[0]['text']}")

    message.append(response)

# Завдання 2
# Напишіть чат-бот, який дає відповіді на питання стосовно умов повернення товару.
# Якщо користувач запитує щось інше, то відповідати, що немає інформації.
# Застосуйте обмеження історії(можна десь 5 повідомлень)

with open("data/lesson9/return_policy.txt", "r", encoding="utf-8") as file:
    rules = file.read()

messages = [
    SystemMessage(f"""
    Ти - консультант магазину
    Твоя задача: надавати відповіді на питання стосовно умов повернення товару

    ### ІНСТРУКЦІЇ ###
    1. Відповіді мають бути короткими (до 2 речень)
    2. Якщо користувач запитує щось не по темі, то відповідь: Такої інформації немає
    3. Якщо правилами не регламентовано, то про це   повідом і НЕ ВИГАДУЙ
    4. Відповідай за правилами без вигадки.

    ### ПРАВИЛА ПОВЕРНЕННЯ ###
    {rules}
    """)
]

# створення тримера повідомлень
trimmer = trim_messages(
    strategy='last',  # залишати останні повідомлення

    token_counter=len,  # рахуємо кількість повідомлень
    max_tokens=5,  # залишати максимум 5 повідомлення(System, AI, Human)

    start_on='human',  # історія завжди починатиметься з HumanMessage
    end_on='human',  # історія завжди закінчуватиметься з HumanMessage
    include_system=True  # SystemMessage не чіпати
)

while True:
    user_text = input("User: ")

    if user_text == "":
        break

    human_message = HumanMessage(user_text)

    messages.append(human_message)

    messages = trimmer.invoke(messages)

    response = llm.invoke(messages)

    print(f"AI: {response.content[0]['text']}")

    messages.append(response)

# Завдання 3
# Напишіть чат-бот, який допомагає у вивченні англійської мови з наступним функціоналом:
#  якщо користувач просить перекласти слово або фразу, то дається переклад слова та приклад використання в реченні
#  якщо користувач просить перекласти речення, то дається переклад самого речення,
# а також пояснення граматики, наприклад структура there is\are, питання в різних часових формах тощо.
# Приклади реалізуйте як HumanMessage та AIMessage

messages: list[BaseMessage] = [
    SystemMessage(r"""
    Ти - досвічений вчитель англійської мови
    Твоя задача: допомагати вивчати англійську

    ###ІНСТРУКІЇ###
    1. Якщо користувач просить перекласти слово або фразу то дається переклад слова та приклад використання в реченні
    2. якщо користувач просить перекласти речення, то дається переклад самого речення
    3. Пояснюй граматику, наприклад структура there is\are, питання в різних часових формах, тощо.

"""),
    HumanMessage("Переклади слово Стіл"),
    AIMessage("""Переклад вашого слова стіл - table.
            Ось приклад в реченні: На столі лежить книга - There is a book on the table.
            """)
]

while True:
    user_text = input("User: ")

    if user_text == " ":
        print("Розмову закінчено!")
        break

    human_message = HumanMessage(user_text)

    messages.append(human_message)

    response = llm.invoke(messages)

    data = {
        "text": response.text
    }

    eng_works = chain.invoke(data)

    print(f"AI: {response.text}")
    print(eng_works)

    messages.append(response)


# Завдання 4
# Модифікуйте попереднє завдання таким чином, щоб в SystemMessage передавався список вивчених слів користувачем.
# Для цього напишіть окрему модель яка буде діставати з відповіді(AIMessage) усі англійські слова
# (вважаємо що користувач знає лише ті слова, про які йому сказала модель).
# Список вивчених слів треба зберігати в json файлі та відвантажувати при запуску програми.
# Змініть функціонал таким чином:
#  якщо користувач просить перекласти слово або фразу, то дається переклад слова та приклад використання в
# реченні з вивченими словами
#  якщо користувач просить перекласти речення, то додатково пояснюється значення невідомих слів

