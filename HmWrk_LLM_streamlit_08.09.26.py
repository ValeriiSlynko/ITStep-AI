# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 7
# Завдання 1
# Напишіть додаток з чат ботом по допомозі з вивченням англійської мови.
#  Якщо користувач просить перекласти слово або фразу, то вивести переклад та приклад використання у реченні
#  Якщо користувач просить перекласти речення, то вивести переклад та пояснення граматики,
# наприклад: структура there is/are, пасивна форма дієслова тощо

# / Завантажуємо бібліотеки /
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

# 1. Отримаємо API ключ
api_key = st.secrets["GEMINI_API_KEY"]

# 2. Ініціалізація/створення моделі
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",   # назва моделі
    api_key=api_key     # ключ до сервера з моделлю
)
# 3. Називаємо чат-бот для користування
st.title("Вивчаємо англійську мову з досвіченим **АІ** вчителем")

# 3. Ініціалізація історії (написання ПРОМПТ)
if 'history' not in st.session_state:
    st.session_state['history'] = [
        SystemMessage(
            content="""
            Ти - досвічений та приязний вчитель англійської мови.
            Твоя задача: допомагати користувачеві вивчати англійську мову.
            
            ### Правила відповіді
            1. Якщо користувач просить перекласти окреме слово або коротку фразу:
                - Дай точний переклад.
                - Наведи 1-2 яскравих приклади використання цього слова/фрази в реченні з перекладом українською.
            2. Якщо користувач просить перекласти ціле речення:
                - Дай точний переклад речення.
                - Поясни граматичні конструкції, використані в ньому (наприклад: структура there is/are, пасивна форма дієслова тощо)
            """
        ),
        # приклад 1: Переклад окремого слова
        HumanMessage(content="Переклади слово Стіл"),
        AIMessage(
            content="Переклад слова 'Стіл' - 'table' \n\n"
                    "Приклад у реченні:\n"
                    "'There is a book on the table' - На столі лежить книга."
        ),

        # приклад 2: Переклад речення та слова в контексті речення
        HumanMessage(content="Переклади речення: Він читає книгу щодня"),
        AIMessage(
            content="Переклад слова 'She reads a book every day'\n\n"
                    "**Граматичне пояснення:**\n"
                    "Використано час *Present Simple* (теперішній простий), оскільки мова йде про регулярну дію.\n"
                    "До дієслова *read* додано закінчення **-s** (*reads*), тому що підмет - 3-тя особа однини (**He**)."
        )
    ]

# 5. Поле введення від користувача
user_query = st.chat_input("Введіть слово або речення для перекладу!")

if user_query:
    # додаємо повідомлення користувача до історії
    human_message = HumanMessage(content=user_query)
    st.session_state['history'].append(human_message)

    # викликаємо модель з усією історією (вкл.промпт та приклади)
    response = llm.invoke(st.session_state['history'])

    # зберігаємо відповідь від АІ
    st.session_state['history'].append(response)

# 5. Відображення історії діалогу
for message in st.session_state['history']:
    # пропускаємо системні інструкції
    if isinstance(message, SystemMessage):
        continue

    # отримати вміст
    text = message.text

    # визначення ролі та імені
    if isinstance(message, HumanMessage):
        role = "User"
        name = "You"
    else:
        role = "Assistant"
        name = "Teacher"

    # отримання чистого тексту (

    # виводимо в інтерфейс чату/переписки
    with st.chat_message(role):
        st.write(f"**{name}:**")
        st.markdown(text)























