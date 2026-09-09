import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 7
# Завдання 1
# Напишіть додаток, який симулює спілкування з певною відомою людиною.
# З ким саме спілкуватись вводить користувач через st.text_input()
api_key = st.secrets["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",   # назва моделі
    api_key=api_key     # ключ до сервера з моделлю
)
st.title("The author of the chatbot is Valeriy Slynko")

user_query = st.chat_input("Ваше повідомлення")

person = st.text_input("Вкажіть Ім'я Прізвище з ким ви хочете поговорити")

# якщо це початок, то створити історію в session state
if 'history' not in st.session_state and person is not None:
    # історія повідомлень
    st.session_state['history'] = [
        # перше повідомлення з основними інструкціями(промпт)
        SystemMessage(
            f"""
            Ти -- {person} вихований, ерудований та об'єктивний чат-бот. 
            Твоя задача: давати зрозумілі повні відповіді на питання.
            """
        )
    ]

if user_query:
    # переводимо повідомлення в HumanMessage
    human_message = HumanMessage(user_query)

    # добавляємо до історії повідомлень
    st.session_state['history'].append(human_message)

    # запускаємо модель
    response = llm.invoke(st.session_state['history'])

    # response -- AIMessage
    # добавляємо до історії повідомлень
    st.session_state['history'].append(response)

    # вивести всю історію спілкування
    for message in st.session_state['history']:
        # пропускаємо SystemMessage
        if isinstance(message, SystemMessage):
            continue

        # отримати вміст
        text = message.text

        # отримати роль
        if isinstance(message, HumanMessage):
            role = "Human"
        else:
            role = 'AI'

        # вивести повідомлення з підписом
        with st.chat_message(role):
            st.markdown(text)


# Завдання 2
# Напишіть додаток, який симулює проходження співбесіди на певну посаду.
# Користувач може ввести назву посади через st.text_input()
# Користувач може ввести опис вакансії через st.file_uploader()
# Далі починається чат із спілкуванням



# Завдання 3
# Напишіть чат бота з доступом до інтернету