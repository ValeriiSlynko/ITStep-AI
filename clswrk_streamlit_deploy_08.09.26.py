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


# Ініціалізація моделі
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",   # назва моделі
    api_key=api_key     # ключ до сервера з моделлю
)
st.title("The author of the chatbot is Valeriy Slynko")

# 1. Спочатку зчитуємо персонажа
person = st.text_input("Вкажіть Ім'я Прізвище з ким ви хочете поговорити", value="Альберт Ейнштейн")

# Ініціалізація або оновлення історії при зміні персонажа
if 'history' not in st.session_state or st.session_state.get('current_person') != person:
    st.session_state['current_person'] = person
    st.session_state['history'] = [
        SystemMessage(
            content=f"""
            Ти — {person}. Поводься, розмовляй та відповідай як {person}. 
            Ти вихований, ерудований та об'єктивний. 
            Твоя задача: давати зрозумілі та повні відповіді на питання в обраному образі.
            """
        )
    ]

# 2. Потім приймаємо повідомлення від користувача
user_query = st.chat_input("Ваше повідомлення")

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

    # 3. Відображення всієї історії спілкування
    for message in st.session_state['history']:
        # Пропускаємо системні інструкції
        if isinstance(message, SystemMessage):
            continue

        # отримати вміст
        text = message.text

        # Визначення ролі для аватарки та підпису
        if isinstance(message, HumanMessage):
            role = "user"
            name = "Ви"
        else:
            role = "assistant"
            name = person

        # Вивід повідомлення
        with st.chat_message(role):
            st.write(f"**{name}:**")
            st.markdown(message.content)


# Завдання 2
# Напишіть додаток, який симулює проходження співбесіди на певну посаду.
# Користувач може ввести назву посади через st.text_input()
# Користувач може ввести опис вакансії через st.file_uploader()
# Далі починається чат із спілкуванням



# Завдання 3
# Напишіть чат бота з доступом до інтернету