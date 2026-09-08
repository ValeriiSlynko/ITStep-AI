# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 6
import dotenv
import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)
from langchain_core.documents import Document
from uuid import uuid4

# Завдання 1
# Створіть векторну базу даних, де кожен документ – це
# вміст файлу з папки data/lesson_rag/files
#  добавте в метадані шлях до файлу (НЕ РОБИТИ, бо не вивчали)
#  створіть для кожного документу ID
#  збережіть створені ID та назви відповідних файлів в окремий json файл
# Перевірте чи працює правильно пошук

# завантажити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# serper_key = os.getenv("SERPER_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # назва моделі
    api_key=api_key,    # ключ до сервера з моделлю
)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

pc = Pinecone(api_key=pinecone_api_key)

index_name = "vs-data"  # назва бази даних

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,    # кількість чисел у векторі
        metric="cosine",   # формула для пошуку схожих текстів
        spec=ServerlessSpec(
            cloud="aws",        # хмарна платформа(амазон)
            region="us-east-1"  # регіон
        ),
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,          # база даних
    embedding=embedding   # модель для кодування
)

with open(r"D:\IT_STEP_Academy\AI\ITStep-AI\data\lesson_rag\files\future_of_ai.txt", "r", encoding = "utf-8") as future:
    text1 = future.read()
    doc1 = Document(
        page_content=text1)

with open(r"D:\IT_STEP_Academy\AI\ITStep-AI\data\lesson_rag\files\intro.txt", "r", encoding = "utf-8") as into:
    text2 = into.read()
    doc2 = Document(
        page_content=text2)

with open(r"D:\IT_STEP_Academy\AI\ITStep-AI\data\lesson_rag\files\machine_learning.txt", "r", encoding = "utf-8") as machine_learning:
    text3 = machine_learning.read()
    doc3 = Document(
        page_content=text3)

with open(r"D:\IT_STEP_Academy\AI\ITStep-AI\data\lesson_rag\files\neural_networks.txt", "r", encoding = "utf-8") as neural_networks:
    text4 = neural_networks.read()
    doc4 = Document(
        page_content=text4)

documents = [doc1, doc2, doc3, doc4]

uuids = [str(uuid4()) for _ in range(len(documents))]


# добавити документи в базу даних
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding,
)
vector_store.add_documents(
    documents=documents,
    ids=uuids
)


# Завдання 2
# На основі створеної бази даних створіть агента та
# реалізуйте його у вигляді чат-бота
@tool

def document_search(query: str):
    """
    Пошук документів у векторній базі даних

    База даних містить інформацію про ШТУЧНИЙ ІНТЕЛЕКТ
    :param query: str -- запит користувача
    :return: знайдені документи
    """
    result = vector_store.similarity_search(
        query,
        k=1
    )
    return result

agent = create_agent(
    model=llm,  # нейромережа агента
    tools=[document_search],    # список інструментів
    )

messages = [
    SystemMessage("""
    ТИ - крутий чат-бот

    ### ІНСТРУКЦІЯ ###
    1. Якщо користувач запитує про штучний інтелект то використовуй document_search
    2. Якщо інформації в документах немає то про повідомляй і нічого не вигадуй
    """)
]

while True:
    # запит від користувача
    query = input("YOU: ")

    # умова закінчення
    if query == "":
        break

    # зробити HumanMessage
    user_messages = HumanMessage(query)

    # добавляємо повідомлення в історію
    messages.append(user_messages)

    # отримати відповідь від агента
    # агент сам додає повідомлення в історію і повертає її


    # агенту треба передавати словник з ключем "messages"
    data = {
        "messages": messages,
    }

    data = agent.invoke(data)
    # агент так само повертає словник

    # дістаємо нову історію повідомлень
    message = data["messages"]

    # відповідь моделі - останнє повідомлення в історії
    response = message[-1]

    # вивести відповідь на екран
    print(response.text)

    # виведення історії
    print()
    print("-----ІСТОРІЯ-----")

    for message in messages:
        print(repr(messages))


# Завдання 3
# Внесіть зміни в декілька файлів. Змініть базу даних для цього:
#  визначте назви файлів які були змінені(вручну вказати списком в коді)
#  отримайте їхні ID
#  видаліть їх з бази даних
#  створіть нові документи та добавте в базу даних