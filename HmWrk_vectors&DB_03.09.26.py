# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 6
# Завдання 1
# Добавте в створену базу даних файл data/lesson_rag/huge_file.txt про умови користування гуглом
# Оскільки файл надто великий, то його треба добавляти частинами.
# Для цього:
# 1. прочитайте вміст файлу
# 2. розділіть його на окремі блоки(між блоками два порожніх рядка, дивись файл)
# 3. отримайте перший рядок кожного блоку – це його назва
# 4. створіть документи для кожного блоку.
# В метаданих:
# o назва файлу
# o назва блоку
# 5. створіть ID та добавте все в існуючу базу даних
# 6. добавте ID у json файл
# 7. перевірте агента
from uuid import uuid4

import dotenv
import os
import json

from langchain_core.documents import Document
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

# 1. Завантажуємо ключі з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# 2. Створюємо Модель ембеддінгів
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

# 3. Читаємо файл huge_file.txt
with open("data/lesson_rag/huge_file.txt", "r", encoding="utf-8") as f:
    text_content = f.read()

# 4. Ділимо на блоки (по двох порожніх рядках)
blocks = [b.strip() for b in text_content.split("\n\n\n") if b.strip()]

# 5. Створюємо документи з метаданими (file_name та block_title)
documents = []
for block in blocks:
    title = block.split("\n")[0].strip()
    doc = Document(
        page_content=block,
        metadata={
            "file_name": "huge_file.txt",
            "block_title": title
        }
    )
    documents.append(doc)

# 6. Генеруємо ID
uuids = [str(uuid4()) for _ in range(len(documents))]

# 7. Підключаємося до Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "vs-data"
index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding,
)

# Додаємо документи у базу
vector_store.add_documents(
    documents=documents,
    ids=uuids
)

# 8. Зберігаємо ID у JSON-файл
json_path = "data/lesson_rag/added_ids.json"

existing_ids = []
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            existing_ids = json.load(f)
        except Exception:
            existing_ids = []

all_ids = existing_ids + uuids

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_ids, f, ensure_ascii=False, indent=4)

print("Усі документи успішно додано в Pinecone, а ID збережено у JSON!")


# 1. Створюємо інструмент пошуку у Pinecone
@tool
def search_knowledge_base(query: str) -> str:
    """Шукає інформацію у векторній базі даних Pinecone за запитом користувача."""
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "Інформації за цим запитом не знайдено."

    formatted = []
    for doc in results:
        title = doc.metadata.get("block_title", "Без заголовка")
        formatted.append(f"Заголовок: {title}\nТекст: {doc.page_content}")

    return "\n\n---\n\n".join(formatted)


# 2. Створюємо мовну модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key    # ключ до сервера з моделлю
)

# 3. Створюємо агента і передаємо йому наш інструмент
tools = [search_knowledge_base]
agent = create_agent(llm, tools)

# 4. Інтерактивний чат у консолі
print("\n=== Чат з агентом запущено! (введіть 'exit' для виходу) ===")

while True:
    user_query = input("\nВи: ")
    if user_query.lower().strip() in ["exit", "quit", "вихід"]:
        print("Бувай! Чат завершено.")
        break

    if not user_query.strip():
        continue

    response = agent.invoke({"messages": [("user", user_query)]})

    # Витягуємо та красивіше виводимо текст відповіді (допомога ШІ)
    ans = response["messages"][-1].content
    if isinstance(ans, list) and len(ans) > 0 and 'text' in ans[0]:
        ans = ans[0]['text']

    print(f"\nАгент: {ans}")
