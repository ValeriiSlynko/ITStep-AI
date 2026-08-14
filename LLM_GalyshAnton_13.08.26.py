# LLM
# Large Language Model
# велика мовна модель
# завантеження api key як змінну середовища
import dotenv
import os

import langchain
from langchain_google_genai import GoogleGenerativeAI
import langchain_google_genai

# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# print(api_key)


# # модель
llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",  # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)

# # використання
# response = llm.invoke("Привіт")
# print(response)


# параметри генерації
llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",  # назва моделі
    api_key=api_key,  # ключ до сервера з моделлю
    # temperature=1.9,    # температура
    # top_p=0.8,
    # top_k=10
)
#
# response = llm.invoke("Придумай коротку історії про ельфа(до 5 речень)")
# print(response)


with open("data/lesson9/rules.txt", "r", encoding="utf-8") as text_file:
    rules = text_file.read()
instructions = f"""
        Ти адміністратор та консультант парку-атракціону.
        Надавай відповіді відвідувачам на основі правил {rules}.
        Опирайся лише на ті правила, що прописані в правилах."""

questions = []
responses = []

while True:
    question = input("My question: ")

    questions.append(question)

    # формуємо історію спілкування

    history = ""

    for old_question, old_response in zip(questions, responses):
        history += f"\nUser: {old_question}"
        history += f"\nModel: {old_response}"

    whole_text = f"""
    {instructions}

    Історія спілкування
    {history}
    User: {question}
    Model:
    """

    response = llm.invoke(whole_text)

    print(f"Відповідь: {response}")