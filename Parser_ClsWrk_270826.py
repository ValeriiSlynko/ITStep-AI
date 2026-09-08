import dotenv
import os

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# завантажити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",   # назва моделі
    api_key=api_key  # ключ до сервера з моделлю
)

# Завдання 4
# Модифікуйте попереднє завдання таким чином, щоб в SystemMessage передавався список вивчених слів користувачем.
# Для цього напишіть окрему модель яка буде діставати з відповіді(AIMessage) усі англійські слова
# (вважаємо що користувач знає лише ті слова, про які йому сказала модель).
# Список вивчених слів треба зберігати в json файлі та відвантажувати при запуску програми.
# Змініть функціонал таким чином:
#  якщо користувач просить перекласти слово або фразу, то дається переклад слова та приклад використання в
# реченні з вивченими словами
#  якщо користувач просить перекласти речення, то додатково пояснюється значення невідомих слів

class Response(BaseModel):
    words: list[str] = Field(description="Список унікальних англійський слів")


parser = PydanticOutputParser(pydantic_object=Response)

instructions = parser.get_format_instructions()

prompt = PromptTemplate.from_template("""
    Твоя задача дістати всі унікальні англійські слова з тексту.

    ### ІНСТРУКЦІЇ ###
    1. Ігноруй артиклі.
    2. Ігноруй стоп слова (am, are, is, she, he, it тощо)

    ### ФОРМАТ ВІДПОВІДІ ###
    {format_instructions}

    ### ВХІДНІ ДАНІ ###
    {text}
""",
    partial_variables={"format_instructions": instructions},
)

chain = prompt | llm | parser

if __name__ == "__main__":

    text = """
    I am reading a book. — Я читаю книгу.
    She bought a new book yesterday. — Вона вчора купила нову книгу.
    I want to book a hotel room. — Я хочу забронювати номер у готелі.
    We booked a flight to London. — Ми забронювали рейс до Лондона.
    """

    data = {
        "text": text
    }

    response = chain.invoke(data)

    print(response)