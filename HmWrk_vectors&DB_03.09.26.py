# Курс: AI+Python
# Модуль 3. Generative AI, LLM
# Тема: Langchain. Частина 6
# Завдання 1
# Добавте в створену базу даних файл data/lesson_rag/huge_file.txt про умови користування гуглом
# Оскільки файл надто великий, то його треба добавляти частинами.
# Для цього:
#  прочитайте вміст файлу
#  розділіть його на окремі блоки(між блоками два порожніх рядка, дивись файл)
#  отримайте перший рядок кожного блоку – це його назва
#  створіть документи для кожного блоку.
# В метаданих:
# o назва файлу
# o назва блоку
#  створіть ID та добавте все в існуючу базу даних
#  добавте ID у json файл
#  перевірте агента
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

