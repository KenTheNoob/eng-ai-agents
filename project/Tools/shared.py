# This file contains shared functions used by multiple files
import os
import sys

import pymongo
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from qdrant_client import QdrantClient


# Unused since usage limit reached since years ago...rip
def getOpenAiModel():
    MODEL = "gpt-3.5-turbo"
    try:
        load_dotenv()
    except Exception:
        load_dotenv(sys.path[1] + "/.env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)


# Create a mongoDB connection
def getMongoClient():
    try:
        load_dotenv()
    except Exception:
        load_dotenv(sys.path[1] + "/.env")
    DATABASE_HOST = os.getenv("DATABASE_HOST")
    return pymongo.MongoClient(DATABASE_HOST)


# Create a qdrant connection
def getQdrantClient():
    try:
        load_dotenv()
    except Exception:
        load_dotenv(sys.path[1] + "/.env")
    USE_QDRANT_CLOUD = os.getenv("USE_QDRANT_CLOUD")
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
    if USE_QDRANT_CLOUD:
        return QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_APIKEY)
    else:
        return QdrantClient(url=QDRANT_CLOUD_URL)


# Setup the text embedder
def getEmbeddingsModel(MODEL="llama3.2"):
    # Make sure you run ollama serve first
    return OllamaEmbeddings(model=MODEL)


def setupClearML():
    try:
        load_dotenv()
    except Exception:
        load_dotenv(sys.path[1] + "/.env")
    CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST")
    CLEARML_API_HOST = os.getenv("CLEARML_API_HOST")
    CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST")
    CLEARML_API_ACCESS_KEY = os.getenv("CLEARML_API_ACCESS_KEY")
    CLEARML_API_SECRET_KEY = os.getenv("CLEARML_API_SECRETKEY")
    return (
        CLEARML_WEB_HOST,
        CLEARML_API_HOST,
        CLEARML_FILES_HOST,
        CLEARML_API_ACCESS_KEY,
        CLEARML_API_SECRET_KEY,
    )
