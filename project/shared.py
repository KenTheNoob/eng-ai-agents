# This file contains shared functions used by multiple files
import os
import sys

import pymongo
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.llms import Ollama


# Unused since usage limit reached since years ago...rip
def getOpenAiModel():
    MODEL = "gpt-3.5-turbo"
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)


# Create a mongoDB connection
def getMongoClient():
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    DATABASE_HOST = os.getenv("DATABASE_HOST")
    return pymongo.MongoClient(DATABASE_HOST)


# Create a qdrant connection
def getQdrantClient():
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    USE_QDRANT_CLOUD = os.getenv("USE_QDRANT_CLOUD")
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
    if USE_QDRANT_CLOUD=="True":
        return QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_APIKEY)
    else:
        return QdrantClient(url=QDRANT_CLOUD_URL)


# Setup the text embedder
def getEmbeddingsModel(MODEL="llama3.2"):
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    USE_DOCKER = os.getenv("USE_DOCKER")
    if USE_DOCKER == "True":
        return OllamaEmbeddings(model=MODEL, base_url="http://host.docker.internal:11434")
    else:
        return OllamaEmbeddings(model=MODEL)


# Setup the model
def getModel(MODEL="llama3.2"):
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    USE_DOCKER = os.getenv("USE_DOCKER")
    if USE_DOCKER == "True":
        return Ollama(model=MODEL, base_url="http://host.docker.internal:11434")
    else:
        return Ollama(model=MODEL)


# Setup clearML
def setupClearML():
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
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
