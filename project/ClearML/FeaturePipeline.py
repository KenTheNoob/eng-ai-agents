# See README for more info on how the FeaturePipeline works
# The Ingestion pipeline is part of the FeaturePipeline
# Make sure to ollama serve before running!
import os
import sys

import pymongo
from clearml import PipelineDecorator
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# Setup ClearML
try:
    load_dotenv(override=True)
except Exception:
    load_dotenv(sys.path[1] + "/.env", override=True)
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST")
CLEARML_API_HOST = os.getenv("CLEARML_API_HOST")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST")
CLEARML_API_ACCESS_KEY = os.getenv("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = os.getenv("CLEARML_API_SECRETKEY")


@PipelineDecorator.component(cache=False, return_values=["links, resultTypes, texts"])
def retreiveDocuments():
    links = []
    resultTypes = []
    texts = []
    # Create a mongoDB connection
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    DATABASE_HOST = os.getenv("DATABASE_HOST")
    mongoHost = pymongo.MongoClient(DATABASE_HOST)
    mongoDatabase = mongoHost["twin"]
    collections = mongoDatabase.list_collection_names()
    for collection in collections:
        mongoCollection = mongoDatabase[collection]
        results = mongoCollection.find()
        for result in results:
            # For each document, split it into chunks
            links.append(result["link"])
            resultTypes.append(result["type"])
            texts.append(result["content"])
    return links, resultTypes, texts


@PipelineDecorator.component(cache=False, return_values=["cleanTexts"])
def cleanDocuments(texts):
    cleanTexts = []
    for text in texts:
        cleanTexts.append("".join(char for char in text if 32 <= ord(char) <= 126))
    return cleanTexts


@PipelineDecorator.component(cache=False, return_values=["chunks", "chunkNums"])
def chunkDocuments(texts):
    chunks = []
    chunkNums = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    for text in texts:
        textChunks = text_splitter.split_text(text)
        chunkNum = 0
        for chunk in textChunks:
            chunks.append(text_splitter.split_text(chunk))
            chunkNums.append(chunkNum)
            chunkNum += 1
    return chunks, chunkNums


@PipelineDecorator.component(cache=False, return_values=["embeddings"])
def embedChunks(chunks):
    # Setup the text embedder
    MODEL = "llama3.2"
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    USE_DOCKER = os.getenv("USE_DOCKER")
    if USE_DOCKER == "True":
        embeddingsModel = OllamaEmbeddings(model=MODEL, base_url="http://host.docker.internal:11434")
    else:
        embeddingsModel = OllamaEmbeddings(model=MODEL)
    return embeddingsModel.embed_documents(chunks)


# Create embeddings for each chunk, of length 3072 using the embedding model
@PipelineDecorator.component(cache=False)
def storeEmbeddings(embeddings, links, resultTypes, chunks, chunkNums):
    # Create a qdrant connection
    try:
        load_dotenv(override=True)
    except Exception:
        load_dotenv(sys.path[1] + "/.env", override=True)
    USE_QDRANT_CLOUD = os.getenv("USE_QDRANT_CLOUD")
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
    if USE_QDRANT_CLOUD:
        qClient = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_APIKEY)
    else:
        qClient = QdrantClient(url=QDRANT_CLOUD_URL)

    # Create qdrant collections to store embeddings
    if not qClient.collection_exists("Github"):
        qClient.create_collection(
            collection_name="Github",
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
    if not qClient.collection_exists("Document"):
        qClient.create_collection(
            collection_name="Document",
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
    chunkIndex = -1
    documentIndex = -1
    for chunkNum in chunkNums:
        chunkIndex += 1
        if chunkNum == 0:
            documentIndex += 1
        # Store the embedding along with some metadata into the Qdrant vector database
        qClient.upsert(
            collection_name=resultTypes[documentIndex],
            wait=True,
            points=[
                PointStruct(
                    id=chunkIndex,
                    vector=embeddings[chunkIndex],
                    payload={
                        "link": links[documentIndex],
                        "type": resultTypes[documentIndex],
                        "chunk": chunkNum,
                        "text": chunks[chunkIndex],
                    },
                )
            ],
        )


# Ingestion Pipeline
@PipelineDecorator.pipeline(
    name="Feature Pipeline",
    project="RAG LLM",
    version="0.3",
)
def main():
    links, resultTypes, texts = retreiveDocuments()
    texts = cleanDocuments(texts)
    chunks, chunkNums = chunkDocuments(texts)
    embeddings = embedChunks(chunks)
    storeEmbeddings(embeddings, links, resultTypes, chunks, chunkNums)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    main()
