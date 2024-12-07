# See README for more info on how the DataCollectionPipeline works
# The retrieval pipeline is part of the DataCollectionPipeline
import os
import sys
from operator import itemgetter

from clearml import PipelineDecorator
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient

# Setup ClearML
try:
    load_dotenv()
except Exception:
    load_dotenv(sys.path[1] + "/.env")
CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST")
CLEARML_API_HOST = os.getenv("CLEARML_API_HOST")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST")
CLEARML_API_ACCESS_KEY = os.getenv("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = os.getenv("CLEARML_API_SECRETKEY")


# Query expansion(I only generate one additional prompt for simplicity)
@PipelineDecorator.component(cache=False, return_values=["newQuery"])
def queryExpansion(query):
    # Setup the model
    MODEL = "llama3.2"
    model = Ollama(model=MODEL)

    template = """
    Rewrite the prompt. The new prompt must offer a different perspective.
    Do not change the meaning. Output only the rewritten prompt with no introduction.
        Prompt: {prompt}
    """
    prompt = PromptTemplate.from_template(template)
    chain = {"prompt": itemgetter("prompt")} | prompt | model
    return chain.invoke({"prompt": query})


# Self-querying(The metadata I will be generating determines whether to look through the Qdrant collection containing github code)
@PipelineDecorator.component(cache=False, return_values=["codingQuestion"])
def selfQuerying(query):
    # Setup the model
    MODEL = "llama3.2"
    model = Ollama(model=MODEL)

    template = """
    You are an AI assistant. You must determine if the prompt requires code as the answer.
    Output a 1 if it is or a 0 if it is not and nothing else.
        Prompt: {prompt}
    """
    prompt = PromptTemplate.from_template(template)
    chain = {"prompt": itemgetter("prompt")} | prompt | model
    return chain.invoke({"prompt": query})


# Filtered vector search for each of the N=2 queries after expansion
@PipelineDecorator.component(cache=False, return_values=["results1, results2"])
def filteredVectorSearch(query, newQuery, codingQuestion):
    # Create a qdrant connection
    try:
        load_dotenv()
    except Exception:
        load_dotenv(sys.path[1] + "/.env")
    USE_QDRANT_CLOUD = os.getenv("USE_QDRANT_CLOUD")
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_APIKEY = os.getenv("QDRANT_APIKEY")
    if USE_QDRANT_CLOUD:
        qClient = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_APIKEY)
    else:
        qClient = QdrantClient(url=QDRANT_CLOUD_URL)

    # Setup the text embedder
    MODEL = "llama3.2"
    embeddingsModel = OllamaEmbeddings(model=MODEL)

    # Search the related collection
    relatedCollection = "Document"
    if codingQuestion == "1":
        relatedCollection = "Github"
    results1 = qClient.search(
        collection_name=relatedCollection,
        query_vector=embeddingsModel.embed_query(query),
        limit=10,
    )
    results2 = qClient.search(
        collection_name=relatedCollection,
        query_vector=embeddingsModel.embed_query(newQuery),
        limit=10,
    )
    return results1, results2


# Collecting results
@PipelineDecorator.component(cache=False, return_values=["results"])
def collectingResults(results1, results2):
    return results1 + results2


# Reranking(Instead of using a CrossEncoder, I will manually compare embeddings)
@PipelineDecorator.component(cache=False, return_values=["topTexts"])
def reranking(results):
    ids = [result.id for result in results]
    scores = [result.score for result in results]
    topIds = []
    topIndexes = []
    for x in range(3):
        maxScore = 0
        maxIndex = 0
        for i in range(len(ids)):
            if ids[i] not in topIds and scores[i] > maxScore:
                maxScore = scores[i]
                maxIndex = i
        topIds.append(ids[maxIndex])
        topIndexes.append(maxIndex)
    texts = [result.payload["text"] for result in results]
    topTexts = ""
    for index in topIndexes:
        topTexts += texts[index][0]
    return topTexts


# Building prompt
@PipelineDecorator.component(cache=False, return_values=["prompt"])
def buildingPrompt(codingQuestion):
    if codingQuestion == "1":
        template = """
        Write code for the following question given the related coding document below.

        Document: {document}
        Question: {question}
        """
        return PromptTemplate.from_template(template)
    else:
        template = """
        Answer the question based on the document below. If you can't answer the question, reply "I don't know"

        Document: {document}
        Question: {question}
        """
        return PromptTemplate.from_template(template)


# Obtaining answer
@PipelineDecorator.component(cache=False, return_values=["answer"])
def obtainingAnswer(query, prompt, topTexts):
    # Setup the model
    MODEL = "llama3.2"
    model = Ollama(model=MODEL)

    chain = (
        {"document": itemgetter("document"), "question": itemgetter("question")}
        | prompt
        | model
    )
    chain.invoke({"document": topTexts, "question": query})


# Inference Pipeline
@PipelineDecorator.pipeline(
    name="Inference Pipeline",
    project="RAG LLM",
    version="0.1",
)
def main():
    # User query
    query = "What operating system was ROS written for?"
    newQuery = queryExpansion(query)
    codingQuestion = selfQuerying(query)
    results1, results2 = filteredVectorSearch(query, newQuery, codingQuestion)
    results = collectingResults(results1, results2)
    topTexts = reranking(results)
    prompt = buildingPrompt(codingQuestion)
    return obtainingAnswer(query, prompt, topTexts)


if __name__ == "__main__":
    PipelineDecorator.run_locally()
    main()
