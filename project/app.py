# Make sure you have run "ollama serve"
# This is the same code as ClearML
import os
import sys
from operator import itemgetter

import gradio as gr
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient


def answer(samplePrompt, useSample, Query):
    if useSample:
        query = samplePrompt
    else:
        query = Query
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

    # Setup the model
    MODEL = "llama3.2"
    model = Ollama(model=MODEL)

    # Setup the text embedder
    MODEL = "llama3.2"
    embeddingsModel = OllamaEmbeddings(model=MODEL)

    template = """
    Rewrite the prompt. The new prompt must offer a different perspective.
    Do not change the meaning. Output only the rewritten prompt with no introduction.
        Prompt: {prompt}
    """
    prompt = PromptTemplate.from_template(template)
    chain = {"prompt": itemgetter("prompt")} | prompt | model
    newQuery = chain.invoke({"prompt": query})

    # Self-querying(The metadata I will be generating determines whether to look through the Qdrant collection containing github code)

    template = """
    You are an AI assistant. You must determine if the prompt requires code as the answer.
    Output a 1 if it is or a 0 if it is not and nothing else.
        Prompt: {prompt}
    """
    prompt = PromptTemplate.from_template(template)
    chain = {"prompt": itemgetter("prompt")} | prompt | model
    codingQuestion = chain.invoke({"prompt": query})

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

    # Collecting results
    results = results1 + results2

    # Reranking(Instead of using a CrossEncoder, I will manually compare embeddings)
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

    # Building prompt
    if codingQuestion == "1":
        template = """
        Write code for the following question given the related coding document below.

        Document: {document}
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
    else:
        template = """
        Answer the question based on the document below. If you can't answer the question, reply "I don't know"

        Document: {document}
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

    # Obtaining answer
    chain = (
        {"document": itemgetter("document"), "question": itemgetter("question")}
        | prompt
        | model
    )
    return chain.invoke({"document": topTexts, "question": query})


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Dropdown(
            ["What is ROS?", "Write me code to move a robot"], label="Sample Prompt"
        ),
        "checkbox",
        "text",
    ],
    outputs=["text"],
)

demo.launch(share=False)
