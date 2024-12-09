# Make sure ollama serve is running(docker or terminal)
# This is the same code as ClearML
from operator import itemgetter
import gradio as gr
from langchain.prompts import PromptTemplate
from shared import getModel, getEmbeddingsModel, getQdrantClient

def answer(samplePrompt, useSample, Query):
    if useSample:
        query = samplePrompt
    else:
        query = Query
    # Create a qdrant connection
    qClient = getQdrantClient()

    # Setup the text embedder
    embeddingsModel = getEmbeddingsModel()

    # Setup the model
    model = getModel()

    # Retrieval Pipeline
    # Retrieve the chunks with the most similar embeddings from Qdrant
    def retriever(text, collection):
        results = qClient.search(
            collection_name=collection,
            query_vector = embeddingsModel.embed_query(text),
            limit=10
        )
        return results

    # Query expansion(I only generate one additional prompt for simplicity)
    template = """
    Rewrite the prompt. The new prompt must offer a different perspective.
    Do not change the meaning. Output only the rewritten prompt with no introduction.
        Prompt: {prompt}
    """
    prompt = PromptTemplate.from_template(template)
    chain = {"prompt": itemgetter("prompt")} | prompt | model
    queryExpansion = chain.invoke({"prompt": query})

    # Self-querying(The metadata I will be generating determines whether to look through the Qdrant collection containing github code)
    template = """
    You are an AI assistant. You must determine if the prompt requires code as the answer.
    Output a 1 if it is or a 0 if it is not and nothing else.
        Prompt: {prompt}
    """
    prompt = PromptTemplate.from_template(template)
    chain = {"prompt": itemgetter("prompt")} | prompt | model
    codingQuestion = chain.invoke({"prompt": query})

    # Filtered vector search for each of the N queries after expansion
    relatedCollection = 'Document'
    if (codingQuestion == '1'):
        relatedCollection = 'Github'
    results1 = retriever(query, relatedCollection)
    results2 = retriever(queryExpansion, relatedCollection)

    # Collecting results
    results = results1+results2

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
    texts = [result.payload['text'] for result in results]
    links = [result.payload['link'] for result in results]
    topTexts = ''
    for index in topIndexes:
        topTexts += texts[index]

    # Building prompt
    if(codingQuestion == '1'):
        template = """
        Write code for the following question given the related coding document below.

        Document: {document}
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
    else:
        template = """
        You are an AI agent that has retreived a document from the web.
        If the document is useful for answering the question use it.
        If the document is not useful, answer normally.
        Do not mention the document.

        Document: {document}
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

    # Obtaining answer
    chain = {"document": itemgetter("document"), "question": itemgetter("question")} | prompt | model
    return chain.invoke({"document": topTexts, "question": query})


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Dropdown(
            ["How can I develop the navigation stack of an agent with egomotion?",
             "What is ROS?", "How many companies is Nav2 trusted by worldwide?",
             "How would I build a ROS 2 Navigation Framework and System?",
             "Write me code to move a robot using Moveit"], label="Sample Prompt"
        ),
        "checkbox",
        "text",
    ],
    outputs=["text"],
)

demo.launch(share=False)
