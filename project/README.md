<h1>Installation:</h1>
<h3>Docker setup(easy):<h3>

* Clone the repository from huggingface
* Reopen the repository in a dev container
* Copy the .env.example into a new .env file in the project folder
* If you want to run files in the ClearML folder, fill out the ClearML env variables, otherwise no changes needed.
* Open a shell on the host machine(not the dev container) and navigate to the project folder
* Run "docker compose up -d"
* Run "docker exec -it ollama ollama pull llama3.2"
* Select the python 3.12.7 kernels for the notebooks and run DataCollectionPipeline.ipynb and FeaturePipeline.ipynb(to populate the mongodb and qdrant databases)
* The app is available on localhost:7860

<h3>Non-Docker(web based) setup:<h3>

If for some reason the docker setup does not work try connecting to mongodb, qdrant, ollama, and gradio from the web:
* Clone the repository from huggingface or the entire repository from github
* Reopen the repository in a dev container
* Copy the .env.example into a new .env file in the project folder
* Modify the .env file as instructed in the comments(create accounts for each website)
* Install ollama in the dev container
  * curl -fsSL https://ollama.com/install.sh | sh
* Start up ollama
  * ollama serve
* Download llama3.2(in a new dev container terminal)
  * ollama pull llama3.2
* Select the python 3.12.7 kernels for the notebooks and run DataCollectionPipeline.ipynb and FeaturePipeline.ipynb(to populate the mongodb and qdrant databases)
* Run app.py and click on the link

<h1>Project infrastructure</h1>

Note some files may have similar code with other files, such as the ClearML files containing ipynb files rewritten in python in order to work in ClearML or gradio containing code from InferencePipeline.ipynb. The ipynb file prints output to help see what is happening.

# app.py
Sends a query to the inference pipeline to generate an answer. The DataCollectionPipeline.ipynb and FeaturePipeline.ipynb files must be run first to populate the databases.

# Data Collection Pipeline
The Data Collection pipeline takes as input a list of links to domains. The links are fed into the ETL pipeline which Extracts data from the links using a crawler, Transforms the data into a standardized format, and Loads the extracted data into a NoSQL data warehouse, which in this case is MongoDB. The ETL pipeline uses a different method of extracting and transforming based on the link type. In this project, I classify links as either a github repository or document each with their own crawler and cleaner. This raw data is used by the feature pipeline.

# Feature Pipeline
The Feature pipeline contains the ingestion pipeline.
* The ingestion pipeline extracts documents from MongoDB that were stored by the Data Collection Pipeline. It further cleans the data, breaks it into chunks depending on the data category, passes the chunks through an embedding model to generate embeddings, then loads the embeddings plus their metadata into a vector database, which in this case is Qdrant. The embeddings are passed with additional metadata that contains the document link, type, chunk number, and content.

# Training Pipeline
The training pipeline performs finetuning. I skipped this step since it was not required.

# Inference Pipeline
The inference pipeline contains the retrieval client/pipeline.
* The retreival client takes a prompt as input. It uses the same embedding model as the ingestion pipeline in order to create an embedding for the prompt. It then queries the Qdrant database for the 10 closest embeddings using cosine distance and extracts the text chunk stored in the embeddings' metadata. This returns chunks that are related to the prompt.
* The inference pipeline takes a query as input. It expands the query into N=2 queries using a prompt template, performs self-querying to extract metadata (document type) from the original query, searches the Qdrant for K=10 relevant chunks to each of the N=2 queries plus metadata using the retrieval client, combines the K=10 results from each of the N=2 queries, filters out only the most relevant 3 results, prompts the LLM with the results as context, and generates an answer.

# ClearML
The ClearML folder contains the notebook (.ipynb) pipeline files rewritten to work with ClearML. It is similar code to the notebooks, however ClearML does not print any output but instead logs all output in website. The website stores the pipelines which take input and produces output stored in artifacts. These are the differences between the notebook(.ipynb) pipeline files and the ClearML pipeline files(.py):
* The ClearML Data Collection Pipeline works the same way, running the entire ETL pipeline in a single step (I could not split the ETL pipeline into 3 steps (Extract, Transform, Load) since my list of links gets bigger while looping through it(Since I also goes through some links inside of the websites crawled). Breaking it into steps would require more HTTP requests which would greatly slow down the pipeline).
* The Feature Pipeline breaks down the notebook's loop (from the ingestion pipeline) into 5 stages: retrieve documents, clean documents, chunk documents, embed chunks, and store embeddings.
* The Inference Pipeline simply puts each step in the notebook's version into a function. These functions are query expansion, self-querying, filtered vector search, collecting results, reranking, building prompt, and obtaining answer.

# Tools
The tools folder contains code for viewing/deleting what has been stored in MongoDB and Qdrant

# shared.py
shared.py is in both the project folder and project/Tools folder. It contains functions for setting up the connections with either the docker containers or web services. If you are running into errors connecting to any of the services, consider editing this file or double checking the .env file. Note the ClearML folder hardcodes all functions since it had trouble importing code.
