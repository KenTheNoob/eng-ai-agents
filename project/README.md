<h1>Installation:</h1>

* Clone the repository, then reopen in docker dev container(in the base folder not project folder)
* Run "docker compose up -d" in the project folder(Only if you want to run Gradio, Qdrant, MongoDB, and Ollama as docker containers instead of connecting to the web)
* Run "docker exec -it ollama ollama pull llama3.2"
* The default MongoDB docker connection string is mongodb://localhost:27017
* The default Qdrant url is localhost:6333
* Make sure you have ollama running (with the llama3.2 model) either in the docker container or in your terminal(accessible via localhost:11434)
* The app is available on localhost:7860

* Add the following environment variables to the project .env file(ClearML is optional)
  * HUGGINGFACE_ACCESS_TOKEN=your_token_here
    * https://huggingface.co/docs/hub/en/security-tokens
  * DATABASE_HOST=your_mongodb_access_string
    * https://www.mongodb.com/resources/products/fundamentals/mongodb-cluster-setup
  * USE_QDRANT_CLOUD=true
  * QDRANT_CLOUR_URL=your_qdrant_cloud_url
    * https://qdrant.tech/documentation/cloud/create-cluster/
  * USE_QDRANT_CLOUD=false
  * QDRANT_APIKEY=your_qdrant_api_key
  * CLEARML_WEB_HOST=your_clearml_web_host
    * https://clear.ml/docs/latest/docs/clearml_serving/clearml_serving_setup/
  * CLEARML_API_HOST=your_clearml_api_host
  * CLEARML_FILES_HOST=yourclearml_files_host
  * CLEARML_API_ACCESS_KEY=your_clearml_api_key
  * CLEARML_API_SECRET_KEY=your_clearml_secret_key
* All docker containers should now be set up and running

* If you wish to not use docker containers(may be easier)
* Install ollama
  * curl -fsSL https://ollama.com/install.sh | sh
* Start up ollama
  * ollama serve
* Download llama3.2
  * ollama pull llama3.2
* Set .env USE_QDRANT_CLOUD=true
* Run app.py

# app.py
Runs sends a query to the inference pipeline.

# Data Collection Pipeline
The Data Collection pipeline takes as input a list of links to domains. The links are fed into the ETL pipeline which Extracts data from the links using a crawler, Transforms the data into a standardized format, and Loads the extracted data into a NoSQL data warehouse, which in this case is MongoDB. The ETL pipeline uses a different method of extracting and transforming based on the link type. In this project, I classify links as either a github repository or document each with their own crawler and cleaner. This raw data is used by the feature pipeline.

# Feature Pipeline
The Feature pipeline contains the ingestion pipeline and retreival pipeline.
* The ingestion pipeline extracts documents from MongoDB that were stored by the Data Collection Pipeline. It further cleans the data, breaks it into chunks depending on the data category, passes the chunks through an embedding model to generate embeddings, then loads the embeddings plus their metadata into a vector database, which in this case is Qdrant. The embeddings are passed with additional metadata that contains the document link, type, and content.
* The retreival pipeline takes a prompt as input. It uses the same embedding model as the ingestion pipeline in order to create an embedding for the prompt. It then queries the Qdrant database for the 4 closest embeddings using cosine distance and extracts the chunk stored in the embeddings' metadata. This returns chunks that are related to the prompt.

# Training Pipeline
The training pipeline performs finetuning. I skipped this step since it was not required.

# Inference Pipeline
The inference pipeline takes a query as input. It expands the query into N queries using a prompt template, performs self-querying to extract metadata from the original query, searches the Qdrant for K relevant chunks to each of the N queries plus metadata using the retrieval client, combines K results from each of the N queries, filters out only the most relevant results, prompts the LLM with the results as context, and generates an answer.

# ClearML
The ClearML folder contains the notebook (.ipynb) pipeline files rewritten to work with ClearML. It is similar code to the notebooks, however ClearML does not print any output but instead logs all output in website. The website stores the pipelines which take input and produces output stored in artifacts. These are the differences between the notebook(.ipynb) pipeline files and the ClearML pipeline files(.py)<br>
* The ClearML Data Collection Pipeline works the same way with the entire ETL pipeline as a single step (I could not split the ETL pipeline into 3 steps (Extract, Transform, Load) since my list of links gets bigger while looping through it(Since it also goes through some links inside of the websites it crawls). Breaking it into steps would require more HTTP requests which would greatly slow down the pipeline).
* The Feature Pipeline breaks down the notebook's loop (from the ingestion pipeline) into 5 stages: retrieve documents, clean documents, chunk documents, embed chunks, and store embeddings.
* The Inference Pipeline simply puts each step in the notebook's version into a function. These functions are query expansion, self-querying, filtered vector search, collecting results, reranking, building prompt, and obtaining answer.

# Tools
The tools folder contains code for viewing/deleting what has been stored in MongoDB and Qdrant
