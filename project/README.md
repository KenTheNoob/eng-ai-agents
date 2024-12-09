<h3>My Github and Huggingface</h3>

* GitHubID: 32941731
* GitHub username: KenTheNoob
* GitHub link(private): https://github.com/KenTheNoob/eng-ai-agents
* Huggingface username: KenTheNoob
* Huggingface link: https://huggingface.co/KenTheNoob/RAG_LLM

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
  * Note: Consider changing the links in the DataCollectionPipeline file to only the first one if you want to do a quick test, otherwise data collection and featurization will take hours
  * Note: You can use the code in the Tools folder to show what is in the mongo or qdrant database or clear the databases
* The app is available on localhost:7860

<h3>Non-Docker(web based) setup:<h3>

If for some reason the docker setup does not work try connecting to mongodb, qdrant, ollama, and gradio from the web(otherwise ignore this section):
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

Note some files may have similar code with other files, such as the ClearML files containing ipynb files rewritten in python in order to work in ClearML. The ipynb file prints output to help see what is happening unlike the ClearML py files.

# Data Collection Pipeline
The Data Collection pipeline takes as input a list of links to domains. The links are fed into the ETL pipeline which Extracts data from the links using a crawler, Transforms the data into a standardized format, and Loads the extracted data into a NoSQL data warehouse, which in this case is MongoDB. The ETL pipeline uses a different method of extracting and transforming based on the link type. In this project, I classify links as either a github repository or document each with their own crawler and cleaner. This raw data is used by the feature pipeline.

# Feature Pipeline
The Feature pipeline contains the ingestion pipeline.
* The ingestion pipeline extracts documents from MongoDB that were stored by the Data Collection Pipeline. It further cleans the data(remove non-printable characters), breaks it into chunks, passes the chunks through an embedding model to generate embeddings, then loads the embeddings plus their metadata into a vector database, which in this case is Qdrant. The embeddings are passed with additional metadata that contains the document link, type, chunk number, and content.

# Training Pipeline
The training pipeline performs finetuning. I skipped this step since it was not required.

# Inference Pipeline
The inference pipeline contains the retrieval client/pipeline.
* The retreival client takes a prompt as input. It uses the same embedding model as the ingestion pipeline in order to create an embedding for the prompt. It then queries the Qdrant database for the 10 closest embeddings using cosine distance and extracts the text chunk stored in the embeddings' metadata. This returns chunks that are related to the prompt.
* The inference pipeline takes a query as input. It expands the query into N=2 queries using a prompt template, performs self-querying to extract metadata (document type) from the original query, searches the Qdrant for K=10 relevant chunks to each of the N=2 queries plus metadata using the retrieval client, combines the K=10 results from each of the N=2 queries, filters out only the most relevant 3 results, prompts the LLM with the results and query metadata as context, and pipes the prompt into the model to generate an answer.

# app.py
Sends a query to the inference pipeline to generate an answer. The DataCollectionPipeline.ipynb and FeaturePipeline.ipynb files must be run first to populate the databases. Note that the docker compose already runs the app in a docker container. The python file allows you to run the app outside a container if you install gradio. When using the gradio app, you can check the useSample box and select an Sample Prompt from the dropdown menu to run the sample prompts, or uncheck the box and run your own custom query.

# ClearML(optional setup)
The ClearML folder contains the notebook (.ipynb) pipeline files rewritten to work with ClearML. It is similar code to the notebooks, however ClearML does not print any output but instead logs all output in website. The website stores the pipelines which take input and produces output stored in artifacts. These are the differences between the notebook(.ipynb) pipeline files and the ClearML pipeline files(.py):
* The ClearML Data Collection Pipeline works the same way, running the entire ETL pipeline in a single step (I could not split the ETL pipeline into 3 steps (Extract, Transform, Load) since my list of links gets bigger while looping through it(Since I also goes through some links inside of the websites crawled). Breaking it into steps would require more HTTP requests which would greatly slow down the pipeline).
* The Feature Pipeline breaks down the notebook's loop (from the ingestion pipeline) into 5 stages: retrieve documents, clean documents, chunk documents, embed chunks, and store embeddings.
* The Inference Pipeline simply puts each step in the gradio app into a function that is tracked by ClearML. These functions are query expansion, self-querying, filtered vector search, collecting results, reranking, building prompt, and obtaining answer.

# Tools
The tools folder contains code for viewing/deleting what has been stored in MongoDB and Qdrant(very useful for debugging!).
* Tools/mongoTools.ipynb can show the amount of documents in the MongoDB database(which consists of two collections), show the full list of links visited, and the first document in each collection(a sample to show what the data stored in MongoDB looks like). The second cell deletes everything in the mongo database if you want to rerun the DataCollection pipeline with fewer links. The DataCollection pipeline will automatically ignore visited links, but if it takes too long, I suggest using the tool to delete everything, then rerunning the pipeline with only the ROS documentation and github links. Nav2 and moveit are massive sites/repositories to crawl.
* Tools/QdrantTools.ipynb can show the amount of documents in the Qdrant database(which consists of two collections), the first document in each collection(a sample to show what the data stored in Qdrant looks like), and runs a sample search for the closest embeddings/vectors to a sample query(prints out the metadata of the embeddings). Note that the embeddings themselves are not shown because with_vectors=false since normally Qdrant will search for the closest embeddings, but return the payload associated with the embedding(since the embedding itself is useless for generating an answer). The second cell counts how many chunks need to be embedded by the FeaturePipeline and compares it to the total number of chunks from the first cell to give an idea of how close to completion the feature pipeline is(run first cell first). The third cell is an explaination of how Qdrant finds the closest embeddings using cosine distance. The fourth cell allows you to delete everything in the Qdrant database(use with caution!).
* Tools/InferenceTool.ipynb contains the inference pipeline used by the gradio app. It allows you to generate answers to queries without running the gradio app along with printing out useful debugging information for everything that is being fed into the model. This includes the query expansion(reworded query(s)), whether the query is a coding question(self-querying), which Qdrant collection is being searched, the chunks/text being passed as context, the RAG model's answer, and the original model's answer to compare with the RAG model to see if it performed better.

# shared.py
shared.py is in both the project folder and project/Tools folder. It contains functions for setting up the connections with either the docker containers or web services. If you are running into errors connecting to any of the services, consider editing this file or double checking the .env file. Note the ClearML folder hardcodes all functions since it had trouble importing code.
