FROM python:3.11.9

WORKDIR /gradio-app
COPY . .
RUN pip install --no-cache-dir python-dotenv==1.0.1 langchain==0.3.10 langchain-community==0.3.10 \
    qdrant-client==1.12.1 gradio==5.8.0 pymongo==4.10.1 langchain-openai==0.2.11
    
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
