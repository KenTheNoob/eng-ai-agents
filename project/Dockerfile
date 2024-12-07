FROM python:3.11.9

WORKDIR /gradio-app
COPY . .
RUN pip install --no-cache-dir python-dotenv==1.0.1 && \
    pip install --no-cache-dir langchain==0.3.10 && \
    pip install --no-cache-dir langchain-community==0.3.10 && \
    pip install --no-cache-dir qdrant-client==1.12.1 && \
    pip install --no-cache-dir gradio==5.8.0
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
