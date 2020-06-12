FROM python:3.8-buster
RUN pip install torch sentencepiece Keras-Preprocessing
COPY . .
EXPOSE 8001
CMD ["python3", "-u", "server.py"]
