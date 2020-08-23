FROM python:3.8-buster
ENV PYTHONUNBUFFERED 1
RUN pip install torch==1.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install sentencepiece==0.1.91 Keras-Preprocessing==1.1.2 gunicorn==20.0.4
COPY . .
EXPOSE 8000
CMD gunicorn --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread --log-file=- --bind=0.0.0.0:8000 gunicorn_server:get_completion
