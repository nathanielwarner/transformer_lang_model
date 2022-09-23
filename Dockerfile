FROM python:3.10-bullseye
ENV PYTHONUNBUFFERED 1
COPY . .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
EXPOSE $PORT
CMD gunicorn --worker-tmp-dir /dev/shm --workers=2 --threads=4 --worker-class=gthread --log-file=- --bind=0.0.0.0:$PORT -e MODEL_PATH=saved_models/epsilon -e DATASET_PATH=data/leclair_java gunicorn_server:get_completion
