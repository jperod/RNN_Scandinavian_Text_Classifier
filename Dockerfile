FROM python:3

MAINTAINER Pedro Rodrigues

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY './train_rnn.py' .
COPY './predict_rnn.py' .
COPY './saves' .

CMD ["python", "app.py"]