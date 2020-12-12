FROM python:3

MAINTAINER Pedro Rodrigues

WORKDIR /app

COPY './requirements.txt' .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]